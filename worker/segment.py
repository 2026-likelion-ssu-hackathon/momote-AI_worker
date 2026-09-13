"""대화 분절 — 스트림을 화제 단위로 끊는다.

설계와 실측 근거는 `docs/design.md` 1부. 구조만 옮기면:

    ① 룰 컷     3시간 이상 공백에서 자른다 (LLM 에게 묻지 않는다)
    ② LLM 채점   마지막 조각의 발화마다 "직전 맥락과 얼마나 이어지는가" (호출 1회)
    ③ 룰 판정    임계값으로 붙일지 자를지 결정한다

**LLM 은 경계를 정하지 않는다. 점수만 낸다.** 자르는 판단은 전부 `_should_cut()` 에 있다.
경계가 LLM 안에 있으면 과분절이 나와도 프롬프트를 다시 쓰는 것 말고 할 수 있는 게 없고,
그건 재현 가능한 조정이 아니다. 임계값으로 빼두면 숫자로 만질 수 있고 왜 잘렸는지가
트레이스에 남는다.

**경계 신호로 쓸 수 있는 룰은 시간 공백 하나뿐이다.** 화제 전환 표지어("그건 그렇고"),
어휘 겹침, 임베딩 거리를 전부 재봤고 셋 다 실측에서 무너졌다 (문서 5장).

    표지어    같은 화제인 case9 에 '근데' 가 있고, 진짜 경계엔 표지어가 없었다
    어휘 겹침  단일 화제 안에서도 겹침이 0 이다. case4 는 어휘가 같은데 다른 대화다
    임베딩    경계에서 오히려 거리가 낮았다. 분포가 통째로 겹친다

**룰 컷과 채점이 서로 다른 구멍을 메운다.** LLM 은 화제는 보는데 시간을 못 본다 —
`case4_routine`(사흘치, 공백 1432분)을 한 덩어리로 본다. 룰 컷이 그걸 메운다.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta

from worker.llm import ask, load_prompt
from worker.models import Message, Segment, SegmentLLMOutput, SegmentScore

__all__ = ["SegmentResult", "segment", "active_context", "rule_tail", "clear_score_cache"]

# ① 룰 컷 — 이 이상 침묵하면 채점하지 않고 자른다.
#
# ⚠️ **실측으로 정한 값이 아니다.** 단일 화제 안 최대 공백이 9분이고 경계가 140분 이상이라
# 그 사이 구간에 표본이 없다. 그래서 "틀렸을 때 복구 가능한 쪽"으로 넉넉하게 잡았다 —
# 룰 컷은 LLM 없이 확정이라 잘못 자르면 고칠 기회가 없지만, 관대하면 채점이 고친다.
GAP_HARD = timedelta(hours=3)

# ③ 룰 판정 임계값.
#
# ⚠️ **실측으로 정한 값이 아니다. 프롬프트 앵커에서 역산한 값이다** (문서 11장).
#
# 프롬프트가 100 / 80 / 50 / 20 / 0 다섯 개를 기준점으로 준다. 실측해 보니 모델이
# **거의 기준점 값만 쓴다** — case11 에서 100, 100, 100, 80, 80, 80, 80 이 나왔다.
# 그래서 임계값은 기준점 사이에 놓아야 의미가 있다.
#
#     100 같은 것에 대해 계속       → 무조건 유지
#      80 이어지는 이야기           → **회색.** 이어진다고 한 대화라는 뜻은 아니다
#      50 느슨하게 연결             → 회색
#      20 다른 이야기               → 무조건 자름
#
# 처음에 KEEP_SOFT 를 80 으로 뒀더니 case11 의 진짜 경계(2시간 20분 뒤 다툼 시작)가
# 80 을 받아 통째로 안 잘렸다. "이어지는 이야기"는 유지 근거가 아니라 회색이다.
CUT_HARD = 35    # 이 아래면 무조건 자른다 (앵커 20 을 잡는다)
KEEP_SOFT = 90   # 이 위면 무조건 붙인다 (앵커 100 만 잡는다)
TONE_CUT = 40    # 회색지대에서만 쓰는 보조 기준
GAP_SOFT = timedelta(minutes=30)  # 회색지대에서만 쓰는 보조 기준

# 조각이 이보다 짧으면 채점하지 않는다. 나눌 것이 없다.
MIN_FOR_LLM = 3

# 말투 판정에 최소한 확보해야 하는 메시지 수 (마지막 1개 + `tone.CONTEXT_TURNS` 3턴).
# 활성 세그먼트가 이보다 짧으면 앞 세그먼트에서 뒤에서부터 채운다.
CONTEXT_MIN = 4


@dataclass
class SegmentResult:
    """분절 결과. `scores` 는 트레이스용이고 판정에는 이미 반영돼 있다.

    `cached` / `scored` 는 점수 캐시 계측이다 — 채점 대상 중 몇 개를 캐시에서 가져왔고
    몇 개를 LLM 에 물었는지. 판정에는 관여하지 않고 로그·트레이스에만 쓴다.
    """

    segments: list[Segment] = field(default_factory=list)
    scores: list[SegmentScore] = field(default_factory=list)
    cached: int = 0
    scored: int = 0


# --------------------------------------------------------------------------
# ① 룰 컷 — 시간 공백
# --------------------------------------------------------------------------
def _rule_cut(messages: list[Message]) -> list[list[Message]]:
    """3시간 이상 벌어진 지점에서 자른다. 입력은 시간순으로 정렬돼 있다고 본다."""
    chunks: list[list[Message]] = []
    current: list[Message] = []
    for m in messages:
        if current and m.sent_at - current[-1].sent_at >= GAP_HARD:
            chunks.append(current)
            current = []
        current.append(m)
    if current:
        chunks.append(current)
    return chunks


def rule_tail(messages: list[Message]) -> list[Message]:
    """룰 컷만 적용했을 때의 마지막 조각. **LLM 없이 즉시 나온다.**

    활성 세그먼트는 언제나 이 조각의 꼬리(부분집합)다. 분절 LLM 을 기다리지 않고
    먼저 시작할 수 있는 단계(말투 선행 실행, `router.run`)가 이걸 본다.
    """
    if not messages:
        return []
    ordered = sorted(messages, key=lambda m: (m.sent_at, m.message_id))
    return _rule_cut(ordered)[-1]


# --------------------------------------------------------------------------
# ② LLM 채점
# --------------------------------------------------------------------------
def _lines(messages: list[Message]) -> str:
    """`[HH:MM]` 을 붙인다.

    안 붙이면 **LLM 이 시간 공백을 아예 못 본다.** 2시간 50분이 벌어져도 안 보이고,
    3시간 룰 컷 바로 아래 구간이 통째로 사각지대가 된다 (문서 3-7).
    다른 프롬프트는 `text.format_transcript()` 가 같은 일을 한다.
    """
    return "\n".join(
        f"[{m.message_id}] ({m.sent_at:%m-%d %H:%M}) {m.sender}: {m.content}"
        for m in messages
    )


def _transcript(context: list[Message], targets: list[Message]) -> str:
    """채점 대상과 앞 맥락을 구획으로 나눠 준다. 맥락이 없으면 기존 단일 형식 그대로다."""
    body = f"## 대화 ({len(targets)}개)\n{_lines(targets)}"
    if not context:
        return body
    return (
        f"## 앞 맥락 (채점하지 않는다)\n{_lines(context)}\n\n"
        f"## 채점 대상 ({len(targets)}개)\n{_lines(targets)}"
    )


# 한 호출이 채점하는 최대 발화 수. **출력이 발화 수에 비례해서 지연도 비례한다** —
# 생산 창(메시지 30개)을 한 호출로 채점하면 출력 ~470토큰에 7~8초다.
# 넘으면 배치로 나눠 **동시에** 부른다. 각 배치 호출은 자기 앞의 전체 맥락을 '앞 맥락'
# 구획으로 그대로 보므로 **판단 재료는 한 호출일 때와 같다** — 출력만 나뉜다.
#
# ⚠️ **10 으로 줄여봤다가 되돌렸다** (2026-09-13). 벽시계는 가장 큰 배치의 출력 길이가
# 정하니 줄이면 빨라지는데(창 25개 1.9초 → 1.5초), 캐시 끄고 3회씩 재보니 **배치가 작으면
# 경계를 잃는다** — case19(싸움 → 화해) [10,4] → [14] 3/3, case21 [13,2,4,4,3] → 4·4 가
# 8 로 합쳐짐 3/3. 채점 대상이 짧으면 모델이 낮은 점수를 안 준다. 15 는 그 경계가 유지되는
# 검증된 값이다. 아래 점수 캐시가 붙은 뒤로 이 값은 **방의 첫 요청(콜드)** 에서만
# 의미가 있다 — 그 뒤로는 새 발화 한두 개만 채점해서 배치가 하나다.
SCORE_BATCH = 15

# --------------------------------------------------------------------------
# 점수 캐시 — 같은 발화를 요청마다 다시 채점하지 않는다 (2026-09-13)
# --------------------------------------------------------------------------
# 백엔드는 **메시지 1건마다** 최근 창 30개를 통째로 보낸다. 요청 N 이 발화 1~30 을
# 채점했으면 요청 N+1 (발화 2~31) 에서 새로 온 것은 31 하나인데, 지금까지는 30개를
# 매번 다시 채점했다 — 분절이 맨 앞에 혼자 서서 모든 단계가 기다리는 자리라 그 시간이
# 요청마다 고스란히 붙었다 (창 30개 = 3~4초).
#
# 발화 하나의 점수는 "직전까지의 맥락과 얼마나 이어지는가"라 **같은 발화·같은 직전
# 발화면 재료가 같다.** 그래서 (방, 발화 id) 로 점수를 기억해 두고, 캐시에 없는 발화
# — 대개 마지막 한두 개 — 만 그 앞 전체를 '앞 맥락' 구획으로 붙여 채점한다. 판단
# 재료는 분할 채점과 똑같고(각 발화는 자기 앞 전체를 본다) 출력만 준다.
#
# **워커 무상태 원칙과의 관계.** 이건 상태가 아니라 메모다 — 없으면 전부 다시
# 채점하고 결과 규칙은 하나도 안 바뀐다. 프로세스가 재시작되면 비고, 다른 인스턴스와
# 공유하지 않는다. 부수 효과가 하나 있는데 **좋은 쪽이다**: 같은 발화의 점수가 요청마다
# 흔들리지 않아 세그먼트 경계가 안정된다 — 유튜브·데이트의 "같은 화제엔 하나만" 억제가
# 세그먼트 시작 시각에 기대고 있어서, 경계가 흔들리면 억제가 새던 자리다.
#
# **키에 내용 지문을 넣는다.** 픽스처는 전부 `chatRoomId=1` 에 id 101~ 이 겹치고,
# 방 id 를 서버가 재사용할 수도 있다. (직전 발화 id·내용, 이 발화 id·내용) 의 해시가
# 다르면 다른 발화로 본다 — 같은 방·같은 id 라도 내용이 바뀌면 캐시가 안 맞는다.
#
# 크기는 LRU 로 묶는다. 항목 하나가 수십 바이트라 4,000개면 방 130개 × 창 30개다.
# `KAKAPO_SEGMENT_CACHE=0` 이면 끈다 (회귀 비교·벤치용).
SCORE_CACHE_MAX = int(os.getenv("KAKAPO_SEGMENT_CACHE", "4000"))

# 캐시가 다 맞아도 **마지막 이 개수는 다시 채점한다.**
#
# 새 발화 하나만 채점 대상으로 주면 모델이 관대해진다 — case2(2개짜리 소화제 4개)를
# 메시지 하나씩 늘려가며 돌리면 콜드 [2,2,2,2] 6/6 이 증분에서는 [2,4,2]·[2,6] 으로
# 3/4 무너졌다. 채점 대상이 몇 개 이어져야 앞뒤를 견주어 낮은 점수를 준다. 값은 아래
# 실측으로 정했다 (`docs/refactoring.md`). 출력은 발화당 ~11토큰이라 비용은 거의 없다.
SCORE_TAIL_MIN = int(os.getenv("KAKAPO_SEGMENT_TAIL", "3"))

_score_cache: OrderedDict[tuple[str, int], tuple[str, SegmentScore]] = OrderedDict()
_score_cache_lock = threading.Lock()


def _fingerprint(prev: Message, cur: Message) -> str:
    raw = f"{prev.message_id}\x1f{prev.content}\x1f{cur.message_id}\x1f{cur.content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_get(room: str, message_id: int, fingerprint: str) -> SegmentScore | None:
    with _score_cache_lock:
        hit = _score_cache.get((room, message_id))
        if hit is None or hit[0] != fingerprint:
            return None
        _score_cache.move_to_end((room, message_id))
        return hit[1]


def _cache_put(room: str, message_id: int, fingerprint: str, score: SegmentScore) -> None:
    with _score_cache_lock:
        _score_cache[(room, message_id)] = (fingerprint, score)
        _score_cache.move_to_end((room, message_id))
        while len(_score_cache) > SCORE_CACHE_MAX:
            _score_cache.popitem(last=False)


def clear_score_cache() -> None:
    """테스트·벤치용 — 캐시를 비운다."""
    with _score_cache_lock:
        _score_cache.clear()


def _ask_scores(context: list[Message], targets: list[Message]) -> list[SegmentScore]:
    out = ask(SegmentLLMOutput, load_prompt("segment"), _transcript(context, targets))
    return out.scores


def _score(
    chunk: list[Message], cache_key: str | None = None
) -> tuple[list[SegmentScore] | None, int, int]:
    """발화별 연속성 점수 + (캐시 적중 수, LLM 채점 수). 호출 전부가 실패하면 None → 자르지 않는다.

    **점수가 빠지거나 엉뚱한 id 가 섞여도 통째로 버리지 않는다.** 아는 id 만 남기고
    나머지는 없는 대로 둔다 — 빠진 발화는 `_cut_by_score()` 에서 "안 자름"으로 처리된다.
    배치 호출 하나가 실패해도 같다 — 그 구간만 "안 자름"이 되고 나머지 경계는 산다.

    전량 대조로 하면 실패 반경이 너무 크다. 점수 하나가 어긋났다고 조각 전체를 세그먼트
    1개로 되돌리면 **길게 나눠 놨던 경계가 통째로 사라진다.** 대화가 길수록 어긋날 확률은
    올라가는데 잃는 것도 같이 커진다 — 가장 나쁜 조합이다.

    **캐시 (`cache_key` 가 있을 때).** 앞에서부터 캐시에 있는 발화는 그 점수를 쓰고,
    **첫 미적중부터 끝까지**를 채점 대상으로 삼는다 — 그 앞 전체가 '앞 맥락' 구획이다.
    미적중이 중간에 끼면 그 뒤의 적중분도 같이 다시 채점한다 (채점 대상은 언제나 꼬리
    하나로 이어져야 배치 형식이 그대로다). 새로 받은 점수는 캐시에 넣는다.
    """
    to_score = chunk[1:]  # 첫 발화는 비교할 앞이 없다
    if not to_score:
        return None, 0, 0

    use_cache = cache_key is not None and SCORE_CACHE_MAX > 0
    prints: dict[int, str] = {}
    cached: dict[int, SegmentScore] = {}
    if use_cache:
        for prev, cur in zip(chunk, to_score):
            prints[cur.message_id] = _fingerprint(prev, cur)
            hit = _cache_get(cache_key, cur.message_id, prints[cur.message_id])  # type: ignore[arg-type]
            if hit is not None:
                cached[cur.message_id] = hit

    first_miss = next(
        (i for i, m in enumerate(to_score) if m.message_id not in cached), len(to_score)
    )
    # 꼬리는 캐시가 있어도 다시 채점한다 (SCORE_TAIL_MIN 주석 참조).
    first_miss = min(first_miss, max(0, len(to_score) - SCORE_TAIL_MIN))
    reused = to_score[:first_miss]
    scores: list[SegmentScore] = [cached[m.message_id] for m in reused]

    targets = to_score[first_miss:]
    n_batches = -(-len(targets) // SCORE_BATCH)
    size = -(-len(targets) // n_batches)

    raw: list[SegmentScore] = []
    if n_batches == 1 and first_miss == 0:
        # 캐시 적중이 없고 배치도 하나면 예전 그대로의 단일 형식이다.
        try:
            raw = _ask_scores([], chunk)
        except Exception:  # noqa: BLE001 — 채점 실패는 오류가 아니라 '안 나눔'이다
            return (scores or None), len(scores), len(targets)
    else:
        jobs = [
            (chunk[: 1 + first_miss + i], targets[i : i + size])
            for i in range(0, len(targets), size)
        ]
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(_ask_scores, ctx, tgt) for ctx, tgt in jobs]
            for future in futures:
                try:
                    raw.extend(future.result())
                except Exception:  # noqa: BLE001 — 이 배치만 '안 자름'이 된다
                    continue

    known = {m.message_id for m in targets}
    seen: set[int] = set()
    for s in raw:
        if s.id in known and s.id not in seen:
            seen.add(s.id)
            scores.append(s)
            if use_cache:
                _cache_put(cache_key, s.id, prints[s.id], s)  # type: ignore[arg-type]
    return (scores or None), len(reused), len(targets)


# --------------------------------------------------------------------------
# ③ 룰 판정 — 임계값
# --------------------------------------------------------------------------
def _should_cut(score: SegmentScore, gap: timedelta) -> bool:
    """이 발화 앞에서 자를 것인가.

    회색지대(CUT_HARD ~ KEEP_SOFT)에서는 **붙이는 쪽이 기본값이다.** 잘못 자르면 뒤
    단계가 맥락을 잃지만, 안 자르면 분절 전과 같아질 뿐이다 — 되돌릴 수 있는 실수를 택한다.
    """
    if score.topic >= KEEP_SOFT:
        return False
    if score.topic < CUT_HARD:
        return True

    # 회색지대 — 보조 신호로만 결정한다.
    #
    # ⚠️ `tone_score` 는 **단독으로 자르지 않는다.** case7_tone 은 호칭이 "오빠 → 야"로
    # 바뀌지만 처음부터 끝까지 저녁 약속 얘기 하나다. 말투로 자르면 말투 판정에서 "왜
    # 화가 났는지"(야근으로 약속이 깨짐)가 다른 세그먼트로 넘어가고, 대체 문장이 근거
    # 없는 일반론이 된다 — 말투 교정이 자기 발밑을 판다 (문서 3-5).
    if gap >= GAP_SOFT:
        return True
    return score.tone < TONE_CUT and not score.same


def _cut_by_score(chunk: list[Message], scores: list[SegmentScore]) -> list[Segment]:
    """점수를 id 로 찾아 붙인다. **점수가 없는 발화는 자르지 않는다.**

    순서대로 zip 하지 않는 이유: 점수가 하나라도 빠지면 그 뒤가 전부 한 칸씩 밀려서
    엉뚱한 자리에서 잘린다. id 로 맞추면 빠진 것만 조용히 넘어간다.
    """
    by_id = {s.id: s for s in scores}
    segments: list[Segment] = []
    current: list[Message] = [chunk[0]]

    for message in chunk[1:]:
        score = by_id.get(message.message_id)
        gap = message.sent_at - current[-1].sent_at
        if score is not None and _should_cut(score, gap):
            segments.append(Segment(messages=current, by_rule=False))
            current = [message]
        else:
            current.append(message)

    segments.append(Segment(messages=current, by_rule=False))
    return segments


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------
def _whole(messages: list[Message]) -> list[Segment]:
    """조각 전체를 세그먼트 1개로. 폴백이자 '나눌 게 없음' 경로다."""
    return [Segment(messages=messages, by_rule=True)] if messages else []


def segment(messages: list[Message], cache_key: str | None = None) -> SegmentResult:
    """스트림을 세그먼트 목록으로. 마지막 원소가 **활성 세그먼트**다.

    `cache_key` 는 점수 캐시의 방 식별자다 (보통 `chatRoomId`). None 이면 캐시를 안 쓴다.
    """
    if not messages:
        return SegmentResult()

    ordered = sorted(messages, key=lambda m: (m.sent_at, m.message_id))
    chunks = _rule_cut(ordered)

    segments: list[Segment] = []
    for chunk in chunks[:-1]:
        segments.extend(_whole(chunk))

    last = chunks[-1]
    if len(last) < MIN_FOR_LLM:
        return SegmentResult(segments=segments + _whole(last))

    scores, cached, scored = _score(last, cache_key)
    if scores is None:
        # 폴백 = 점수 전부 100 = 안 자름 = 분절 전 동작. 더 나빠지지 않는다.
        return SegmentResult(segments=segments + _whole(last), cached=cached, scored=scored)

    return SegmentResult(
        segments=segments + _cut_by_score(last, scores),
        scores=scores, cached=cached, scored=scored,
    )


def active_context(segments: list[Segment]) -> list[Message]:
    """말투 판정 프롬프트에 넣을 맥락.

    **게이트·트리거·RAG 는 활성 세그먼트만 쓴다.** 이건 말투 판정 전용이다 —
    활성 세그먼트가 1개짜리면 "와 미친 ㅋㅋ" 가 장난인지 갈등인지 구분할 수가 없다.
    부족한 만큼만 앞 세그먼트에서 뒤에서부터 끌어온다.
    """
    if not segments:
        return []

    messages = list(segments[-1].messages)
    for seg in reversed(segments[:-1]):
        if len(messages) >= CONTEXT_MIN:
            break
        need = CONTEXT_MIN - len(messages)
        messages = seg.messages[-need:] + messages
    return messages

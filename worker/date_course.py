"""후보 기능 — 데이트 코스 추천.

대화에서 데이트 계획 의도가 감지되면, **두 사람이 과거에 남긴 발화**(먹고 싶다 · 가보고
싶다 · 저장한 링크)를 최우선 근거로 장소를 구성해 카드로 제시한다.

    [룰 트리거] → [기억 검색] → [LLM 계획] → [카카오 로컬 검색] → [LLM 이유 문구]

**LLM 과 카카오의 역할이 엄격히 갈린다.**

    LLM  → 무엇을 찾을지 (지역, 검색어, 코스 의도, 추천 이유)
    카카오 → 실제로 무엇이 있는지 (상호명, 카테고리, place_url)

LLM 이 상호명을 만들면 존재하지 않는 가게가 나온다. 규격서 9장의 `externalUrl` 은 필수
필드라 환각이 그대로 죽은 링크로 나간다. 그래서 장소 이름은 **카카오가 준 것만** 쓴다.

명세의 적합도 스코어링 5가지 중 구현한 것은 ①과거 기억 ②현재 스트림 ⑤취향 셋이다.
③영업시간은 카카오 로컬 API 가 주지 않고(크롤링 영역), ④날씨·예산도 데이터 소스가 없다.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from worker.copy import DATE_GUIDE
from worker.llm import ask, load_prompt
from worker.models import (
    AiResult,
    DateCourseResultData,
    DateGateResult,
    DateIntentKind,
    DatePlanLLMOutput,
    DateReasonLLMOutput,
    Memory,
    Message,
    Place,
    to_key,
)
from worker.places import KakaoPlace, region_center, search_places
from worker.text import format_transcript

# 코스 구성 — **밥 → 구경 → 카페 세 자리로 고정이다** (2026-08-17 결정).
#
# 검색어만 순서대로 받아 앞에서부터 채웠더니 **카페가 두 곳 나오는 코스**가 만들어졌다
# (`case9_date` 실측 — `브런치` 검색이 카페를 물어와 카페 → 카페 → 서울숲). 검색어에는
# 업종이 담겨 있어도 카카오가 그 업종으로 준다는 보장이 없다.
#
# 그래서 자리마다 **허용 카테고리를 강제한다.** 검색 결과에서 그 카테고리가 아닌 것은
# 아예 후보에서 뺀다. 자리 수가 곧 코스 길이라 2곳/3곳이 섞이지도 않는다.
#
#     (키, 화면에서 부르는 이름, 허용 카테고리)
SLOTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("meal", "밥", ("RESTAURANT",)),
    ("sight", "구경", ("CULTURE", "ATTRACTION", "SHOP", "ACTIVITY")),
    ("cafe", "카페", ("CAFE",)),
)

COURSE_PLACES = len(SLOTS)

# 한 자리에 받는 검색어 수. **2개인 이유는 0건 대비다** —
# 카카오는 수식어가 붙거나 그 지역에 그 업종이 없으면 빈 목록을 준다.
#
# **파이썬이 대체 검색어를 지어내지 않는다.** 그러면 "무엇을 찾을지는 LLM" 이라는 역할
# 구분이 깨진다. 예비까지 LLM 에게 받고, 파이썬은 순서대로 시도하기만 한다.
QUERIES_PER_SLOT = 2

# 기억 저장소는 파킹됐다 (2026-09-13, `parked/memory/`). `memories` 인자는 되살릴 때
# 그대로 쓰려고 남겨 두고, 지금은 항상 빈 목록이 들어온다 — 프롬프트는 "기억 없음"을 처리한다.


# --------------------------------------------------------------------------
# 룰 트리거 — 명세 [데이트 의도 감지] 4종
# --------------------------------------------------------------------------
# ① 명시적 계획 질문
#
# "데이트할까/하자/갈까"는 가장 정석적인 제안 문장인데 빠져 있었다 (2026-08-20 발견).
# 배포에서는 시연 강제 트리거('데이트' 낱말)가 대신 잡아줘서 안 드러났다 — 데모 모드를
# 끄면 "성수에서 데이트할까?"로 코스가 안 뜨는 상태였다. `case20_date_fight` 가 이
# 문구로 게이트를 지나 격앙 보류까지 가는 경로를 고정한다.
_PLAN_RE = re.compile(
    r"(어디\s*(갈까|가지|가면|서\s*볼까|서\s*만날)|뭐\s*(먹지|먹을까|하지|할까|해먹)|"
    r"데이트\s*(어디|뭐|코스|할까|하자|갈까|가자|어때)|오늘\s*뭐\s*(하|할)|주말에\s*뭐|만나서\s*뭐|"
    r"뭐\s*하고\s*싶|어디\s*좋을까|계획\s*(짤|세울|있어))"
)

# ② 일정 확정 발화
_SCHEDULE_RE = re.compile(
    r"((월|화|수|목|금|토|일)요일에?\s*(보자|만나|볼까|봐)|"
    r"(내일|모레|이번\s*주|다음\s*주|주말|담주)에?\s*(보자|만나|볼까|봐|시간)|"
    r"몇\s*시에?\s*(봐|보자|만나)|시간\s*(돼|되|있어)\?*|"
    r"\d{1,2}일에?\s*(보자|만나|볼까|어때))"
)

# ③ 장소·음식 카테고리 언급
_CATEGORY_RE = re.compile(
    r"(배고파|배고프|맛집|카페|전시|영화|산책|드라이브|공원|방탈출|팝업|브런치|"
    r"[가-힣]+\s*(먹고\s*싶|마시고\s*싶|가보고\s*싶|해보고\s*싶|가고\s*싶)|"
    r"먹고\s*싶|가보고\s*싶|해보고\s*싶|땡긴다|땡겨)"
)

# ④ 이전 대화 기억의 재언급
_RECALL_RE = re.compile(
    r"(저번에\s*그|지난번에?\s*그|우리\s*가보자던|가보자고\s*했던|"
    r"네가\s*말한\s*그|니가\s*말한\s*그|먹고\s*싶다고\s*했던|가고\s*싶다고\s*했던|"
    r"그때\s*그|말했던\s*(그|데|곳))"
)

_PATTERNS: list[tuple[DateIntentKind, re.Pattern[str], str]] = [
    ("plan_question", _PLAN_RE, "명시적 계획 질문"),
    ("schedule_fixed", _SCHEDULE_RE, "일정 확정 발화"),
    ("category", _CATEGORY_RE, "장소·음식 카테고리 언급"),
    ("recall", _RECALL_RE, "이전 대화 기억의 재언급"),
]


def check_date_gate(messages: list[Message]) -> DateGateResult:
    """데이트 의도가 있는가. 확정하지 않고 후보만 잡는다 — 확정은 LLM 이 한다.

    **활성 세그먼트를 받는다.** 예전에는 최근 12개를 스스로 잘랐는데, 그 12개 안에 몇 개의
    화제가 들어 있는지 알 수 없어서 두 시간 전에 끝난 데이트 얘기가 지금 싸우는 요청에서
    코스를 발동시켰다 (`docs/design.md` 1부 1장). 범위는 이제 분절이 정한다.
    """
    if not messages:
        return DateGateResult(triggered=False)

    recent = sorted(messages, key=lambda m: m.sent_at)
    kinds: list[DateIntentKind] = []
    details: list[str] = []
    hit_ids: list[int] = []

    for kind, pattern, label in _PATTERNS:
        for m in recent:
            hit = pattern.search(m.content)
            if hit is None:
                continue
            if kind not in kinds:
                kinds.append(kind)
                details.append(f"[{kind}] {label} — '{hit.group(0).strip()}'")
            if m.message_id not in hit_ids:
                hit_ids.append(m.message_id)
            break

    return DateGateResult(
        triggered=bool(kinds),
        kinds=kinds,
        detail=" / ".join(details) or None,
        message_ids=sorted(hit_ids),
    )


# --------------------------------------------------------------------------
# LLM — 무엇을 찾을지 정한다
# --------------------------------------------------------------------------
def _memory_block(memories: list[Memory]) -> str:
    if not memories:
        return "(없음 — 기억이 비어 있다)"
    lines = []
    for m in memories:
        when = m.occurred_at.strftime("%Y-%m-%d") if m.occurred_at else "시점 미상"
        lines.append(f'- [{m.kind}] {m.content} ({when}) ← "{m.source_quote}"')
    return "\n".join(lines)


def plan_date(
    messages: list[Message], memories: list[Memory], gate: DateGateResult
) -> DatePlanLLMOutput:
    body = "\n".join(
        [
            "## 최근 대화",
            format_transcript(messages),
            "",
            "## 두 사람이 과거에 남긴 발화 (기억 저장소)",
            _memory_block(memories),
            "",
            "## 룰이 잡은 데이트 의도 신호",
            gate.detail or "(없음)",
        ]
    )
    return ask(DatePlanLLMOutput, load_prompt("date_plan"), body)


# --------------------------------------------------------------------------
# 카카오 로컬 — 실재하는 장소로 코스를 채운다
# --------------------------------------------------------------------------
_QUERY_TOKEN_RE = re.compile(r"[가-힣A-Za-z]+")


def _intent_tokens(query: str) -> list[str]:
    """검색어에서 업종 대조에 쓸 조각들.

    한국어 합성어는 뒤쪽이 업종이다 — `평양냉면`의 업종은 `냉면`, `크림파스타`는 `파스타`.
    카카오 카테고리는 `음식점 > 한식 > 냉면` 이라 뒷조각으로 대조해야 걸린다.
    """
    tokens: list[str] = []
    for word in _QUERY_TOKEN_RE.findall(query):
        if len(word) < 2:
            continue
        tokens.append(word)
        # 합성어 뒤쪽 2~3글자를 업종 후보로 함께 본다
        for size in (3, 2):
            if len(word) > size:
                tokens.append(word[-size:])
    return tokens


def _fits_intent(place: KakaoPlace, query: str) -> bool:
    """검색 의도와 업종이 맞는가.

    카카오는 유사 매칭이 후해서 `평양냉면` 으로 검색해도 고기집이 1위로 온다.
    그대로 쓰면 "먹어보고 싶다던 평양냉면집" 이라는 **근거가 통째로 날아간다** —
    추천 이유가 이 기능의 전부인데 그게 무너진다.
    """
    haystack = f"{place.name} {place.category_name}"
    return any(t in haystack for t in _intent_tokens(query))


def _choose(
    found: list[KakaoPlace], query: str, taken: set[str], allowed: tuple[str, ...] = ()
) -> KakaoPlace | None:
    """검색 결과에서 한 곳을 고른다. 이미 쓴 상호와 **자리에 안 맞는 카테고리**는 뺀다.

    `allowed` 가 비어 있으면 카테고리를 보지 않는다 (슬롯 밖에서 쓸 때).
    """
    fresh = [
        p for p in found
        if p.name not in taken and (not allowed or p.category in allowed)
    ]
    if not fresh:
        return None
    return next((p for p in fresh if _fits_intent(p, query)), fresh[0])


def slot_queries(plan: DatePlanLLMOutput) -> list[tuple[str, tuple[str, ...], list[str]]]:
    """`(자리 이름, 허용 카테고리, 검색어들)` — 자리 순서대로."""
    by_key = {
        "meal": plan.meal_queries,
        "sight": plan.sight_queries,
        "cafe": plan.cafe_queries,
    }
    return [
        (label, allowed, [q.strip() for q in by_key[key][:QUERIES_PER_SLOT] if q.strip()])
        for key, label, allowed in SLOTS
    ]


def _fill_slot(
    slot_found: list[list[KakaoPlace]],
    queries: list[str],
    allowed: tuple[str, ...],
    taken: set[str],
) -> KakaoPlace | None:
    """한 자리를 채운다. 첫 검색어가 비면 예비 검색어로 넘어간다."""
    for found, query in zip(slot_found, queries):
        picked = _choose(found, query, taken, allowed)
        if picked is not None:
            return picked
    return None


def _search_all(
    queries: list[str], region: str | None, center: tuple[str, str] | None
) -> list[list[KakaoPlace]]:
    """검색어들을 **동시에** 던진다. 순서는 입력 순서를 지킨다.

    카카오 호출은 서로 독립이라 순차로 돌 이유가 없었다 — 실측 0.75초 → 0.20초.
    """
    if len(queries) == 1:
        return [search_places(queries[0], region=region, center=center)]
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        return list(pool.map(
            lambda q: search_places(q, region=region, center=center), queries
        ))


def build_course(plan: DatePlanLLMOutput, region: str | None) -> list[KakaoPlace]:
    """**밥 → 구경 → 카페** 세 자리를 채운다. 자리마다 카테고리를 강제한다.

    자리 하나에 검색어가 2개 있고, 첫 검색어가 0건이거나 카테고리가 안 맞으면 두 번째를
    쓴다. 검색어 의도에 맞는 결과를 우선하고, 하나도 없으면 (카테고리가 맞는 것 중) 1위로
    물러선다 — `카페` 처럼 업종명이 상호에 안 뜨는 검색어가 있다.

    **지역명을 못 잡았을 때는 첫 장소(밥)를 코스의 중심으로 삼는다.** 안 그러면 자리마다
    전국에서 독립적으로 고르게 되고, 실측에서 `영화관 / 필름 현상 / 카페` 가
    **용산 → 미상 → 남양주 북한강**으로 흩어졌다. 차로 한 시간 넘는 곳들이라 코스가 아니다.
    `search_places` 의 반경 방어는 지역명이 있을 때만 걸리므로, 없을 때는 여기서 건다.

    한 자리라도 못 채우면 **짧은 코스를 만들지 않고 그대로 돌려준다** — 호출부가 길이를
    보고 미발동시킨다. 억지로 먼 곳이나 엉뚱한 업종을 끼워 넣으면 코스가 아니게 된다.
    """
    slots = slot_queries(plan)
    if not any(queries for _, _, queries in slots):
        return []

    course: list[KakaoPlace] = []
    taken: set[str] = set()

    def _search(targets, center):
        """자리별 검색어를 **한꺼번에** 던지고 자리별로 다시 묶는다."""
        flat = [q for _, _, queries in targets for q in queries]
        if not flat:
            return []
        found = _search_all(flat, region if center is None else None, center)
        out, i = [], 0
        for _, _, queries in targets:
            out.append(found[i:i + len(queries)])
            i += len(queries)
        return out

    def _take(slot, slot_found) -> None:
        _, allowed, queries = slot
        picked = _fill_slot(slot_found, queries, allowed, taken)
        if picked is not None:
            course.append(picked)
            taken.add(picked.name)

    # 지역명이 있으면 자리끼리 의존이 없다 — 전부 동시에 던진다.
    # 좌표 조회를 미리 한 번 돌려 `lru_cache` 를 데운다. 안 그러면 병렬 호출이 같은 지역
    # 조회를 동시에 중복해서 날린다.
    if region is not None:
        region_center(region)
        for slot, slot_found in zip(slots, _search(slots, None)):
            _take(slot, slot_found)
        return course

    # 지역명이 없으면 **첫 자리(밥)가 코스의 중심**이라 그것만 먼저 확정해야 한다.
    head_slot = slots[0]
    head_found = [search_places(q, region=None) for q in head_slot[2]]
    _take(head_slot, head_found)
    if not course:
        return []  # 중심을 못 잡으면 뒤 자리가 전국에서 흩어진다

    anchor = (course[0].x, course[0].y) if course[0].x and course[0].y else None
    rest = list(slots[1:])
    for slot, slot_found in zip(rest, _search(rest, anchor)):
        _take(slot, slot_found)
    return course


# 근거 문구에 **묻는 말이 그대로 남았는가** — `date_reason.md` 6번 규칙의 자동 점검.
#
# `성수 쪽은 어때?` 를 그대로 인용하면 "성수 쪽은 어때라고 하신 말" 이 화면에 나간다.
# 실제로 나왔던 출력이라 회귀를 잡으려고 둔다.
#
# ⚠️ **걸려도 재생성하지 않는다.** 원인이 `write_reason` 이 아니라 그 앞의 `reason_seed`
# 라서, 같은 seed 로 다시 돌리면 또 걸린다 — 실측에서 6번 중 5번 재현됐다. 2초를 쓰고
# 고쳐지지 않는다. 고칠 자리는 `date_plan.md` 의 reason_seed 규칙이다 (거기서 잡았다).
# 여기서는 트레이스에 남겨 다음에 규칙이 무너지면 눈에 띄게만 한다.
_ASK_QUOTE = re.compile(
    r"(\?|(어때|어떨까|어떠|갈까|할까|먹을까|뭐\s*먹지|뭐\s*하지|뭐\s*할까|어디\s*갈)"
    r"\s*(라고|라는|고|이라고)?\s*(하[신셨]|했다|한다는|말)?)"
)


def question_quote(text: str) -> str | None:
    """근거 문구에 남은 묻는 말. 없으면 None."""
    hit = _ASK_QUOTE.search(text)
    return hit.group(0).strip() if hit else None


def write_reason(
    messages: list[Message],
    memories: list[Memory],
    plan: DatePlanLLMOutput,
    course: list[KakaoPlace],
) -> DateReasonLLMOutput:
    """확정된 장소를 보고 코스명·요약·추천 이유·장소 설명을 쓴다.

    계획 단계와 나눈 이유: 계획 시점에는 어떤 가게가 잡힐지 모른다. 카카오가 준 실제
    상호를 보고 나서 문구를 써야 "성수다락에서 브런치 먹고" 같은 문장이 나온다.
    """
    place_lines = "\n".join(
        f"{i}. {p.name} — {p.category_name} ({p.address})" for i, p in enumerate(course, 1)
    )
    body = "\n".join(
        [
            "## 최근 대화",
            format_transcript(messages),
            "",
            "## 근거로 삼은 기억",
            _memory_block(memories),
            "",
            "## 계획 단계에서 정한 방향",
            f"- 근거: {plan.reason_seed}",
            "",
            "## 확정된 장소 (카카오 로컬 검색 결과 — 이름을 바꾸지 말 것)",
            place_lines,
        ]
    )
    return ask(DateReasonLLMOutput, load_prompt("date_reason"), body)


# --------------------------------------------------------------------------
# 조립
# --------------------------------------------------------------------------
def to_result(
    plan: DatePlanLLMOutput,
    reason: DateReasonLLMOutput,
    course: list[KakaoPlace],
    trigger_message_ids: list[int],
) -> AiResult:
    summaries = list(reason.place_summaries) + [""] * len(course)

    places = [
        Place(
            order=i,
            name=p.name,
            category=p.category,
            summary=(summaries[i - 1] or p.category_name).strip(),
            external_url=p.url,
        )
        for i, p in enumerate(course, 1)
    ]

    # mainPlace 는 코스의 첫 장소다. 규격서상 order 가 없어야 해서 따로 만든다.
    head = places[0]
    main = Place(
        name=head.name,
        category=head.category,
        summary=head.summary,
        external_url=head.external_url,
    )

    individual = plan.scope == "individual" and plan.target in ("A", "B")
    return AiResult(
        result_type="DATE_RECOMMENDATION",
        visibility_type="INDIVIDUAL" if individual else "COUPLE",
        target_participant=to_key(plan.target) if individual else None,  # type: ignore[arg-type]
        content_type="MIXED",
        trigger_message_ids=trigger_message_ids,
        result_data=DateCourseResultData(
            guide_message=DATE_GUIDE,
            course_name=reason.course_name.strip(),
            course_summary=reason.course_summary.strip(),
            main_place=main,
            course_places=places,
            recommendation_reason=reason.recommendation_reason.strip(),
        ),
    )

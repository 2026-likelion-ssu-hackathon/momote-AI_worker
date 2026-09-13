"""개인별 평소 말투 기준선.

말투 교정은 **절대 기준으로 판정하면 안 된다.** 평소 "ㅇㅇ"을 자주 쓰는 커플에게 "ㅇㅇ"은
무례가 아니다. 특정 단어가 아니라 **그 사람의 평소 대비 변화량**을 본다.

기준선은 두 곳에서 온다. **앞에 있는 것이 이긴다.**

1. 요청의 `speakerProfiles` — 서버가 저장·집계해서 실어주는 값 (2026-08-17 합의, 미전송)
2. `GENERIC_PROFILE` — **평균적인 사람의 카톡 말투** (2026-09-13 사용자 결정)

**들어온 대화에서 직접 계산하지 않는다.** 요청에는 최근 30개만 오는데 그걸로 "평소"를
계산하면 방금 화나서 보낸 메시지가 기준선에 섞여서 변화량이 안 잡힌다.

시연 커플 한 쌍의 값이던 시드(`parked/speaker_profiles.json`)는 방이 여러 개가 되면서
뺐다 — 모든 방이 그 커플의 기준선으로 판정되고 있었다. 하루 시연이라 방별 누적이나
서버 집계 대신 **공통 기준선 하나**로 간다. 값의 근거는 `GENERIC_PROFILE` 주석.
"""

from __future__ import annotations

import re
from collections import Counter

from worker.models import (
    Message,
    Speaker,
    SpeakerProfile,
    SpeakerProfileInput,
    to_speaker,
)
from worker.text import norm_len as _norm_len

# 평균적인 사람의 카톡 말투 — **모든 방·모든 화자에 같은 기준선**을 쓴다 (2026-09-13).
#
# 두 참고치 사이에서 잡았다. 픽스처 21종 156발화(우리가 커플 채팅을 흉내 내 쓴 것)는
# 평균 12자 · 마침표 종결 2% · 메시지당 ㅋ/ㅎ 0.38개 · 이모지 0% 였고, 시연 커플 시드는
# 18~21자 · 3~5% · 2.0~2.8개 · 20~35% 였다. 외부 평가셋(MSD)은 존댓말·마침표 종결의
# 격식 대화라 카톡의 "평균"으로 쓸 수 없었다.
#
# 게이트 임계와의 관계 — 값을 고칠 때 `tone.py` 의 임계를 같이 볼 것:
#   period_rate 0.05 < LOW_PERIOD_RATE 0.15  → 마침표 종결("됐어.")은 급변 신호로 센다
#   laugh_per_msg 2.0 ≥ LAUGH_BASELINE 1.5    → ㅋ/ㅎ 이 사라진 진지한 메시지는 신호 하나
#   emoji_rate 0.15 < EMOJI_BASELINE 0.3      → 이모지 부재는 세지 않는다 (안 쓰는 사람이 많다)
#   avg_length 14, SHORT_RATIO 0.4            → 공백 제외 5자 이하의 비동의 단답이 "짧아짐"
#   top_address 비움                          → 거친 호칭(야·너·니가…)은 전부 "평소와 다름"
GENERIC_PROFILE = SpeakerProfile(
    speaker="A",  # 자리 채움 — resolve_profile 이 화자에 맞춰 바꾼다
    avg_length=14.0,
    period_rate=0.05,
    laugh_per_msg=2.0,
    emoji_rate=0.15,
    top_address=[],
)

# 평소 호칭 후보. 상위 2개를 기준선으로 잡고, 여기서 벗어나면 호칭 변화로 본다.
ADDRESS_TERMS = [
    "오빠", "언니", "누나", "형", "자기야", "자기", "여보", "애기야", "애기",
    "야", "너", "니가", "네가", "당신", "그쪽",
]

# 공격적으로 읽히기 쉬운 호칭
HARSH_ADDRESS = {"야", "너", "니가", "네가", "당신", "그쪽"}

_LAUGH_RE = re.compile(r"[ㅋㅎ]")
_EMOJI_RE = re.compile(r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF]")
_PERIOD_END_RE = re.compile(r"[^.]\.\s*$")  # "..." 은 제외, 단일 마침표 종결만

# 호칭 뒤에 붙을 수 있는 조사. 이것만 허용해서 "너무"가 "너"로, "거야"가 "야"로 잡히지 않게 한다.
_PARTICLES = {"", "는", "가", "도", "의", "한테", "랑", "이랑", "와", "과", "을", "를", "만", "은"}

_WORD_RE = re.compile(r"[가-힣]+")


def addresses_in(text: str) -> list[str]:
    """문장에 등장한 호칭.

    부분 문자열로 찾으면 "거야"의 '야', "너무"의 '너'까지 호칭으로 잡힌다.
    어절 단위로 보고 뒤에 조사만 붙은 경우까지만 인정한다.
    긴 것부터 대조해 '자기야'가 '자기'로 잘리지 않게 한다.
    """
    terms = sorted(ADDRESS_TERMS, key=len, reverse=True)
    found: list[str] = []
    for token in _WORD_RE.findall(text):
        for term in terms:
            if token.startswith(term) and token[len(term):] in _PARTICLES:
                if term not in found:
                    found.append(term)
                break
    return found


def compute_profile(messages: list[Message], speaker: Speaker) -> SpeakerProfile:
    """주어진 대화에서 한 사람의 말투 기준선을 계산한다.

    파이프라인은 안 쓴다 (모듈 설명 참조). 서버가 `speakerProfiles` 를 만들 때 산식을
    맞추는 기준이자, 시드·공통 기준선 값을 잴 때의 도구로 남긴다.
    """
    mine = [m for m in messages if m.sender == speaker]
    if not mine:
        return SpeakerProfile(speaker=speaker)

    n = len(mine)
    address_counter: Counter[str] = Counter()
    for m in mine:
        address_counter.update(addresses_in(m.content))

    return SpeakerProfile(
        speaker=speaker,
        avg_length=sum(_norm_len(m.content) for m in mine) / n,
        period_rate=sum(bool(_PERIOD_END_RE.search(m.content)) for m in mine) / n,
        laugh_per_msg=sum(len(_LAUGH_RE.findall(m.content)) for m in mine) / n,
        emoji_rate=sum(bool(_EMOJI_RE.search(m.content)) for m in mine) / n,
        top_address=[t for t, _ in address_counter.most_common(2)],
    )


def from_request(given: SpeakerProfileInput) -> SpeakerProfile:
    """요청으로 들어온 기준선을 내부 스키마로. 경계에서만 `USER_A` → `A` 로 바꾼다."""
    return SpeakerProfile(
        speaker=to_speaker(given.participant_key),
        avg_length=given.avg_length,
        period_rate=given.period_rate,
        laugh_per_msg=given.laugh_per_msg,
        emoji_rate=given.emoji_rate,
        top_address=list(given.top_address),
    )


def resolve_profile(
    speaker: Speaker,
    messages: list[Message],
    given: list[SpeakerProfileInput] | None = None,
) -> SpeakerProfile:
    """요청 값 > 공통 기준선. 앞에 있는 것이 이긴다 (모듈 설명 참조).

    `messages` 는 더 이상 쓰지 않는다 — 창 30개로 "평소"를 계산하면 방금 화난 메시지가
    섞인다. 시그니처는 호출부 호환으로 남긴다.
    """
    for item in given or []:
        if to_speaker(item.participant_key) == speaker:
            return from_request(item)
    return GENERIC_PROFILE.model_copy(update={"speaker": speaker})


def describe(profile: SpeakerProfile) -> str:
    """LLM 프롬프트에 넣을 기준선 요약."""
    lines = [
        f"평소 평균 길이: {profile.avg_length:.0f}자",
        f"마침표로 끝내는 비율: {profile.period_rate:.0%}",
        f"메시지당 ㅋ/ㅎ 개수: {profile.laugh_per_msg:.1f}개",
        f"이모지 사용 비율: {profile.emoji_rate:.0%}",
        f"평소 호칭: {', '.join(profile.top_address) or '없음'}",
    ]
    if profile.conflict_style:
        lines.append(f"화났을 때의 평소 패턴: {profile.conflict_style}")
    return "\n".join(f"- {line}" for line in lines)

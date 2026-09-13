"""LLM 접근 레이어.

모든 LLM 호출은 여기를 거친다. `import openai` 직접 호출은 금지 — 트레이싱 일관성 때문이다.
후보 3종과 기억 추출이 모델 설정·토큰 계량을 공유하도록 한 곳에 모아둔다.
지금 이 모듈을 거치는 호출은 7종이다 (`worker/prompts/*.md` 와 1:1).
"""

from __future__ import annotations

import os
import threading
import time
from functools import lru_cache
from typing import NamedTuple, TypeVar

from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from worker import PROMPT_DIR

T = TypeVar("T", bound=BaseModel)

# gpt-5 는 케이스 1건에 2분씩 걸려서 픽스처를 돌려보며 고치는 속도가 안 나왔다.
# 판정 품질보다 반복 횟수가 중요한 단계라 응답이 빠른 모델로 내렸다.
DEFAULT_MODEL = "openai:gpt-4.1-mini"

# **단계별 모델.** 스키마 이름 → 모델. 없으면 `DEFAULT_MODEL`.
#
# ⚠️ **지금은 비어 있다. 채우기 전에 반드시 A/B 를 돌리고 결과를 남길 것.**
# `gpt-4.1-nano` 로 7개 단계를 전부 재봤고 **하나도 통과하지 못했다** (`worker-tasks.md` 17장).
#
#     분절 채점    14개 중 2개에서 경계를 통째로 놓친다 (case2 4→1, case5 2→1)
#     실 상태 채점  점수를 낮게 매겨 화면이 전부 "평온해요"가 된다
#     말투 판정    case8_banter 를 2/2 로 갈등 판정 — 장난에 교정 카드가 뜬다
#     데이트 계획   프롬프트가 "0건 나온다"고 금지한 검색어를 만든다 (`분위기 좋은 카페`)
#     데이트 문구   근거가 사라지고 장소 설명으로 바뀐다
#     말투 생성    주어가 '나'에서 '너'로 돌아가고, 진단문을 대체 문장 자리에 넣는다
#     고민 분류    `apology` 를 `contact` 로 뭉갠다. **게다가 mini 보다 느리다**
#     화제 분류    격리 테스트는 통과했는데 파이프라인에서 깨졌다 — `햄버거 먹방` 대화에
#                 `쯔양 먹방` 을 검색해 소고기 영상을 물어온다. 구체 소재를 놓친다
#
# 마지막 것이 교훈이다. **단계만 떼어 재면 통과한 것처럼 보인다** — 뒤 단계(검색·선정)에
# 미치는 영향은 파이프라인 전체를 돌려야 보인다.
STAGE_MODEL: dict[str, str] = {}

# 이 시간을 넘긴 호출은 **비정상으로 보고 트레이스에 남긴다** (`pipeline.analyze`).
#
# 판정에 관여하지 않는다. 왜 두느냐면 — 한 요청이 22초 걸린 적이 있는데 화면에 단서가
# 없었다. 단계별 시간은 보이지만 그게 정상 범위인지 알 방법이 없었고, LangChain 은
# 429·5xx 재시도를 **조용히 삼켜서** 백오프가 그냥 "느린 호출"로 보인다.
#
# 정상 범위는 1~4초다 (gpt-4.1-mini, 입력 1~3천 토큰). 6초를 넘으면 대개 재시도가 끼었다.
SLOW_CALL_SECONDS = 6.0


# 호출 하나의 HTTP 타임아웃(초). **없으면 OpenAI SDK 기본값 600초다** — 연결이 멎으면
# 워커 마감(25초)까지 기다렸다가 요청 전체가 `ANALYSIS_TIMEOUT` 이 된다. 여기서 먼저
# 끊으면 그 단계만 자기 폴백(분절 → 안 자름, 상태 → 갱신 없음, 후보 → 미발동)으로
# 떨어지고 나머지는 산다. 정상 호출은 1~4초라 15초는 재시도 백오프까지 넉넉하다.
LLM_TIMEOUT = float(os.getenv("KAKAPO_LLM_TIMEOUT", "15"))

# OpenAI 처리 우선순위. 비우면 기본(auto). `priority` 는 지연이 줄지만 토큰 단가가
# 오른다 — 시연·피크 시간에만 켜는 손잡이로 둔다 (`docs/refactoring.md`).
SERVICE_TIER = os.getenv("KAKAPO_SERVICE_TIER", "").strip() or None


@lru_cache(maxsize=8)
def _model(with_temperature: bool, name: str):
    kwargs: dict = {"timeout": LLM_TIMEOUT}
    if SERVICE_TIER:
        kwargs["service_tier"] = SERVICE_TIER
    raw = os.getenv("KAKAPO_TEMPERATURE", "0.3").strip()
    if with_temperature and raw:
        kwargs["temperature"] = float(raw)
    return init_chat_model(name, **kwargs)


@lru_cache(maxsize=64)
def _structured(schema: type, with_temperature: bool, name: str):
    """스키마별 구조화 출력 러너블. **호출마다 다시 만들지 않는다.**

    `with_structured_output` 은 부를 때마다 pydantic 에서 JSON 스키마를 뽑고 러너블을
    새로 엮는다 — 단계 7종 × 요청마다 반복되는 순수 CPU 낭비라 한 번만 만든다.
    """
    return _model(with_temperature, name).with_structured_output(
        schema, method="json_schema", strict=True, include_raw=True
    )


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    """`worker/prompts/{name}.md` 를 읽는다."""
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


class Call(NamedTuple):
    """호출 1건의 계량. `stage` 는 출력 스키마 이름이라 단계와 1:1 로 대응한다."""

    stage: str
    seconds: float
    input: int
    output: int


class Usage:
    """이 프로세스에서 쓴 토큰과 시간. **유료는 OpenAI 뿐이라 여기만 센다.**

    카카오·유튜브는 무료 쿼터라 돈이 아니라 횟수 문제다.

    `records` 에 호출별로 남긴다. 합계만 보면 어느 단계가 느린지 알 수 없어서 —
    분절이 붙은 뒤로는 "요청당 몇 초"보다 "어느 단계가 몇 초"가 더 필요해졌다.

    ⚠️ **`router.run()` 이 단계를 병렬로 돌리므로 여러 스레드가 동시에 `add()` 를 부른다.**
    `+=` 는 원자적이지 않아서 락 없이 두면 카운트가 새고, `records` 도 순서가 아니라
    **개수**가 어긋난다. 그래서 락을 건다 — 호출당 한 번이라 경합 비용은 없다.

    `seconds` 의 합은 병렬 실행에서 **벽시계 시간보다 크다.** "LLM 에 쓴 시간의 총합"이지
    "요청이 걸린 시간"이 아니다. 화면에서 그렇게 표시할 것.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.input = 0
        self.output = 0
        self.records: list[Call] = []

    def add(self, meta: dict | None, stage: str = "", seconds: float = 0.0) -> None:
        got_in = meta.get("input_tokens", 0) if meta else 0
        got_out = meta.get("output_tokens", 0) if meta else 0
        with self._lock:
            self.calls += 1
            self.input += got_in
            self.output += got_out
            self.records.append(
                Call(stage=stage, seconds=seconds, input=got_in, output=got_out)
            )

    def reset(self) -> None:
        with self._lock:
            self.calls = self.input = self.output = 0
            self.records = []

    def __str__(self) -> str:
        return f"LLM {self.calls}회 · 입력 {self.input:,} 토큰 · 출력 {self.output:,} 토큰"


USAGE = Usage()


def ask(schema: type[T], system: str, user: str) -> T:
    """구조화 출력 단발 호출.

    툴 루프가 없는 단발 분류/생성이라 create_agent 를 쓰지 않는다.
    reasoning 계열 모델은 temperature 를 거부하므로 한 번 재시도한다.

    `include_raw=True` 로 받는 이유는 **토큰 사용량을 세기 위해서다.** 파싱된 결과만
    받으면 usage_metadata 가 딸려오지 않아 비용을 알 수 없다.
    """
    messages = [("system", system), ("human", user)]
    # 기본 모델은 환경변수로, 단계별 예외는 `STAGE_MODEL` 로 정한다.
    name = STAGE_MODEL.get(schema.__name__) or os.getenv("KAKAPO_MODEL", DEFAULT_MODEL)
    for with_temperature in (True, False):
        model = _structured(schema, with_temperature, name)
        started = time.perf_counter()
        try:
            result = model.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            if with_temperature and "temperature" in str(exc).lower():
                continue
            raise

        USAGE.add(
            getattr(result.get("raw"), "usage_metadata", None),
            stage=schema.__name__,
            seconds=time.perf_counter() - started,
        )
        if result.get("parsing_error") is not None:
            raise RuntimeError(f"구조화 출력 파싱 실패: {result['parsing_error']}")
        return result["parsed"]
    raise RuntimeError("unreachable")

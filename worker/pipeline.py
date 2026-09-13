"""전체 조립.

규격서(`docs/contract.md`)의 **공통 분석 요청**을 받아 **공통 분석 응답**을 만든다.

    AnalysisRequest → [대화 분절] → [기억 수확 · 실 상태 · 후보 3종] → AnalysisResponse

`router.run()` 이 분절부터 **독립인 것을 동시에** 돌린다 — 말투 후보는 분절도 안 기다리고
먼저 출발한다(입력이 분절과 무관해서). 의존은 말투 → 유튜브 하나다.
(기억 저장소·추출은 2026-09-13 파킹 — `parked/memory/`.)

위젯은 두 줄이고 응답의 배열도 두 개다. ①번 줄(상시) = `emotionAnalyses`,
②번 줄(3종 개입, 미발동이면 빈 줄) = `results`. 서로 밀어내지 않는다.

분절이 맨 앞에 서는 이유는 `docs/design.md` 1부 에 있다. 요약하면 — 서버는 최근 N개를
통째로 보내고 화제 경계를 모른다. 그대로 두면 두 시간 전에 끝난 데이트 얘기가 지금 막
싸우기 시작한 요청에서 데이트 코스를 발동시킨다 (실측으로 확인한 고장이다).

> ⏸️ HTTP 서버(`POST /internal/v1/chat-analyses`)는 아직 만들지 않았다.
> 규격서가 초안 v1 이고 미확정 항목(감정 분석, 타임아웃, 오류 코드)이 남아 있어서,
> 지금은 **DTO 와 판정 로직만** 규격에 맞춰두고 CLI 로 검증한다.
> 서버 담당자와 규격이 확정되면 `analyze()` 를 FastAPI 핸들러에 그대로 물리면 된다.

`emotionAnalyses` 는 **실 상태 표현**이 채운다 (`docs/design.md` 2부).
규격 변경 없이 규격서 11장이 비워 둔 자리를 그대로 쓴다.
"""

from __future__ import annotations

from pydantic import ValidationError

from worker.llm import SLOW_CALL_SECONDS, USAGE
from worker.models import AnalysisRequest, AnalysisResponse
from worker.router import Context, Trace, run

__all__ = ["Context", "Trace", "analyze"]


def analyze(payload: dict, persist: bool = True) -> tuple[AnalysisResponse, Trace]:
    """규격서 요청 JSON 을 받아 응답을 만든다.

    예외를 밖으로 던지지 않는다. 규격서 12장의 `FAILED` 응답으로 바꿔서 돌려준다 —
    워커가 죽으면 채팅 서버가 타임아웃까지 기다리게 된다.

    `persist` 는 기억 저장소 파킹(2026-09-13) 뒤로 아무것도 하지 않는다 — CLI `--no-persist`,
    `KAKAPO_PERSIST` 와의 인터페이스 호환으로만 남겨 둔다.
    """
    trace = Trace()

    try:
        request = AnalysisRequest(**payload)
    except ValidationError as exc:
        return _failed(payload.get("analysisRequestId", ""), "INVALID_REQUEST", str(exc)), trace

    if not request.messages:
        return _skipped(request.analysis_request_id), trace

    known = {p.participant_key.removeprefix("USER_") for p in request.participants}
    unknown = {m.sender for m in request.messages} - known
    if unknown:
        return (
            _failed(
                request.analysis_request_id,
                "INVALID_PARTICIPANT",
                f"참여자 목록에 없는 발화자: {', '.join(sorted(unknown))}",
            ),
            trace,
        )

    ctx = Context(
        request=request,
        messages=sorted(request.messages, key=lambda m: (m.sent_at, m.message_id)),
        now=request.now(),
        persist=persist,
        trace=trace,
    )

    # 이번 요청에서 난 LLM 호출만 보려고 위치를 잡아둔다 (USAGE 는 프로세스 전역 누적).
    mark = len(USAGE.records)

    try:
        states, results = run(ctx)
    except Exception as exc:  # noqa: BLE001
        return _failed(request.analysis_request_id, "MODEL_ERROR", str(exc)), trace

    # 비정상적으로 느린 호출을 남긴다. **판정을 바꾸지 않는다** — 왜 느렸는지 되짚을
    # 단서만 만든다. 뺄셈이라 비용이 없다 (`llm.SLOW_CALL_SECONDS` 주석 참조).
    for record in USAGE.records[mark:]:
        if record.seconds >= SLOW_CALL_SECONDS:
            trace.warn(
                "llm",
                f"{record.stage} {record.seconds:.1f}초 — 정상 범위(1~4초) 밖. "
                "재시도가 끼었을 수 있다 (레이트 리밋·API 지연)",
            )

    # 규격서 12장 — `COMPLETED` 는 "기능 결과 **또는 감정 분석 결과**가 존재".
    # 실 상태 표현이 상시라서 `states` 가 거의 항상 차고, 따라서 **`SKIPPED` 는 거의
    # 나오지 않는다.** 상태 산출까지 실패했을 때만 남는다.
    # 서버가 `SKIPPED` 를 "아무것도 안 함"으로 보고 전송을 건너뛰면 ①번 줄이 영영 안 뜬다 —
    # `docs/server-handoff.md` 에 고지해 둔 항목이다.
    if not results and not states:
        return _skipped(request.analysis_request_id), trace

    return (
        AnalysisResponse(
            analysis_request_id=request.analysis_request_id,
            status="COMPLETED",
            results=results,
            emotion_analyses=states,
        ),
        trace,
    )


def _skipped(request_id: str) -> AnalysisResponse:
    """분석은 정상 완료됐지만 발동할 기능이 없는 상태 (규격서 12장)."""
    return AnalysisResponse(
        analysis_request_id=request_id,
        status="SKIPPED",
        results=[],
        emotion_analyses=[],
    )


def _failed(request_id: str, code: str, message: str) -> AnalysisResponse:
    return AnalysisResponse(
        analysis_request_id=request_id,
        status="FAILED",
        results=[],
        emotion_analyses=[],
        error_code=code,  # type: ignore[arg-type]
        error_message=message,
    )

"""HTTP 서버 — 채팅 서버가 부르는 진입점.

    POST /internal/v1/chat-analyses   공통 분석 요청 → 공통 분석 응답 (규격서 4~6장)
    GET  /health                      살아 있는지
    GET  /docs · /openapi.json        FastAPI 가 자동으로 만든다

**판정 로직을 하나도 갖지 않는다.** 요청을 `pipeline.analyze()` 에 넘기고 그 결과를
규격서 JSON 으로 직렬화하는 것이 전부다. CLI(`tools/run.py`)·devui 와 **같은 진입점**을
부른다 — 세 갈래가 같은 함수를 부르니 "CLI 에서는 되는데 서버에서는 다르다"가 생기지 않는다.

## 이 파일에서 결정한 것

**응답은 항상 HTTP 200 이다.** 규격서에 HTTP 상태 코드 규정이 없고, 분석 결과는
`status`(`COMPLETED`/`SKIPPED`/`FAILED`)와 `errorCode` 로 이미 표현된다. 잘못된 요청도
규격서 12·13장의 `FAILED` 봉투에 담아 200 으로 돌려준다 — 그래야 백엔드가 HTTP 예외
경로와 분석 실패 경로를 따로 짜지 않아도 된다. 서버가 4xx 를 원하면 여기만 고치면 된다.

**직렬화는 `to_json_dict()` 가 한다.** FastAPI 의 기본 직렬화를 쓰면 `exclude_none` 이
걸리지 않아 `null` 필드가 나가고, 규격서 14장("선택 필드에 값이 없으면 필드 생략")이
깨진다. 그래서 `JSONResponse` 로 우리가 만든 dict 를 그대로 내보낸다.
`response_model` 은 **Swagger 문서용으로만** 선언한다.

**요청 본문은 `AnalysisRequest` 로 선언한다.** /docs 에 요청 스키마가 뜨게 하려면 필요하다.
대신 검증 실패는 FastAPI 기본값(422)이 아니라 `FAILED` 봉투로 바꿔서 내보낸다 (아래 핸들러).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from worker import places, ytapi
from worker.models import AnalysisRequest, AnalysisResponse
from worker.pipeline import analyze
from worker.retrieve import warm_index

# 서버 쪽 요구는 "처리시간 30초 이내"다. 그보다 앞에서 우리가 끊는다 —
# 백엔드가 커넥션을 자르면 남는 게 타임아웃 로그뿐이지만, 우리가 끊으면
# 규격서 13장의 `ANALYSIS_TIMEOUT` 을 담은 정상 응답이 나간다.
#
# 실측 최장이 11.3초(데이트 코스 + 콜드 스타트)라 25초면 두 배 이상 여유가 있다.
# 여기 걸리면 정상 지연이 아니라 OpenAI 쪽이 멈춘 것이다.
ANALYSIS_DEADLINE = float(os.getenv("KAKAPO_DEADLINE", "25"))

# 기억을 파일에 쓸지. 배포 환경의 파일시스템은 재배포하면 날아가므로 영속 저장이 아니다.
# 그래도 켜 두는 이유는 **한 대화 안에서** 방금 추출한 기억이 다음 요청의 데이트 코스
# 근거가 되기 때문이다. 반복 시연으로 결과를 고정하고 싶으면 `KAKAPO_PERSIST=0`.
PERSIST = os.getenv("KAKAPO_PERSIST", "1") not in {"0", "false", "False"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 어떤 키가 꽂혀 있는지 기동 로그에 남긴다. **값은 절대 찍지 않는다.**
    #
    # 키가 빠져도 워커는 죽지 않고 그 기능만 조용히 미발동한다 — 설계가 그렇다.
    # 그래서 **배포 상태로는 절대 안 드러나고** 한참 뒤에 "왜 아무것도 안 떠요"로
    # 나타난다. 배포 로그가 그때 되짚을 수 있는 유일한 흔적이라 여기서 찍는다.
    log = logging.getLogger("uvicorn.error")
    log.info(
        "PORT=%s · 외부 API 키 — openai=%s kakao=%s youtube=%s  (없는 쪽은 해당 기능만 미발동)",
        os.getenv("PORT", "8000"),
        bool(os.getenv("OPENAI_API_KEY")),
        places.available(),
        ytapi.available(),
    )

    # 기억 인덱스를 미리 만든다. 27건 임베딩에 2.6초가 걸리는데, 안 하면 그게
    # **첫 요청의 데이트 코스 경로 한가운데** 얹힌다. 결과는 바뀌지 않고 시점만 앞당긴다.
    # 실패해도 넘어간다 — 뒤에서 `_get_store()` 가 다시 시도한다.
    threading.Thread(target=warm_index, daemon=True).start()
    yield


app = FastAPI(
    title="kakapo AI Worker",
    description="커플 대화 분석 워커. 연동 규격은 `docs/contract.md`.",
    version="1.0.0",
    lifespan=lifespan,
)


def _failed(request_id: str, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        AnalysisResponse(
            analysis_request_id=request_id,
            status="FAILED",
            error_code=code,  # type: ignore[arg-type]
            error_message=message,
        ).to_json_dict()
    )


@app.exception_handler(RequestValidationError)
async def _on_invalid_body(request: Request, exc: RequestValidationError) -> JSONResponse:
    """본문 검증 실패 → 422 대신 규격서 `FAILED` 봉투.

    `analysisRequestId` 는 **되찾을 수 있으면 되찾아서** 실어 보낸다. 백엔드가 어느 요청이
    깨졌는지 대조할 유일한 열쇠라, 파싱이 실패했다는 이유로 비워 보내면 추적이 끊긴다.
    """
    request_id = ""
    try:
        body = await request.json()
        if isinstance(body, dict):
            request_id = str(body.get("analysisRequestId", "") or "")
    except Exception:  # noqa: BLE001 — 본문이 JSON 이 아닐 수도 있다
        pass

    # **여기도 남긴다.** 이 경로는 `analyze()` 를 타지 않아서 아래 `_log_outcome` 을
    # 지나가지 않는다. 본문이 규격과 어긋나는 건 연동 초기에 가장 흔한 실패라,
    # 어느 필드가 걸렸는지가 로그에 없으면 양쪽이 서로를 의심하게 된다.
    logging.getLogger("uvicorn.error").warning(
        "분석 %s · FAILED · 오류=INVALID_REQUEST · %s",
        request_id or "(id 없음)",
        exc.errors(),
    )
    return _failed(request_id, "INVALID_REQUEST", str(exc.errors()))


@app.get("/health", tags=["health"])
def health() -> dict:
    """살아 있는지 + 외부 API 키가 꽂혀 있는지.

    키 상태를 같이 주는 이유는 **없어도 워커가 죽지 않기 때문이다.** 카카오·유튜브 키가
    빠지면 그 기능만 조용히 미발동하는데, 화면에서는 "추천할 게 없었다"와 구분되지 않는다.
    값은 내보내지 않고 꽂혔는지만 알린다.
    """
    return {
        "status": "UP",
        "features": {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "kakao": places.available(),      # 데이트 코스
            "youtube": ytapi.available(),     # 유튜브 추천
        },
    }


def _log_outcome(
    request_id: str,
    started: float,
    status: str,
    error_code: str | None = None,
    result_types: list[str] | None = None,
    state_count: int = 0,
    skipped: list[tuple[str, str]] | None = None,
    segment_cache: tuple[int, int] | None = None,
) -> None:
    """요청 하나의 결과를 한 줄로 남긴다.

    **액세스 로그만으로는 성공·실패를 구분할 수 없다.** 오류도 `FAILED` 봉투에 담아
    HTTP 200 으로 내보내는 설계라(파일 상단 참조), uvicorn 이 남기는 `200 OK` 는
    "응답이 나갔다"는 뜻일 뿐이다. 배포 환경에는 `--verbose` 도 devui 도 없어서,
    이 줄이 없으면 **무엇이 발동했는지 알 방법이 자체가 없다.**

    `analysisRequestId` 를 앞에 두는 이유는 백엔드가 같은 값을 로그에 남기기 때문이다.
    양쪽 로그를 그 값으로 맞대면 어디서 끊겼는지가 바로 나온다.
    """
    seconds = time.perf_counter() - started
    tail = f"오류={error_code}" if error_code else (
        f"결과={','.join(result_types) if result_types else '없음'} 상태표현={state_count}건"
    )
    # 미발동 사유도 같이 남긴다. "결과=없음" 만으로는 쿨다운·말투 보류·게이트 미통과가
    # 구분되지 않아서, 프론트가 "안 뜬다"고 할 때 배포 로그만으로 원인을 좁힐 수 없었다.
    # 단, 쿼터 소진은 여기에도 안 남는다 — 검색이 0건으로 돌아올 뿐이다 (demo-checklist 6장).
    if skipped:
        tail += " · 보류=" + "; ".join(f"{name}:{reason}" for name, reason in skipped)
    # 분절 점수 캐시 적중. "캐시 N/M" = 채점 대상 M개 중 N개를 캐시에서 가져왔다.
    # 방의 첫 요청은 0/M, 그 뒤는 (M-1)/M 이어야 정상이다 — 계속 0 이면 캐시가 안 맞는 것.
    if segment_cache is not None and sum(segment_cache) > 0:
        tail += f" · 분절캐시={segment_cache[0]}/{sum(segment_cache)}"
    logging.getLogger("uvicorn.error").info(
        "분석 %s · %.1f초 · %s · %s", request_id or "(id 없음)", seconds, status, tail
    )


@app.post(
    "/internal/v1/chat-analyses",
    tags=["analysis"],
    response_model=AnalysisResponse,          # Swagger 문서용. 실제 직렬화는 아래 참조
    response_model_by_alias=True,
    response_model_exclude_none=True,
)
async def chat_analyses(request: AnalysisRequest) -> JSONResponse:
    """공통 분석 요청 → 공통 분석 응답 (규격서 5·6장).

    `analyze()` 는 동기 함수이고 안에서 LLM 을 병렬로 부른다. `async def` 안에서 그냥
    부르면 이벤트 루프가 통째로 멈춰서 **동시 요청이 직렬화된다.** 스레드풀로 넘긴다.
    """
    payload = request.model_dump(by_alias=True, mode="json")
    started = time.perf_counter()

    try:
        response, trace = await asyncio.wait_for(
            run_in_threadpool(analyze, payload, PERSIST),
            timeout=ANALYSIS_DEADLINE,
        )
    except asyncio.TimeoutError:
        # 넘긴 스레드는 계속 돈다 — 파이썬은 스레드를 밖에서 못 죽인다. 응답만 먼저 보낸다.
        _log_outcome(request.analysis_request_id, started, "FAILED", "ANALYSIS_TIMEOUT")
        return _failed(
            request.analysis_request_id,
            "ANALYSIS_TIMEOUT",
            f"{ANALYSIS_DEADLINE:g}초 안에 분석을 끝내지 못했습니다",
        )

    _log_outcome(
        response.analysis_request_id,
        started,
        response.status,
        response.error_code,
        [r.result_type for r in response.results],
        len(response.emotion_analyses),
        trace.skipped,
        (trace.segment_cached, trace.segment_scored),
    )
    return JSONResponse(response.to_json_dict())


def main() -> None:
    """로컬·배포 공통 실행 진입점 — `python -m worker.api`. 포트는 `PORT`, 기본 8000.

    **IPv4 와 IPv6 를 한 소켓으로 같이 받는다.** 둘 중 하나만 묶으면 한쪽이 끊긴다:

        0.0.0.0  → IPv4 만. Railway 의 프로젝트 내부 통신(사설망)이 **IPv6 전용**이라
                   같은 프로젝트의 채팅 서버가 붙지 못한다
        ::       → IPv6 만. `uvicorn --host ::` 가 이렇게 된다 (asyncio 가 소켓에
                   IPV6_V6ONLY 를 켠다). `127.0.0.1` 로 접속하는 쪽이 전부 끊긴다

    그래서 `IPV6_V6ONLY` 를 끈 듀얼스택 소켓을 직접 만들어 uvicorn 에 넘긴다.
    이러면 `[::1]` 도 `127.0.0.1` 도 같은 서버에 닿는다. **실측으로 확인하고 넣은 코드다** —
    `--host ::` 로 띄웠더니 IPv6 만 응답했다.

    듀얼스택을 못 쓰는 환경이거나 `HOST` 를 명시하면 그 주소로만 묶는다.
    """
    import socket

    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST")
    config = uvicorn.Config("worker.api:app", host=host or "::", port=port)
    server = uvicorn.Server(config)

    if host or not socket.has_dualstack_ipv6():
        server.run()
        return

    sock = socket.create_server(
        ("::", port), family=socket.AF_INET6, dualstack_ipv6=True, reuse_port=False
    )
    server.run(sockets=[sock])


if __name__ == "__main__":
    main()

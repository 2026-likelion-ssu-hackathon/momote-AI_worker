# kakapo AI 워커 — 배포 이미지.
#
# 런타임에 필요한 것만 넣는다. 평가셋(`data/eval/`, 136MB)·픽스처·개발 도구는 뺀다.
# `yt_seed.json` 은 유튜브 API 가 죽었을 때의 폴백이라 필수다. 기억 시드·말투 기준선 시드는
# 2026-09-13 에 뺐다 (`parked/`) — 말투 기준선은 코드 안의 공통값(`profile.GENERIC_PROFILE`)이다.
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY worker/ worker/
COPY data/yt_seed.json data/

# `uvicorn` CLI 를 직접 부르지 않는다. `worker/api.py` 의 `main()` 이 IPv4·IPv6 를
# 같이 받는 듀얼스택 소켓을 만들어 넘기기 때문이다 — Railway 의 프로젝트 내부 통신은
# IPv6 전용이고, `--host 0.0.0.0` 은 IPv4 만, `--host ::` 는 IPv6 만 받는다.
#
# 포트는 플랫폼이 `PORT` 로 준다. 없으면 8000.
CMD ["python", "-m", "worker.api"]

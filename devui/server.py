"""로컬 테스트용 개발 UI 서버.

    .venv/bin/python devui/server.py          # http://127.0.0.1:8765 자동 오픈
    .venv/bin/python devui/server.py --port 9000 --no-open

CLI(`tools/run.py`)와 **같은 진입점**(`worker.pipeline.analyze`)을 부른다.
판정 로직은 이 폴더에 하나도 없다 — 대화를 넣고 결과를 보는 껍데기다.

새 의존성을 쓰지 않는다. 표준 라이브러리 `http.server` 하나로 끝낸다 —
개발 도구 때문에 워커 런타임 의존성이 늘어나면 안 된다.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from dataclasses import asdict, is_dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel  # noqa: E402

from worker import segment, state  # noqa: E402
from worker.llm import USAGE  # noqa: E402
from worker.pipeline import Trace, analyze  # noqa: E402

FIXTURE_DIR = ROOT / "fixtures"

# 기억 저장소(data/memories.json)와 USAGE 가 프로세스 전역이다.
# 브라우저에서 두 번 연타하면 섞이므로 한 번에 하나만 돌린다.
_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# 직렬화
# --------------------------------------------------------------------------
def _jsonable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def _cut_reason(seg, scores: dict, prev_end) -> str:
    """이 세그먼트가 **왜 여기서 시작됐는가.** 판정을 코드로 옮긴 값어치가 이것이다.

    `worker.segment._should_cut()` 과 같은 순서로 되짚는다. 로직이 바뀌면 여기도 바꾼다.
    """
    gap = _minutes(prev_end, seg.messages[0])
    hard = int(segment.GAP_HARD.total_seconds() // 60)

    # 룰 컷 경계는 **점수가 아예 없다.** 채점은 조각 안에서만 돌기 때문에, 조각의 첫
    # 발화에는 비교 대상이 없어서 점수가 매겨지지 않는다. by_rule 만 보면 안 되는 이유다 —
    # 마지막 조각은 채점을 거쳐서 by_rule=False 인데 그 앞 경계는 룰 컷이다.
    if gap >= hard:
        return f"룰 컷 — 공백 {gap}분 ≥ 3시간"

    score = scores.get(seg.messages[0].message_id)
    if score is None:
        return "판정 정보 없음 (폴백)"

    if score.topic < segment.CUT_HARD:
        return f"화제 {score.topic} < {segment.CUT_HARD} — 무조건 자름"

    soft = int(segment.GAP_SOFT.total_seconds() // 60)
    if gap >= soft:
        return f"회색(화제 {score.topic}) + 공백 {gap}분 ≥ {soft}분"
    return f"회색(화제 {score.topic}) + 말투 {score.tone} < {segment.TONE_CUT}"


def _minutes(prev, cur) -> int:
    if prev is None:
        return 0
    return int((cur.sent_at - prev.sent_at).total_seconds() // 60)


def _segments_json(trace: Trace) -> list[dict]:
    scores = {s.id: s for s in trace.scores}
    out = []
    prev_end = None
    for i, seg in enumerate(trace.segments):
        first = seg.messages[0]
        out.append({
            "messageIds": seg.message_ids,
            "head": first.content,
            "byRule": seg.by_rule,
            "startedAt": first.sent_at.isoformat(),
            "gapBefore": _minutes(prev_end, first),
            "cutReason": "첫 세그먼트" if i == 0 else _cut_reason(seg, scores, prev_end),
        })
        prev_end = seg.messages[-1]
    return out


def _scoring_failed(trace: Trace) -> bool:
    """마지막 조각이 채점 대상이었는데 점수가 하나도 없으면 폴백이다."""
    if not trace.segments or trace.scores:
        return False
    return len(trace.segments[-1].messages) >= segment.MIN_FOR_LLM


def _trace_json(trace: Trace) -> dict:
    """`tools/run.py` 의 `--verbose` 와 같은 내용을 JSON 으로."""
    return {
        "fired": list(trace.fired),
        "skipped": [{"candidate": name, "reason": reason} for name, reason in trace.skipped],
        # 결과는 나갔지만 규칙이 어긋난 것. 판정을 바꾸지 않는다.
        "warnings": [{"candidate": name, "note": note} for name, note in trace.warnings],
        "segments": _segments_json(trace),
        # 발화별 연속성 점수 + 판정 임계값. 왜 잘렸는지가 화면에서 보이게 한다.
        "scores": _jsonable(trace.scores),
        # 채점이 통째로 실패했는가. **결과만 보면 "경계 없음"과 구별이 안 된다** —
        # 둘 다 세그먼트 1개다. 화면에서 갈라 보여줘야 원인을 짚을 수 있다.
        "scoringFailed": _scoring_failed(trace),
        "thresholds": {
            "cutHard": segment.CUT_HARD,
            "keepSoft": segment.KEEP_SOFT,
            "toneCut": segment.TONE_CUT,
            "gapSoftMinutes": int(segment.GAP_SOFT.total_seconds() // 60),
        },
        "extracted": _jsonable(trace.extracted),
        "savedIds": [m.id for m in trace.saved],
        # 실 상태 표현 — 감정 점수와 **거기서 룰이 고른 라벨.**
        # 판정을 코드로 옮긴 값어치가 여기서도 보인다. `note` 는 내부용이라 화면 문구가 아니다.
        "stateScored": [
            {**_jsonable(s), "label": state.pick_label(s)[0], "intensity": state.pick_label(s)[1]}
            for s in trace.state_scored
        ],
        "stateThresholds": {
            "minScore": state.MIN_SCORE,
            "angerWins": state.ANGER_WINS,
            "priority": list(state.PRIORITY),
        },
        "toneGate": _jsonable(trace.tone_gate),
        "toneJudged": _jsonable(trace.tone_judged),
        "dateGate": _jsonable(trace.date_gate),
        "dateMemories": _jsonable(trace.date_memories),
        "datePlan": _jsonable(trace.date_plan),
        "datePlaces": _jsonable(trace.date_places),
        "concern": _jsonable(trace.concern),
        "ytCandidates": _jsonable(trace.yt_candidates),
        "ytPickedId": trace.yt_picked.video_id if trace.yt_picked else None,
    }


def _fixtures() -> list[dict]:
    items = []
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            items.append({"name": path.name, "error": str(exc)})
            continue
        items.append({"name": path.name, "payload": payload})
    return items


def _run(payload: dict, persist: bool) -> dict:
    with _LOCK:
        mark = len(USAGE.records)
        started = time.perf_counter()
        response, trace = analyze(payload, persist=persist, wait_background=True)
        elapsed = time.perf_counter() - started
        # 이번 요청에서 난 호출만 잘라낸다. USAGE 는 프로세스 전역 누적이다.
        records = USAGE.records[mark:]

    llm_seconds = sum(r.seconds for r in records)
    return {
        "response": response.to_json_dict(),
        "trace": _trace_json(trace),
        "usage": {
            "calls": len(records),
            "input": sum(r.input for r in records),
            "output": sum(r.output for r in records),
            # ⚠️ **LLM 시간의 합계지 요청이 걸린 시간이 아니다.** `router.run()` 이 단계를
            # 병렬로 돌려서 이 값이 `elapsedMs` 보다 클 수 있다. 화면에서 "합계"로 표시한다.
            "llmMs": round(llm_seconds * 1000),
            # 외부 API(카카오·유튜브)와 룰·파일 I/O. 따로 계측하지 않고 차로 낸다.
            # 병렬 실행에서는 LLM 대기와 겹쳐서 0 으로 떨어질 수 있다 — 안 걸렸다는 뜻이 아니다.
            "otherMs": max(0, round((elapsed - llm_seconds) * 1000)),
            "stages": [
                {
                    "stage": r.stage,
                    "ms": round(r.seconds * 1000),
                    "input": r.input,
                    "output": r.output,
                }
                for r in records
            ],
        },
        "elapsedMs": round(elapsed * 1000),
    }


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "kakapo-devui"

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            html = (HERE / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
            return

        if path == "/api/fixtures":
            self._json(200, {"fixtures": _fixtures()})
            return

        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/api/analyze":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._json(400, {"error": f"요청 JSON 파싱 실패: {exc}"})
            return

        payload = body.get("payload") or {}
        persist = bool(body.get("persist", False))

        try:
            self._json(200, _run(payload, persist))
        except Exception as exc:  # noqa: BLE001 — 브라우저에 그대로 보여준다
            import traceback

            traceback.print_exc()
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args) -> None:
        # 요청 로그는 시끄럽고, 분석 결과는 브라우저에서 본다.
        pass


def main() -> int:
    parser = argparse.ArgumentParser(prog="devui", description="kakapo 워커 개발 UI")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-open", action="store_true", help="브라우저를 열지 않는다")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"kakapo devui  →  {url}   (Ctrl+C 로 종료)")
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

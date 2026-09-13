"""CLI 러너.

    python -m tools.run fixtures/case7_tone.json
    python -m tools.run fixtures/case7_tone.json --verbose
    python -m tools.run fixtures/*.json --no-persist
    python -m tools.run fixtures/case9_date.json --json    # 규격서 응답 그대로

픽스처는 **규격서 5장의 공통 분석 요청 형식 그대로**다. 서버가 보낼 페이로드와 같은 것을
넣고 있으므로, 여기서 통과하면 DTO 는 맞춰진 것이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from worker.models import (
    AiResult,
    AnalysisResponse,
    DateCourseResultData,
    ToneResultData,
    YoutubeResultData,
)
from worker import date_course, segment, state
from worker.llm import USAGE
from worker.pipeline import Trace, analyze

DIM = "\033[2m"
BOLD = "\033[1m"
OFF = "\033[0m"

CANDIDATE_LABEL = {
    "tone": "갈등 중재 (말투 교정)",
    "date": "데이트 코스 추천",
    "youtube": "유튜브 영상 추천",
    "llm": "LLM 호출",
}


def _show_trace(trace: Trace) -> None:
    if trace.segments:
        cache = ""
        if trace.segment_cached or trace.segment_scored:
            cache = (f" · 점수 캐시 {trace.segment_cached}/"
                     f"{trace.segment_cached + trace.segment_scored}")
        print(f"  대화분절 {len(trace.segments)}개  {DIM}(마지막이 활성 세그먼트{cache}){OFF}")
        for i, seg in enumerate(trace.segments):
            head = "→" if i == len(trace.segments) - 1 else " "
            how = "룰컷" if seg.by_rule else "채점"
            ids = seg.message_ids
            span = f"{ids[0]}~{ids[-1]}" if len(ids) > 1 else f"{ids[0]}"
            first = seg.messages[0].content[:30]
            print(f"          {head} [{how}] {span} ({len(ids)}개) {DIM}{first}{OFF}")

    if trace.scores:
        # 왜 잘렸는지가 여기서 보인다. 판정을 코드로 옮긴 이유가 이것이다.
        cut_ids = {s.messages[0].message_id for s in trace.segments[1:]}
        print(f"  연속성점수 {DIM}(높을수록 이전 맥락과 이어짐 · "
              f"컷<{segment.CUT_HARD} 회색 유지≥{segment.KEEP_SOFT}){OFF}")
        for sc in trace.scores:
            mark = "✂" if sc.id in cut_ids else " "
            zone = ("자름" if sc.topic < segment.CUT_HARD
                    else "유지" if sc.topic >= segment.KEEP_SOFT else "회색")
            print(f"          {mark} #{sc.id} 화제 {sc.topic:>3} "
                  f"말투 {sc.tone:>3} 같은맥락={str(sc.same):<5} {zone}")

    if trace.state_scored:
        # 5축 점수와 거기서 룰이 고른 라벨. note 는 내부용이라 화면에 안 나간다.
        print(f"  실상태채점 {DIM}(LLM 은 점수만 · 라벨은 임계값이 정한다"
              f" · anger≥{state.ANGER_WINS} 우선 · 최고<{state.MIN_SCORE} 면 STABLE){OFF}")
        for sc in trace.state_scored:
            label, intensity = state.pick_label(sc)
            axes = (f"애정{sc.affection} 서운{sc.hurt} 활기{sc.joy} 분노{sc.anger}")
            mark = "" if sc.confident else f" {DIM}확신없음 → STABLE{OFF}"
            print(f"           · {sc.speaker}  {axes}   → {BOLD}{label}{OFF} {intensity:.0f}{mark}")
            print(f"             {DIM}{sc.note}{OFF}")

    if trace.extracted:
        print(f"  기억추출 {len(trace.extracted)}건 (신규 저장 {len(trace.saved)}건)")
        for m in trace.extracted:
            print(f'           · [{m.kind}] {m.content} ← "{m.source_quote}"')

    if (tg := trace.tone_gate) is not None and tg.triggered:
        print("  말투게이트")
        for f in tg.flags:
            print(f"           · [{f.kind}] {f.detail}")
        if (tj := trace.tone_judged) is not None:
            print(
                f"  말투판정 should_suggest={tj.should_suggest} "
                f"is_playful={tj.is_playful} emotion={tj.emotion}"
            )
            print(f"           {DIM}{tj.note}{OFF}")

    if (dg := trace.date_gate) is not None and dg.triggered:
        print(f"  데이트게이트 {dg.detail}")
        if trace.date_memories:
            print("  RAG 기억")
            for m in trace.date_memories:
                mark = " " if m.used_at is None else "*"
                print(f"           {mark}[{m.kind}] {m.content}")
        if (dp := trace.date_plan) is not None:
            print(f"  데이트계획 should_recommend={dp.should_recommend} region={dp.region}")
            for label, _, qs in date_course.slot_queries(dp):
                print(f"           {label} {qs}")
            print(f"           {DIM}{dp.reason_seed}{OFF}")
        if trace.date_places:
            print("  카카오 검색 결과")
            for p in trace.date_places:
                print(f"           · {p.name} — {p.category} ({p.category_name})")

    if (c := trace.concern) is not None:
        print(
            f"  고민판정 should_recommend={c.should_recommend} "
            f"concern={c.concern} stage={c.stage}"
        )
        print(f"           검색어 {c.queries}")
        print(f"           {DIM}{c.note}{OFF}")
    if trace.yt_candidates:
        print(f"  영상 후보 {len(trace.yt_candidates)}개")
        for v in trace.yt_candidates:
            mark = "→" if trace.yt_picked and v.video_id == trace.yt_picked.video_id else " "
            print(f"          {mark} {v.title} / {v.channel_name}")


def _show_result(result: AiResult) -> None:
    scope = result.visibility_type
    if result.target_participant:
        scope = f"{scope} → {result.target_participant}"
    data = result.result_data

    print(f"\n  {BOLD}{result.result_type}{OFF}   {scope}   {DIM}"
          f"trigger={result.trigger_message_ids}{OFF}")

    if isinstance(data, ToneResultData):
        print(f"    진단   {data.situation_diagnosis}")
        print(f"    안내   {DIM}{data.guide_message}{OFF}")
        print(f'    대체   "{data.alternative_sentence}"')
        print(f"    이유   {data.correction_reason}")

    elif isinstance(data, DateCourseResultData):
        print(f"    안내   {DIM}{data.guide_message}{OFF}")
        print(f"    코스   {data.course_name}")
        print(f"    요약   {data.course_summary}")
        print(f"    핵심   {data.main_place.name} — {data.main_place.summary}")
        print(f"           {DIM}{data.main_place.external_url}{OFF}")
        for p in data.course_places:
            print(f"    {p.order}.     {p.name} [{p.category}] — {p.summary}")
        print(f"    이유   {data.recommendation_reason}")

    elif isinstance(data, YoutubeResultData):
        print(f"    안내   {DIM}{data.guide_message}{OFF}")
        print(f"    이유   {data.recommendation_reason}")
        print(f"    영상   {data.title}")
        print(f"    채널   {data.channel_name}")
        print(f"    링크   {DIM}{data.video_url}{OFF}")
        if data.video_summary:
            print(f"    요약   {DIM}{data.video_summary[:80]}…{OFF}")


def _show_states(response: AnalysisResponse) -> None:
    """위젯 ①번 줄 — 보는 사람마다 다르다.

    A 화면과 B 화면의 내용이 다른 것이 이 기능의 핵심이라, 화면 기준으로 묶어서 보여준다.
    """
    print(f"\n  {BOLD}실 상태{OFF}  {DIM}(위젯 ①번 줄 · 상시){OFF}")
    for s in response.emotion_analyses:
        keep = "" if s.should_show else f"  {DIM}← 갱신 없음 (직전 문구 유지){OFF}"
        print(f'    {s.viewer_participant} 화면   "{s.state_text}"   {DIM}'
              f"{s.emotion_type} {s.intensity_value:.1f} · 대상 {s.subject_participant}{OFF}{keep}")


def show(path: Path, response: AnalysisResponse, trace: Trace, verbose: bool) -> None:
    print(f"\n{BOLD}▸ {path.name}{OFF}   status={response.status}")

    if verbose:
        _show_trace(trace)

    if response.status == "FAILED":
        print(f"  {response.error_code}: {response.error_message}")
        return

    if response.emotion_analyses:
        _show_states(response)

    for result in response.results:
        _show_result(result)

    if not response.results:
        print(f"\n  {DIM}개입하지 않음 (②번 줄 비어 있음){OFF}")

    # 왜 안 나갔는지는 시연 중 질문이 가장 많이 나오는 지점이다. verbose 없이도 보여준다.
    for name, reason in trace.skipped:
        print(f"  {DIM}· {CANDIDATE_LABEL.get(name, name)} — {reason}{OFF}")
    for name, note in trace.warnings:
        print(f"  ⚠ {CANDIDATE_LABEL.get(name, name)} — {note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.run", description="kakapo 워커 러너")
    parser.add_argument("fixtures", nargs="+", type=Path)
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="분절 점수, 게이트 판정, 검색된 기억, 외부 API 결과 표시")
    parser.add_argument("--no-persist", action="store_true",
                        help="used_at / 기억 저장을 파일에 쓰지 않는다 (반복 시연용)")
    parser.add_argument("--json", action="store_true",
                        help="규격서 응답 JSON 을 그대로 출력한다")
    args = parser.parse_args(argv)

    for path in args.fixtures:
        if not path.exists():
            print(f"파일 없음: {path}", file=sys.stderr)
            return 1
        payload = json.loads(path.read_text(encoding="utf-8"))
        response, trace = analyze(payload, persist=not args.no_persist, wait_background=True)

        if args.json:
            print(json.dumps(response.to_json_dict(), ensure_ascii=False, indent=2))
        else:
            show(path, response, trace, args.verbose)

    # 유료는 OpenAI 뿐이다. 얼마나 썼는지 매번 보여준다.
    if not args.json and USAGE.calls:
        print(f"\n{DIM}{USAGE}{OFF}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

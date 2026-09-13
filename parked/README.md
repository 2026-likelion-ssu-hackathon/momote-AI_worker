# 파킹 — 대화 소재 제시 (`topic`)

우선순위에서 밀려 **현재 파이프라인에서 빠진** 후보 기능이다. 코드는 지우지 않고 여기 보관한다.
이 폴더는 파이썬 패키지가 아니라 **아카이브**다. `worker/` 에서 import 되지 않으며, 실행되지도 않는다.

파킹 시점: 백엔드 연동 규격 v1 반영 작업 (`docs/spec.md` 참조)
파킹 직전 상태 태그: `git tag topic-parked`

---

## 무엇이 여기 있나

| 파일 | 원래 위치 | 내용 |
| --- | --- | --- |
| `gate.py` | `worker/gate.py` | 룰 게이트 — 트리거 ①단답 핑퐁 ②질문 없는 대답 ③한쪽만 발화 ⑤대화 중 정체 |
| `judge.py` | `worker/judge.py` | LLM 판정 — 트리거 ④일상 보고형 반복, 바쁨 판별 (+ 기억 추출) |
| `topic.py` | `worker/topic.py` | 소재 생성 (기억 기반) / 오늘의 질문 |
| `prompts/judge.md` | `worker/prompts/judge.md` | 판정 + 기억 추출 프롬프트 |
| `prompts/topic.md` | `worker/prompts/topic.md` | 소재 생성 프롬프트 |
| `daily_questions.json` | `data/daily_questions.json` | 오늘의 질문 30개 |

## 무엇이 남았나

| 남긴 것 | 이유 |
| --- | --- |
| `worker/text.py` | `gate.py` 안에 있던 문장 판별 유틸(`is_reaction` / `is_question`). 데이트 코스 트리거가 쓴다 |

`retrieve.py` · `extract.py` · `data/memories.json` 은 데이트 코스가 쓰느라 남겨 뒀었는데
**2026-09-13 에 `memory/` 로 파킹됐다** (아래).

---

# 파킹 — 기억 저장소 + RAG (`memory/`, 2026-09-13)

방을 사용자마다 만들 수 있게 되면서 **전역 저장소 하나**(`data/memories.json`)가 문제가 됐다.
`Memory` 에 방 id 가 없어서 방 X 에서 추출한 "마라탕 땡긴다"가 방 Y 의 데이트 코스 근거로
인용될 수 있다 — "지난주에 마라탕 땡긴다고 하셨던 거"가 **다른 커플의 말**이 된다. 시드
27건도 모든 방에 똑같이 보였다. 방 단위로 가르려면 시드를 어느 방에 줄지 정해야 해서
(사용자 결정) **일단 떼어냈다.**

| 파일 | 원래 위치 | 내용 |
| --- | --- | --- |
| `memory/retrieve.py` | `worker/retrieve.py` | RAG 검색 (`OpenAIEmbeddings` + `InMemoryVectorStore`), 파일 I/O, `used_at` 소모 처리 |
| `memory/extract.py` | `worker/extract.py` | 대화에서 기억 추출 (LLM, 원문 인용 검증) |
| `memory/prompts/extract.md` | `worker/prompts/extract.md` | 추출 프롬프트 |
| `memory/memories.json` | `data/memories.json` | 시드 27건 (place 6 / activity 6 / promise 5 / wish 5 / interest 5) |

**남겨 둔 것** — 되살릴 때 손 안 대도 되게:

- `worker/models.py` 의 `Memory` · `ExtractedMemory` · `ExtractLLMOutput`
- `worker/date_course.py` 의 `plan_date(messages, memories, gate)` · `write_reason(..., memories, ...)`
  시그니처와 `_memory_block()` — 지금은 항상 빈 목록이 들어가고 프롬프트가 "기억 없음"을 처리한다
- `worker/prompts/date_plan.md` · `date_reason.md` 의 "기억을 최우선 근거로" 규칙 — 기억이 비어
  있으면 현재 대화만 근거로 쓰라고 이미 적혀 있어 그대로 동작한다
- `analyze(persist=...)` · CLI `--no-persist` · `KAKAPO_PERSIST` — 아무것도 안 하지만 인터페이스 호환용

## 지금 달라진 것

데이트 코스 추천 이유가 **현재 대화만** 인용한다. "3주 전에 성수 가보고 싶다고 하셨던 거"
류의 과거 인용이 사라진다 — 시연에서 가장 인상적이던 지점이라 되살릴 가치가 있다.
요청당 LLM 호출이 1회(추출) 준다.

## 되살리는 방법

1. 네 파일을 원래 위치로 되돌리고 `requirements.txt` 에 `numpy` 를 다시 넣는다
2. **방 단위로 가른다** — `Memory` 에 `room: str` 을 추가하고, `extract`/`save_memories` 는
   `ctx.request.chat_room_id` 를 찍고, `retrieve_many` 는 같은 방만 검색한다. 시드는 시연
   방에만 줄지 전부 줄지 정한다
3. `router.py`: `DateCandidate.build` 의 `memories: list = []` 자리를
   `retrieve_many(recent_context(ctx.active), k=6, now=ctx.now, kinds=(...), room=...)` 로,
   결과 뒤 `mark_used(...)` 복원. 기억 추출은 `run()` 에서 백그라운드 풀로 던지던 코드가
   git 5e19904 에 있다 (`harvest_memories` · `_background` · `Trace.background` ·
   `analyze(wait_background=True)` — CLI·devui 가 추출 결과를 표시하려고 기다리던 경로)
4. `pipeline.py` · `api.py` 의 `warm_index()` 예열 스레드, Dockerfile 의 `memories.json` COPY 복원

---

## 되살리는 방법

1. `gate.py` · `judge.py` · `topic.py` 를 `worker/` 로, 프롬프트를 `worker/prompts/` 로,
   `daily_questions.json` 을 `data/` 로 되돌린다
2. **import 를 고친다** — 파킹 시점 이후 아래가 바뀌었다
   - `gate.py` 안의 `_norm` / `is_reaction` / `is_question` / `_REACTIONS` 는
     `worker/text.py` 로 빠졌다. 중복 정의하지 말고 `from worker.text import ...` 로 바꾼다
   - `worker.models.Message` 의 `ts` 가 **`sent_at`** 으로 바뀌었고 `message_id` 가 생겼다
     (백엔드 규격의 `sentAt` / `messageId`). `m.ts` → `m.sent_at` 으로 전부 치환한다
   - `Decision` 스키마는 없어졌다. 후보는 이제 `AiResult` 를 반환한다
     (`resultType` / `visibilityType` / `targetParticipant` / `contentType` /
     `triggerMessageIds` / `resultData`). `topic.py` 의 반환부를 여기에 맞춰 다시 쓴다
   - `judge.py` 의 기억 추출은 `worker/extract.py` 로 이미 나가 있다. 되살릴 때는
     판정 부분만 남기고 추출을 중복 실행하지 않도록 한다
3. `worker/router.py` 의 `CANDIDATES` 에 `TopicCandidate()` 를 끼운다 (우선순위 마지막)
4. 백엔드 규격에 `resultType` 값을 추가해야 한다 — 지금 규격에는
   `TONE_CORRECTION` / `DATE_RECOMMENDATION` / `YOUTUBE_RECOMMENDATION` 세 개뿐이라
   **서버 담당자와 재합의가 필요하다.** 워커만 고치면 응답 검증에서 막힌다

## 왜 파킹했나

기능 자체에 문제가 있어서가 아니다. 후보 기능이 3개로 늘면서 우선순위가 밀렸고,
위젯 슬롯 정책이 확정되지 않은 상태라 동작하지 않을 코드를 파이프라인에 두는 것보다
아카이브가 낫다고 판단했다. 검증까지 끝난 코드이므로 되살릴 때 로직은 손댈 필요가 없다.

---

# 파킹 — 말투 기준선 시드 (`speaker_profiles.json`, 2026-09-13)

시연 커플 A·B 의 평소 말투(3개월치 가정)였는데, 방이 여러 개가 되면서 **모든 방이 이 커플의
기준선으로 판정**되고 있었다. 하루 시연이라 방별 누적·서버 집계 대신 코드 안의 **공통
기준선 하나**(`worker/profile.py` 의 `GENERIC_PROFILE`)로 바꿨다 (사용자 결정).

되살릴 일은 없을 것이다 — 방별 기준선이 필요해지면 서버가 `speakerProfiles` 를 실어주는
것이 정석이고(`docs/server-handoff.md` 10장 2번) 워커는 이미 받는 자리가 있다. 이 파일은
그때 값의 모양을 보는 참고용이다.

"""기억 추출 — 대화에서 나중에 다시 꺼낼 것을 뽑아 저장소에 넣는다.

원래 `judge.py`(대화 소재 판정)가 판정과 함께 하던 일이다. 대화 소재 후보가 파킹되면서
추출만 떼어냈다. 안 떼면 기억이 더 이상 쌓이지 않고, 데이트 코스 추천의 근거가 마른다.

**별도 배치 파이프라인을 만들지 않는다.** 분석 요청마다 한 번 같이 돈다.
"""

from __future__ import annotations

import hashlib

from worker.llm import ask, load_prompt
from worker.models import ExtractLLMOutput, Memory, Message
from worker.text import format_transcript


def _memory_id(content: str, quote: str) -> str:
    digest = hashlib.sha1(f"{content}|{quote}".encode()).hexdigest()
    return f"m{digest[:8]}"


def extract_memories(messages: list[Message]) -> list[Memory]:
    if not messages:
        return []

    messages = sorted(messages, key=lambda m: m.sent_at)
    out = ask(ExtractLLMOutput, load_prompt("extract"), format_transcript(messages))

    # 원문에 실제로 존재하는 인용만 남긴다. 지어낸 기억을 저장소에 넣으면
    # 나중에 "언제 그런 말 했지?" 소리가 나온다 — 데이트 코스 추천 이유로 그대로 나가는 값이다.
    corpus = "\n".join(m.content for m in messages)
    memories: list[Memory] = []
    for ext in out.memories:
        if not ext.source_quote or ext.source_quote not in corpus:
            continue
        occurred = next(
            (m.sent_at for m in messages if ext.source_quote in m.content),
            messages[-1].sent_at,
        )
        memories.append(
            Memory(
                id=_memory_id(ext.content, ext.source_quote),
                kind=ext.kind,
                content=ext.content,
                source_quote=ext.source_quote,
                occurred_at=occurred,
                used_at=None,
            )
        )
    return memories

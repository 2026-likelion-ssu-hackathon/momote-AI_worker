"""RAG 기억 검색.

단순 최신순 조회가 아니라 현재 대화 맥락과 의미적으로 유사한 기억을 찾는다.
이 차이가 "AI 가 진짜 듣고 있다"는 느낌을 만든다.

기억이 27건 규모라 InMemoryVectorStore 로 충분하다. 프로세스 시작 시 한 번 인덱싱하고,
후속 단계에서 같은 인터페이스로 pgvector 에 갈아끼운다.

지금 실제로 쓰이는 진입점은 `recent_context()` → `retrieve_many()` 다 (데이트 코스).
`retrieve()`(1건 선택) 와 `reset_index()` 는 파킹된 대화 소재 기능이 쓰던 것으로,
복구할 때 그대로 쓰려고 남겨둔다.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from worker import DATA_DIR
from worker.models import Memory, Message
from worker.text import is_reaction

MEMORY_FILE = DATA_DIR / "memories.json"

# 같은 소재를 최근 30일 안에 다시 던지지 않는다
REUSE_AFTER = timedelta(days=30)

KST = timezone(timedelta(hours=9))

_store: InMemoryVectorStore | None = None
_indexed_ids: set[str] = set()

# 인덱스를 만드는 스레드와 쓰는 스레드가 갈렸다 (`router.run()` 의 병렬 실행 + `warm_index()`).
# 락이 없으면 두 스레드가 동시에 `_store is None` 을 보고 **임베딩을 두 번** 돌리고,
# `_indexed_ids` 갱신이 엇갈려 일부 기억이 인덱스에서 빠진다.
_store_lock = threading.Lock()

# 파일 읽기·쓰기 락. 기억 추출(쓰기)과 데이트 코스의 RAG 검색(읽기)이 **이제 동시에 돈다**
# (`router.run()`). 락이 없으면 쓰는 중인 JSON 을 읽어 파싱이 터진다.
#
# ⚠️ **`_store_lock` 을 잡은 채로 이걸 잡지 말 것.** `_get_store()` 는 store → io 순으로
# 잡고, `save_memories()` 는 io 를 놓은 뒤에 store 를 잡는다. 순서가 엇갈리면 교착이다.
_io_lock = threading.Lock()


def now_kst() -> datetime:
    return datetime.now(KST)


# --------------------------------------------------------------------------
# 파일 I/O
# --------------------------------------------------------------------------
def load_memories() -> list[Memory]:
    with _io_lock:
        return _load_locked()


def _load_locked() -> list[Memory]:
    if not MEMORY_FILE.exists():
        return []
    raw = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    return [Memory(**item) for item in raw]


def _write(memories: list[Memory]) -> None:
    """**`_io_lock` 을 잡은 상태에서만 부른다.**"""
    MEMORY_FILE.write_text(
        json.dumps(
            [json.loads(m.model_dump_json()) for m in memories],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# 인덱싱
# --------------------------------------------------------------------------
def _embed_text(m: Memory) -> str:
    """content 만 넣으면 맥락 매칭이 약하다. 원문 인용을 붙여서 임베딩한다."""
    return f"{m.content} — {m.source_quote}"


def _get_store() -> InMemoryVectorStore:
    global _store
    with _store_lock:
        if _store is None:
            embeddings = OpenAIEmbeddings(
                model=os.getenv("KAKAPO_EMBEDDING_MODEL", "text-embedding-3-small")
            )
            _store = InMemoryVectorStore(embeddings)
            _index_locked(_load_locked())
        return _store


def _index_locked(memories: list[Memory]) -> None:
    """**`_store_lock` 을 잡은 상태에서만 부른다.**"""
    fresh = [m for m in memories if m.id not in _indexed_ids]
    if not fresh or _store is None:
        return
    _store.add_texts(
        [_embed_text(m) for m in fresh],
        metadatas=[{"id": m.id} for m in fresh],
    )
    _indexed_ids.update(m.id for m in fresh)


def _index(memories: list[Memory]) -> None:
    with _store_lock:
        _index_locked(memories)


def warm_index() -> None:
    """인덱스를 미리 만들어 둔다. **결과를 바꾸지 않고 시점만 앞당긴다.**

    27건을 임베딩하는 데 2.6초가 걸리는데, 지금은 그게 **데이트 코스 경로 한가운데**서
    일어난다 (`retrieve_many` → `search` → `_get_store`). 첫 요청의 임계 경로에 그대로
    얹히는 값이다.

    분절 LLM 호출이 도는 동안은 어차피 기다리는 시간이라 거기 겹쳐 둔다.
    실패해도 조용히 넘어간다 — 뒤에서 `_get_store()` 가 다시 시도하고, 그때 나는 오류가
    진짜 오류다. 여기서 던지면 워밍업 실패가 요청 실패로 둔갑한다.
    """
    try:
        _get_store()
    except Exception:  # noqa: BLE001
        pass


def reset_index() -> None:
    """테스트용 — 인덱스를 비운다."""
    global _store
    with _store_lock:
        _store = None
        _indexed_ids.clear()


# --------------------------------------------------------------------------
# 검색
# --------------------------------------------------------------------------
def recent_context(messages: list[Message], n: int = 6) -> str:
    """유사도 검색 질의로 쓸 최근 대화.

    리액션("ㅇㅇ", "ㅋㅋ")은 제외하고 내용이 있는 발화만 모은다.
    트리거가 걸리는 대화는 끝부분이 리액션으로 채워져 있어서, 단순히 마지막 n개를
    쓰면 질의가 "ㅇㅇ 응 그래"가 되고 검색이 통째로 헛돈다.
    """
    ordered = sorted(messages, key=lambda m: m.sent_at)
    meaningful = [m for m in ordered if not is_reaction(m.content)]
    return " ".join(m.content for m in (meaningful or ordered)[-n:])


def search(query: str, k: int = 3) -> list[Memory]:
    """유사도 상위 k건. 순위 그대로 돌려준다 (디버깅·시연용)."""
    store = _get_store()
    by_id = {m.id: m for m in load_memories()}
    hits: list[Memory] = []
    for doc in store.similarity_search(query, k=k):
        m = by_id.get(doc.metadata.get("id"))
        if m is not None:
            hits.append(m)
    return hits


def retrieve_many(
    recent: str,
    k: int = 5,
    now: datetime | None = None,
    kinds: tuple[str, ...] | None = None,
) -> list[Memory]:
    """맥락과 유사한 기억 **여러 건**. 데이트 코스 추천이 근거로 쓴다.

    소재 제시는 기억 1건을 골라 문장 하나를 만들면 됐지만, 데이트 코스는
    장소·음식·취향·일정을 조합해야 해서 여러 건이 한꺼번에 필요하다.
    """
    now = now or now_kst()
    hits = search(recent, k=k * 3 if kinds else k)
    if kinds:
        hits = [m for m in hits if m.kind in kinds]
    usable = [m for m in hits if m.used_at is None or now - m.used_at >= REUSE_AFTER]
    return usable[:k]


def retrieve(recent: str, k: int = 3, now: datetime | None = None) -> Memory | None:
    """맥락과 유사한 기억 중 쓸 수 있는 것 하나. 없으면 None.

    ⏸️ 파킹된 대화 소재 기능 전용이다. 지금 파이프라인에서는 호출되지 않는다.
    """
    now = now or now_kst()
    hits = search(recent, k=k)

    # 미소환 우선
    for m in hits:
        if m.used_at is None:
            return m

    # 전부 소환됐다면 30일 지난 것만 재사용
    for m in hits:
        if m.used_at is not None and now - m.used_at >= REUSE_AFTER:
            return m

    return None


# --------------------------------------------------------------------------
# 쓰기
# --------------------------------------------------------------------------
def mark_used(memory_id: str, now: datetime | None = None, persist: bool = True) -> None:
    if not persist:
        return
    with _io_lock:
        memories = _load_locked()
        for m in memories:
            if m.id == memory_id:
                m.used_at = now or now_kst()
                break
        else:
            return
        _write(memories)


def save_memories(new: list[Memory], persist: bool = True) -> list[Memory]:
    """`extract.py` 가 뽑은 기억을 저장소에 넣는다. 같은 id 는 건너뛴다."""
    if not new:
        return []

    with _io_lock:
        memories = _load_locked()
        known = {m.id for m in memories}
        added = [m for m in new if m.id not in known]
        if not added:
            return []
        memories.extend(added)
        if persist:
            _write(memories)

    # **락을 놓고 인덱싱한다.** `_index` 가 `_store_lock` 을 잡는데, io 를 쥔 채로 잡으면
    # `_get_store()`(store → io)와 순서가 엇갈려 교착이 된다.
    if _store is not None:
        _index(added)
    return added

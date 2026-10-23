"""Векторная БД (ChromaDB). Одна коллекция на репозиторий, ключ кэша — SHA коммита."""

from __future__ import annotations

import hashlib
import logging
import re

import chromadb
from chromadb.config import Settings

from .embeddings import Embedder, embedding_text
from .models import CodeChunk, SearchHit

log = logging.getLogger(__name__)


def collection_name(owner: str, repo: str) -> str:
    """Допустимое для Chroma имя коллекции (3–63 символа, [a-zA-Z0-9._-])."""
    slug = re.sub(r"[^a-z0-9_-]", "_", f"{owner}_{repo}".lower())[:40]  # имена на GitHub регистронезависимы
    digest = hashlib.md5(f"{owner}/{repo}".lower().encode()).hexdigest()[:8]
    return f"r-{slug}-{digest}"


class VectorStore:
    def __init__(self, path: str, embedder: Embedder):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))

    def _get(self, owner: str, repo: str):
        try:
            return self.client.get_collection(collection_name(owner, repo))
        except Exception:  # коллекции нет (тип исключения зависит от версии chromadb)
            return None

    def is_indexed(self, owner: str, repo: str, ref: str) -> bool:
        col = self._get(owner, repo)
        if col is None:
            return False
        meta = col.metadata or {}
        return meta.get("ref") == ref and meta.get("embedder") == self.embedder.name and col.count() > 0

    def count(self, owner: str, repo: str) -> int:
        col = self._get(owner, repo)
        return col.count() if col is not None else 0

    def index(self, owner: str, repo: str, ref: str, chunks: list[CodeChunk], batch_size: int = 64) -> int:
        """Пересоздаёт коллекцию репозитория и кладёт в неё все фрагменты."""
        name = collection_name(owner, repo)
        if self._get(owner, repo) is not None:
            self.client.delete_collection(name)
        col = self.client.create_collection(
            name, metadata={"hnsw:space": "cosine", "ref": ref, "embedder": self.embedder.name}
        )
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            col.add(
                ids=[c.id for c in batch],
                documents=[c.code for c in batch],
                embeddings=self.embedder.embed([embedding_text(c) for c in batch]),
                metadatas=[
                    {"path": c.path, "symbol": c.symbol, "kind": c.kind, "start_line": c.start_line, "end_line": c.end_line}
                    for c in batch
                ],
            )
            log.info("Проиндексировано %d/%d фрагментов", min(i + batch_size, len(chunks)), len(chunks))
        return len(chunks)

    def query_many(self, owner: str, repo: str, texts: list[str], n_results: int = 10) -> list[list[SearchHit]]:
        """Для каждого запроса возвращает ближайшие фрагменты (по убыванию близости)."""
        col = self._get(owner, repo)
        if col is None or col.count() == 0 or not texts:
            return [[] for _ in texts]
        n = min(n_results, col.count())
        vectors = self.embedder.embed(texts)
        results = []
        for vector in vectors:
            res = col.query(query_embeddings=[vector], n_results=n)
            hits = []
            for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
                chunk = CodeChunk(
                    id=cid, path=meta["path"], symbol=meta["symbol"], kind=meta["kind"],
                    start_line=int(meta["start_line"]), end_line=int(meta["end_line"]), code=doc,
                )
                hits.append(SearchHit(chunk, 1.0 - float(dist)))
            results.append(hits)
        return results

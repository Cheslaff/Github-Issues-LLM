"""Code Retriever Agent: ищет в VectorDB код, затронутый issue, и собирает список файлов."""

from __future__ import annotations

import json
import logging

from ..llm import LLMClient, LLMError
from ..models import CodeChunk, IssueMetadata
from ..vector_store import VectorStore

log = logging.getLogger(__name__)

RRF_K = 60  # константа Reciprocal Rank Fusion
MENTIONED_BOOST = 1.5

RERANK_PROMPT = """You help a developer find the source files affected by a GitHub issue.
You get the issue and candidate files with code snippets found by semantic search.
For EVERY candidate decide whether the issue most likely requires reading or changing it.
Return ONLY a JSON object: {"files": [{"path": "<candidate path>", "relevant": true|false, \
"reason": "<one short sentence in English>"}]}"""


def _blob_url(meta: IssueMetadata, path: str, start: int | None = None, end: int | None = None) -> str:
    url = f"{meta.issue.ref.repo_url}/blob/{meta.commit}/{path}"
    return f"{url}#L{start}-L{end}" if start else url


def _is_mentioned(path: str, mentioned: list[str]) -> bool:
    for m in mentioned:
        if m == path or path.endswith("/" + m) or m.endswith("/" + path):
            return True
    return False


def _snippet(code: str, max_lines: int = 12, max_chars: int = 800) -> str:
    return "\n".join(code.splitlines()[:max_lines])[:max_chars]


class CodeRetrieverAgent:
    def __init__(
        self,
        store: VectorStore,
        llm: LLMClient | None = None,
        top_k_files: int = 5,
        chunks_per_query: int = 12,
        max_queries: int = 6,
        chunks_per_file: int = 3,
    ):
        self.store = store
        self.llm = llm
        self.top_k_files = top_k_files
        self.chunks_per_query = chunks_per_query
        self.max_queries = max_queries
        self.chunks_per_file = chunks_per_file

    # ---- 1. запросы -------------------------------------------------------------------
    def build_queries(self, meta: IssueMetadata) -> list[str]:
        c = meta.classification
        candidates = list(c.search_queries) + [meta.issue.title]
        if c.keywords:
            candidates.append(" ".join(c.keywords[:8]))
        body = (meta.issue.body or "").strip()
        if body:
            candidates.append(body[:500])
        queries, seen = [], set()
        for q in candidates:
            q = " ".join(q.split())
            if q and q.lower() not in seen:
                seen.add(q.lower())
                queries.append(q)
        return queries[: self.max_queries]

    # ---- 2. поиск и слияние -----------------------------------------------------------
    def _search(self, meta: IssueMetadata, queries: list[str]) -> list[dict]:
        ref = meta.issue.ref
        results = self.store.query_many(ref.owner, ref.repo, queries, self.chunks_per_query)

        fused: dict[str, float] = {}
        best_sim: dict[str, float] = {}
        chunks: dict[str, CodeChunk] = {}
        for hits in results:
            for rank, hit in enumerate(hits):
                if hit.similarity <= 0:  # запрос ничего не нашёл — нулевое совпадение не вклад
                    continue
                cid = hit.chunk.id
                chunks[cid] = hit.chunk
                # RRF, взвешенный близостью: слабые совпадения почти не влияют на итог.
                fused[cid] = fused.get(cid, 0.0) + hit.similarity / (RRF_K + rank + 1)
                best_sim[cid] = max(best_sim.get(cid, -1.0), hit.similarity)

        mentioned = meta.classification.mentioned_files
        for cid, chunk in chunks.items():
            if _is_mentioned(chunk.path, mentioned):
                fused[cid] *= MENTIONED_BOOST

        by_file: dict[str, list[tuple[float, CodeChunk]]] = {}
        for cid, score in fused.items():
            by_file.setdefault(chunks[cid].path, []).append((score, chunks[cid]))

        files = []
        for path, items in by_file.items():
            items.sort(key=lambda x: x[0], reverse=True)
            top = items[: self.chunks_per_file]
            files.append(
                {
                    "path": path,
                    # Лучший фрагмент важнее всего; остальные добавляют бонус, но не перевешивают его.
                    "_raw_score": top[0][0] + 0.25 * sum(score for score, _ in top[1:]),
                    "_mentioned": _is_mentioned(path, mentioned),
                    "chunks": [
                        {
                            "symbol": ch.symbol,
                            "kind": ch.kind,
                            "start_line": ch.start_line,
                            "end_line": ch.end_line,
                            "similarity": round(best_sim[ch.id], 3),
                            "url": _blob_url(meta, ch.path, ch.start_line, ch.end_line),
                            "snippet": _snippet(ch.code),
                            "_code": ch.code,
                        }
                        for _, ch in top
                    ],
                }
            )
        files.sort(key=lambda f: f["_raw_score"], reverse=True)
        return files

    # ---- 3. LLM-проверка (необязательна) ----------------------------------------------
    def _rerank(self, meta: IssueMetadata, files: list[dict]) -> bool:
        if self.llm is None or not files:
            return False
        c = meta.classification
        payload = {
            "issue": {
                "title": meta.issue.title,
                "type": c.issue_type,
                "summary": c.summary,
                "body": (meta.issue.body or "")[:1500],
            },
            "candidates": [
                {
                    "path": f["path"],
                    "snippets": [
                        {"symbol": ch["symbol"], "lines": f'{ch["start_line"]}-{ch["end_line"]}', "code": ch["_code"][:700]}
                        for ch in f["chunks"][:2]
                    ],
                }
                for f in files
            ],
        }
        try:
            answer = self.llm.chat_json(RERANK_PROMPT, json.dumps(payload, ensure_ascii=False))
        except LLMError as exc:
            log.warning("LLM-проверка кандидатов не удалась (%s), оставляем векторный порядок", exc)
            return False
        verdicts = {}
        for item in answer.get("files", []):
            if isinstance(item, dict) and "path" in item:
                verdicts[item["path"]] = (bool(item.get("relevant", True)), str(item.get("reason", "")).strip())
        for f in files:
            if f["path"] in verdicts:
                relevant, reason = verdicts[f["path"]]
                f["llm_relevant"] = relevant
                if reason:
                    f["reason"] = reason
        files.sort(key=lambda f: f.get("llm_relevant") is False)  # стабильно: «нерелевантные» в конец
        return True

    # ---- главный метод ----------------------------------------------------------------
    def run(self, meta: IssueMetadata) -> dict:
        queries = self.build_queries(meta)
        files = self._search(meta, queries)
        max_score = max((f["_raw_score"] for f in files), default=1.0) or 1.0

        # Берём с запасом, чтобы LLM могла отсеять лишнее; без LLM — сразу top-k.
        candidates = files[: self.top_k_files * (2 if self.llm else 1)]
        for f in candidates:
            best = max(ch["similarity"] for ch in f["chunks"])
            symbols = ", ".join(dict.fromkeys(ch["symbol"] for ch in f["chunks"][:3]))
            reason = f"Semantic match: {symbols} (best similarity {best:.2f})"
            if f["_mentioned"]:
                reason += "; file is mentioned in the issue"
            f["reason"] = reason
            f["llm_relevant"] = None
        llm_used = self._rerank(meta, candidates)

        result = []
        for f in candidates[: self.top_k_files]:
            result.append(
                {
                    "path": f["path"],
                    "score": round(f["_raw_score"] / max_score, 3),
                    "url": _blob_url(meta, f["path"]),
                    "reason": f["reason"],
                    "llm_relevant": f["llm_relevant"],
                    "chunks": [{k: v for k, v in ch.items() if not k.startswith("_")} for ch in f["chunks"]],
                }
            )
        return {"queries": queries, "files": result, "llm_used": llm_used}

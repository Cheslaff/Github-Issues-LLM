"""Пайплайн: Issue URL -> GitHub API -> Issue + Repository -> Classification
-> Repository Ingestion -> Chunking -> VectorDB -> Code Retriever Agent -> JSON Report."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .agents import CodeRetrieverAgent, TicketClassificationAgent
from .chunker import chunk_file
from .config import Config
from .embeddings import make_embedder
from .github_client import GitHubClient, parse_issue_url
from .llm import LLMClient
from .models import IssueMetadata
from .vector_store import VectorStore

log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    pass


class Pipeline:
    def __init__(self, config: Config | None = None, github=None, store=None, llm=None):
        self.config = config or Config.from_env()
        cfg = self.config
        self.github = github or GitHubClient(
            token=cfg.github_token,
            max_repo_mb=cfg.max_repo_mb,
            max_file_bytes=cfg.max_file_bytes,
            max_files=cfg.max_files,
        )
        self.store = store or VectorStore(cfg.chroma_dir, make_embedder(cfg.embedder))
        if llm is None and cfg.llm_api_key:
            llm = LLMClient(cfg.llm_api_key, cfg.llm_base_url, cfg.llm_model)
        self.llm = llm

    def _ensure_indexed(self, owner: str, repo: str, sha: str, reindex: bool) -> str:
        """Скачивает и индексирует репозиторий, если этого коммита ещё нет в VectorDB."""
        if not reindex and self.store.is_indexed(owner, repo, sha):
            log.info("Индекс %s/%s@%s найден в кэше", owner, repo, sha[:7])
            return "cached"
        log.info("Скачиваем %s/%s@%s", owner, repo, sha[:7])
        files = self.github.download_python_files(owner, repo, sha)
        if not files:
            raise PipelineError("В репозитории не найдено Python-файлов.")
        chunks = [chunk for f in files for chunk in chunk_file(f)]
        log.info("Python-файлов: %d, фрагментов: %d — индексируем", len(files), len(chunks))
        self.store.index(owner, repo, sha, chunks)
        return "fresh"

    def run(self, issue_url: str, top_k: int = 5, reindex: bool = False, use_llm: bool = True) -> dict:
        llm = self.llm if use_llm else None
        ref = parse_issue_url(issue_url)

        issue = self.github.get_issue(ref)
        sha = self.github.get_head_sha(ref.owner, ref.repo)
        index_state = self._ensure_indexed(ref.owner, ref.repo, sha, reindex)

        classification = TicketClassificationAgent(llm).classify(issue)
        log.info("Классификация (%s): %s", classification.source, classification.issue_type)
        meta = IssueMetadata(issue=issue, commit=sha, classification=classification)

        result = CodeRetrieverAgent(self.store, llm, top_k_files=top_k).run(meta)

        return {
            "metadata": meta.to_dict(max_body=2000),
            "affected_files": result["files"],
            "run": {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "queries": result["queries"],
                "index": index_state,
                "chunks_in_index": self.store.count(ref.owner, ref.repo),
                "embedder": self.store.embedder.name,
                "llm_classification": classification.source == "llm",
                "llm_rerank": result["llm_used"],
            },
        }

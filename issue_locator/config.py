"""Конфигурация из переменных окружения / файла .env."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Config:
    # GitHub: токен необязателен, но без него лимит всего 60 запросов в час.
    github_token: str | None = None

    # LLM: любой OpenAI-совместимый API (по умолчанию DeepSeek).
    # Если ключ не задан, проект работает без LLM (эвристики + чистый векторный поиск).
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"

    # Векторная БД (ChromaDB, хранится локально в папке).
    chroma_dir: str = ".chroma"
    # "default" — модель all-MiniLM-L6-v2 (ONNX, скачивается при первом запуске),
    # "hash"    — лёгкий оффлайн-эмбеддер без нейросети (для тестов и отладки).
    embedder: str = "default"

    # Ограничения при загрузке репозитория.
    max_file_bytes: int = 200_000
    max_files: int = 3000
    max_repo_mb: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        env = os.environ.get
        return cls(
            github_token=env("GITHUB_TOKEN") or None,
            llm_api_key=env("LLM_API_KEY") or env("DEEPSEEK_API_KEY") or None,
            llm_base_url=env("LLM_BASE_URL") or cls.llm_base_url,
            llm_model=env("LLM_MODEL") or cls.llm_model,
            chroma_dir=env("CHROMA_DIR") or cls.chroma_dir,
            embedder=env("EMBEDDER") or cls.embedder,
            max_file_bytes=int(env("MAX_FILE_BYTES") or cls.max_file_bytes),
            max_files=int(env("MAX_FILES") or cls.max_files),
            max_repo_mb=int(env("MAX_REPO_MB") or cls.max_repo_mb),
        )

"""Командная строка: python -m issue_locator <ссылка на issue>."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config
from .github_client import GitHubError
from .llm import LLMError
from .pipeline import Pipeline, PipelineError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="issue_locator",
        description="По ссылке на GitHub issue находит затронутые файлы репозитория и печатает JSON-отчёт.",
    )
    p.add_argument("issue_url", help="https://github.com/<owner>/<repo>/issues/<number>")
    p.add_argument("--top-k", type=int, default=5, help="сколько файлов вернуть (по умолчанию 5)")
    p.add_argument("--out", help="сохранить отчёт в файл вместо вывода в консоль")
    p.add_argument("--reindex", action="store_true", help="переиндексировать репозиторий, игнорируя кэш")
    p.add_argument("--no-llm", action="store_true", help="не обращаться к LLM (эвристики + векторный поиск)")
    p.add_argument("--embedder", choices=["default", "hash"], help="переопределить EMBEDDER из .env")
    p.add_argument("-v", "--verbose", action="store_true", help="подробные логи")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    config = Config.from_env()
    if args.embedder:
        config.embedder = args.embedder

    try:
        report = Pipeline(config).run(args.issue_url, top_k=args.top_k, reindex=args.reindex, use_llm=not args.no_llm)
    except (ValueError, GitHubError, PipelineError, LLMError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Отчёт сохранён: {args.out}", file=sys.stderr)
    else:
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # Windows-консоль с cp1251/cp866
        except AttributeError:
            pass
        print(text)
    return 0

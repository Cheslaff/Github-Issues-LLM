"""Ticket Classification Agent: определяет тип issue и формирует поисковые запросы."""

from __future__ import annotations

import logging
import re
from collections import Counter

from ..llm import LLMClient, LLMError
from ..models import Classification, Issue

log = logging.getLogger(__name__)

ISSUE_TYPES = ("bug", "feature", "documentation", "performance", "question", "refactor", "other")

SYSTEM_PROMPT = """You analyse GitHub issues of Python projects to help find the affected source code.
Return ONLY a JSON object with these keys:
- "issue_type": one of bug, feature, documentation, performance, question, refactor, other
- "summary": 1-2 sentences in English: what exactly is wrong or requested
- "keywords": 5-10 identifiers or terms likely to appear in the source code (function, class, module, \
exception names, config options). Prefer exact names mentioned in the issue
- "search_queries": 2-4 short English queries describing the behaviour of the code that has to be \
found or changed (what the code does, not how the issue is worded)"""

_STOP_WORDS = {
    "the", "and", "for", "with", "this", "that", "when", "from", "does", "not", "are", "was", "have",
    "has", "into", "after", "before", "should", "would", "could", "error", "issue", "problem", "using",
    "use", "get", "set", "add", "fix", "support", "feature", "bug", "doesn", "can", "cannot", "unable",
}
_TYPE_RULES = [
    ("bug", ["bug", "error", "crash", "exception", "traceback", "fail", "broken", "regression", "wrong", "incorrect"]),
    ("performance", ["performance", "slow", "memory leak", "speed", "latency", "optimi"]),
    ("feature", ["feature", "enhancement", "proposal", "add support", "would be nice", "request"]),
    ("documentation", ["documentation", "docs?\\b", "typo", "readme", "docstring"]),
    ("question", ["question", "how to", "how do i", "how can i", "help"]),
    ("refactor", ["refactor", "cleanup", "clean up", "deprecat"]),
]
_DOTTED_SKIP_TAIL = {"com", "org", "io", "net", "md", "txt", "html", "py", "dev", "app", "ru"}


def _has_any(text: str, words: list[str]) -> bool:
    return re.search(r"\b(?:%s)" % "|".join(words), text) is not None


def extract_mentioned_files(text: str) -> list[str]:
    """Пути к .py файлам, упомянутые в тексте или в traceback."""
    # Сначала пути из traceback (они полные), затем любые упоминания вида dir/file.py.
    found = re.findall(r'File "([^"]+\.py)"', text) + re.findall(r"[\w./\\-]+\.py\b", text)
    result: list[str] = []
    for path in found:
        path = re.sub(r"^(?:\./)+", "", path.replace("\\", "/")).lstrip("/")
        if path and not any(known == path or known.endswith("/" + path) for known in result):
            result.append(path)
    return result[:10]


def extract_keywords(title: str, body: str, limit: int = 10) -> list[str]:
    """Кодоподобные токены из текста: `в бэктиках`, snake_case, CamelCase, a.b.c."""
    counter: Counter[str] = Counter()
    for text, weight in ((title, 3), (body, 1)):
        for token in re.findall(r"`([^`\n]{2,60})`", text):
            token = token.strip().rstrip("()")
            if re.fullmatch(r"[\w.]+", token):
                counter[token] += 2 * weight
        for token in re.findall(r"\b[a-z]+(?:_[a-z0-9]+)+\b", text):
            counter[token] += weight
        # CamelCase, в том числе с аббревиатурами: KeyError, HTTPAdapter, JSONDecodeError.
        for token in re.findall(r"\b(?=[A-Za-z0-9]*[a-z])(?=(?:[a-z0-9]*[A-Z]){2})[A-Z][A-Za-z0-9]*\b", text):
            counter[token] += weight
        for token in re.findall(r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+\b", text):
            if len(token) >= 4 and token.rsplit(".", 1)[-1].lower() not in _DOTTED_SKIP_TAIL:
                counter[token] += weight
    keywords = [k for k, _ in counter.most_common(limit)]
    if len(keywords) < 3:  # в тексте нет кода — берём значимые слова заголовка
        known = {k.lower() for k in keywords}
        for word in re.findall(r"[A-Za-z]{4,}", title.lower()):
            if word not in _STOP_WORDS and word not in known:
                keywords.append(word)
                known.add(word)
    return keywords[:limit]


def heuristic_classify(issue: Issue) -> Classification:
    """Запасной вариант без LLM: правила по заголовку/меткам + извлечение идентификаторов."""
    body = issue.body or ""
    haystack = f"{' '.join(issue.labels)} {issue.title}".lower()
    issue_type = "other"
    for name, words in _TYPE_RULES:
        if _has_any(haystack, words):
            issue_type = name
            break
    else:
        if "Traceback (most recent call last)" in body:
            issue_type = "bug"
    keywords = extract_keywords(issue.title, body)
    queries = [issue.title.strip()]
    if keywords:
        queries.append(" ".join(keywords[:6]))
    return Classification(
        issue_type=issue_type,
        summary=issue.title.strip(),
        keywords=keywords,
        search_queries=[q for q in queries if q],
        mentioned_files=extract_mentioned_files(f"{issue.title}\n{body}"),
        source="heuristic",
    )


def _str_list(value, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:limit]


class TicketClassificationAgent:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

    def classify(self, issue: Issue) -> Classification:
        fallback = heuristic_classify(issue)
        if self.llm is None:
            return fallback
        user = f"Title: {issue.title}\nLabels: {', '.join(issue.labels) or '-'}\n\nBody:\n{(issue.body or '')[:6000]}"
        try:
            data = self.llm.chat_json(SYSTEM_PROMPT, user)
        except LLMError as exc:
            log.warning("LLM-классификация не удалась (%s), используем эвристики", exc)
            return fallback
        issue_type = str(data.get("issue_type", "")).strip().lower()
        keywords = _str_list(data.get("keywords"), 10) or fallback.keywords
        queries = _str_list(data.get("search_queries"), 4) or fallback.search_queries
        return Classification(
            issue_type=issue_type if issue_type in ISSUE_TYPES else "other",
            summary=str(data.get("summary") or fallback.summary).strip(),
            keywords=keywords,
            search_queries=queries,
            mentioned_files=fallback.mentioned_files,  # пути надёжнее достаются регулярками
            source="llm",
        )

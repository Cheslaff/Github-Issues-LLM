"""Модели данных, которые передаются между этапами пайплайна."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class IssueRef:
    owner: str
    repo: str
    number: int

    @property
    def repo_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}"

    @property
    def url(self) -> str:
        return f"{self.repo_url}/issues/{self.number}"


@dataclass
class Issue:
    ref: IssueRef
    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    state: str = "open"


@dataclass
class SourceFile:
    path: str
    content: str


@dataclass
class CodeChunk:
    id: str
    path: str
    symbol: str  # имя функции / класса / "Class.method" / "<module>"
    kind: str  # function | method | class | module | window
    start_line: int
    end_line: int
    code: str


@dataclass
class SearchHit:
    chunk: CodeChunk
    similarity: float  # косинусная близость, 1.0 = идентичный вектор


@dataclass
class Classification:
    issue_type: str  # bug | feature | documentation | performance | question | refactor | other
    summary: str
    keywords: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    mentioned_files: list[str] = field(default_factory=list)
    source: str = "heuristic"  # llm | heuristic


@dataclass
class IssueMetadata:
    """Формат, в котором результат классификации передаётся Code Retriever Agent."""

    issue: Issue
    commit: str
    classification: Classification

    def to_dict(self, max_body: int | None = None) -> dict:
        body = self.issue.body or ""
        if max_body is not None and len(body) > max_body:
            body = body[:max_body] + "…"
        ref = self.issue.ref
        return {
            "issue": {
                "url": ref.url,
                "number": ref.number,
                "title": self.issue.title,
                "body": body,
                "labels": self.issue.labels,
                "state": self.issue.state,
            },
            "repository": {
                "owner": ref.owner,
                "name": ref.repo,
                "url": ref.repo_url,
                "commit": self.commit,
            },
            "classification": asdict(self.classification),
        }

"""Работа с GitHub: разбор ссылки на issue, загрузка issue и Python-файлов репозитория."""

from __future__ import annotations

import io
import logging
import re
import zipfile

import requests

from .models import Issue, IssueRef, SourceFile

log = logging.getLogger(__name__)


class GitHubError(RuntimeError):
    pass


_ISSUE_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)"
    r"/issues/(?P<number>\d+)/?(?:[?#].*)?$"
)


def parse_issue_url(url: str) -> IssueRef:
    """https://github.com/owner/repo/issues/123 -> IssueRef."""
    match = _ISSUE_RE.match(url.strip())
    if not match:
        raise ValueError(
            "Ожидалась ссылка вида https://github.com/<owner>/<repo>/issues/<number>, "
            f"получено: {url!r}"
        )
    return IssueRef(match["owner"], match["repo"], int(match["number"]))


# Каталоги и файлы, которые не несут смысла для поиска по коду.
SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env", "__pycache__",
    "build", "dist", ".tox", ".nox", ".eggs", "site-packages", ".mypy_cache", ".pytest_cache",
}
SKIP_SUFFIXES = ("_pb2.py", "_pb2_grpc.py")


def extract_python_files(
    data: bytes, max_file_bytes: int = 200_000, max_files: int = 3000
) -> list[SourceFile]:
    """Достаёт .py файлы из zip-архива репозитория (в архиве GitHub есть корневая папка)."""
    files: list[SourceFile] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = info.filename.split("/")[1:]  # отбрасываем корневую папку архива
            if not parts or not parts[-1].endswith(".py") or parts[-1].endswith(SKIP_SUFFIXES):
                continue
            if any(p in SKIP_DIRS or p.endswith(".egg-info") for p in parts[:-1]):
                continue
            if info.file_size == 0 or info.file_size > max_file_bytes:
                continue
            raw = archive.read(info)
            if b"\x00" in raw:
                continue
            files.append(SourceFile("/".join(parts), raw.decode("utf-8", errors="replace")))
    files.sort(key=lambda f: f.path)
    if len(files) > max_files:
        log.warning("Найдено %d файлов, берём первые %d", len(files), max_files)
        files = files[:max_files]
    return files


class GitHubClient:
    API = "https://api.github.com"

    def __init__(
        self,
        token: str | None = None,
        session: requests.Session | None = None,
        timeout: int = 60,
        max_repo_mb: int = 100,
        max_file_bytes: int = 200_000,
        max_files: int = 3000,
    ):
        self.token = token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_repo_mb = max_repo_mb
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files

    def _headers(self, accept: str = "application/vnd.github+json") -> dict:
        headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "issue-code-locator"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    @staticmethod
    def _check(response: requests.Response) -> None:
        code = response.status_code
        if code < 400:
            return
        if code == 404:
            raise GitHubError("Не найдено (неверная ссылка или приватный репозиторий без токена).")
        if code in (401, 403):
            if response.headers.get("X-RateLimit-Remaining") == "0":
                raise GitHubError("Исчерпан лимит GitHub API. Задайте GITHUB_TOKEN в .env.")
            raise GitHubError(f"Доступ запрещён ({code}). Проверьте GITHUB_TOKEN.")
        raise GitHubError(f"Ошибка GitHub API {code}: {response.text[:200]}")

    def get_issue(self, ref: IssueRef) -> Issue:
        url = f"{self.API}/repos/{ref.owner}/{ref.repo}/issues/{ref.number}"
        response = self.session.get(url, headers=self._headers(), timeout=self.timeout)
        self._check(response)
        data = response.json()
        return Issue(
            ref=ref,
            title=data.get("title") or "",
            body=data.get("body") or "",
            labels=[label["name"] for label in data.get("labels", []) if isinstance(label, dict)],
            state=data.get("state", "open"),
        )

    def get_head_sha(self, owner: str, repo: str) -> str:
        """SHA последнего коммита default-ветки (нужен как ключ кэша индекса)."""
        url = f"{self.API}/repos/{owner}/{repo}/commits/HEAD"
        response = self.session.get(
            url, headers=self._headers("application/vnd.github.sha"), timeout=self.timeout
        )
        self._check(response)
        return response.text.strip()

    def download_python_files(self, owner: str, repo: str, sha: str) -> list[SourceFile]:
        """Скачивает архив репозитория на коммите `sha` одним запросом и достаёт из него .py файлы."""
        url = f"{self.API}/repos/{owner}/{repo}/zipball/{sha}"
        response = self.session.get(url, headers=self._headers(), timeout=self.timeout, stream=True)
        self._check(response)
        limit = self.max_repo_mb * 1024 * 1024
        buffer = io.BytesIO()
        for chunk in response.iter_content(1 << 16):
            buffer.write(chunk)
            if buffer.tell() > limit:
                response.close()
                raise GitHubError(f"Архив репозитория больше {self.max_repo_mb} МБ — слишком большой.")
        return extract_python_files(buffer.getvalue(), self.max_file_bytes, self.max_files)

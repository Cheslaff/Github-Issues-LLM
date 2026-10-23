import pytest
import requests

from fakes import build_zip
from issue_locator.github_client import GitHubClient, GitHubError, extract_python_files, parse_issue_url
from issue_locator.models import IssueRef


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/psf/requests/issues/6000",
        "http://github.com/psf/requests/issues/6000/",
        "https://www.github.com/psf/requests/issues/6000#issuecomment-1",
        "  https://github.com/psf/requests/issues/6000?foo=bar ",
    ],
)
def test_parse_issue_url_ok(url):
    assert parse_issue_url(url) == IssueRef("psf", "requests", 6000)


@pytest.mark.parametrize(
    "url",
    ["https://github.com/psf/requests", "https://github.com/psf/requests/pull/1", "https://example.com/a/b/issues/1", "abc"],
)
def test_parse_issue_url_bad(url):
    with pytest.raises(ValueError):
        parse_issue_url(url)


def test_extract_python_files_filters():
    data = build_zip(
        {
            "pkg/a.py": "x = 1\n",
            "pkg/sub/b.py": "y = 2\n",
            "README.md": "# hi",
            "venv/lib/c.py": "z = 3\n",
            "node_modules/d.py": "z = 3\n",
            "proto_pb2.py": "generated = 1\n",
            "empty.py": "",
            "big.py": "a = 1\n" * 100,
            "bin.py": "a = 1\x00",
        }
    )
    files = extract_python_files(data, max_file_bytes=100)
    assert [f.path for f in files] == ["pkg/a.py", "pkg/sub/b.py"]
    assert files[0].content == "x = 1\n"


def test_extract_python_files_max_files():
    data = build_zip({f"m{i}.py": "x = 1\n" for i in range(5)})
    assert len(extract_python_files(data, max_files=3)) == 3


class _Resp:
    def __init__(self, status=200, json_data=None, text="", headers=None, content=b""):
        self.status_code, self._json, self.text = status, json_data, text
        self.headers = headers or {}
        self._content = content

    def json(self):
        return self._json

    def iter_content(self, size):
        for i in range(0, len(self._content), size):
            yield self._content[i : i + size]

    def close(self):
        pass


class _Session:
    def __init__(self, response):
        self.response, self.calls = response, []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_get_issue():
    resp = _Resp(json_data={"title": "T", "body": None, "labels": [{"name": "bug"}], "state": "open"})
    client = GitHubClient(token="tok", session=_Session(resp))
    issue = client.get_issue(IssueRef("o", "r", 5))
    assert (issue.title, issue.body, issue.labels) == ("T", "", ["bug"])
    url, kwargs = client.session.calls[0]
    assert url.endswith("/repos/o/r/issues/5")
    assert kwargs["headers"]["Authorization"] == "Bearer tok"


def test_get_head_sha():
    client = GitHubClient(session=_Session(_Resp(text="deadbeef\n")))
    assert client.get_head_sha("o", "r") == "deadbeef"


def test_errors():
    ref = IssueRef("o", "r", 1)
    with pytest.raises(GitHubError, match="Не найдено"):
        GitHubClient(session=_Session(_Resp(404))).get_issue(ref)
    with pytest.raises(GitHubError, match="лимит"):
        GitHubClient(session=_Session(_Resp(403, headers={"X-RateLimit-Remaining": "0"}))).get_issue(ref)
    with pytest.raises(GitHubError, match="500"):
        GitHubClient(session=_Session(_Resp(500, text="oops"))).get_issue(ref)


def test_download_python_files_and_size_limit():
    zip_bytes = build_zip({"a.py": "x = 1\n"})
    client = GitHubClient(session=_Session(_Resp(content=zip_bytes)))
    assert [f.path for f in client.download_python_files("o", "r", "sha")] == ["a.py"]
    assert client.session.calls[0][0].endswith("/repos/o/r/zipball/sha")

    tiny = GitHubClient(session=_Session(_Resp(content=zip_bytes)), max_repo_mb=0)
    with pytest.raises(GitHubError, match="слишком большой"):
        tiny.download_python_files("o", "r", "sha")


def test_requests_session_is_default():
    assert isinstance(GitHubClient().session, requests.Session)

import json

import pytest

from fakes import FakeGitHub, FakeLLM
from issue_locator import cli
from issue_locator.config import Config
from issue_locator.github_client import GitHubError
from issue_locator.models import Issue, IssueRef
from issue_locator.pipeline import Pipeline, PipelineError

URL = "https://github.com/owner/repo/issues/7"
ISSUE = Issue(
    ref=IssueRef("owner", "repo", 7),
    title="Login fails for uppercase usernames",
    body='`login()` raises KeyError("Admin") when the username has capital letters.',
    labels=["bug"],
)


@pytest.fixture
def config(tmp_path):
    return Config(chroma_dir=str(tmp_path / "chroma"), embedder="hash")


def test_pipeline_end_to_end_without_llm(config):
    github = FakeGitHub(ISSUE)
    report = Pipeline(config, github=github).run(URL, top_k=3)

    meta = report["metadata"]
    assert meta["issue"]["url"] == URL and meta["issue"]["labels"] == ["bug"]
    assert meta["repository"] == {"owner": "owner", "name": "repo", "url": "https://github.com/owner/repo", "commit": "abc1234def5678"}
    assert meta["classification"]["issue_type"] == "bug" and meta["classification"]["source"] == "heuristic"

    assert report["affected_files"][0]["path"] == "app/auth.py"
    assert len(report["affected_files"]) <= 3
    run = report["run"]
    assert run["index"] == "fresh" and run["chunks_in_index"] > 0 and run["embedder"] == "hash"
    assert run["llm_classification"] is False and run["llm_rerank"] is False
    json.dumps(report)  # отчёт сериализуется целиком


def test_second_run_uses_cache_and_reindex_forces_download(config):
    github = FakeGitHub(ISSUE)
    pipeline = Pipeline(config, github=github)
    pipeline.run(URL)
    assert pipeline.run(URL)["run"]["index"] == "cached" and github.downloads == 1
    assert pipeline.run(URL, reindex=True)["run"]["index"] == "fresh" and github.downloads == 2

    github.sha = "newcommit0000"  # в репозитории появился новый коммит
    assert pipeline.run(URL)["run"]["index"] == "fresh" and github.downloads == 3


def test_pipeline_with_llm(config):
    llm = FakeLLM(
        classification={
            "issue_type": "bug",
            "summary": "Username is not normalised",
            "keywords": ["login", "normalize_username"],
            "search_queries": ["normalize username before user lookup"],
        },
        rerank={"files": [{"path": "app/auth.py", "relevant": True, "reason": "login() looks up the raw username"}]},
    )
    report = Pipeline(config, github=FakeGitHub(ISSUE), llm=llm).run(URL)
    assert report["run"]["llm_classification"] and report["run"]["llm_rerank"]
    assert report["affected_files"][0]["reason"] == "login() looks up the raw username"
    assert len(llm.calls) == 2

    llm.calls.clear()
    report = Pipeline(config, github=FakeGitHub(ISSUE), llm=llm).run(URL, use_llm=False)  # --no-llm
    assert not llm.calls and report["run"]["llm_classification"] is False


def test_repo_without_python_files(config):
    with pytest.raises(PipelineError, match="Python"):
        Pipeline(config, github=FakeGitHub(ISSUE, files={"README.md": "hi"})).run(URL)


def test_invalid_url(config):
    with pytest.raises(ValueError):
        Pipeline(config, github=FakeGitHub(ISSUE)).run("https://github.com/owner/repo")


def test_cli_writes_report_and_reports_errors(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "Pipeline", lambda config: Pipeline(config, github=FakeGitHub(ISSUE)))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    out = tmp_path / "report.json"
    assert cli.main([URL, "--embedder", "hash", "--no-llm", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["affected_files"][0]["path"] == "app/auth.py"

    assert cli.main([URL, "--embedder", "hash"]) == 0  # печать в stdout
    assert json.loads(capsys.readouterr().out)["metadata"]["issue"]["number"] == 7

    class Boom:
        def run(self, *a, **k):
            raise GitHubError("Не найдено")

    monkeypatch.setattr(cli, "Pipeline", lambda config: Boom())
    assert cli.main([URL]) == 1
    assert "Ошибка: Не найдено" in capsys.readouterr().err

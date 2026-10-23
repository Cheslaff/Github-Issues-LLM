import pytest

from fakes import SAMPLE_REPO, FakeLLM
from issue_locator.agents import CodeRetrieverAgent, TicketClassificationAgent
from issue_locator.agents.classifier import extract_keywords, extract_mentioned_files, heuristic_classify
from issue_locator.chunker import chunk_file
from issue_locator.embeddings import HashEmbedder
from issue_locator.models import Classification, Issue, IssueMetadata, IssueRef, SourceFile
from issue_locator.vector_store import VectorStore

REF = IssueRef("o", "r", 1)

TRACEBACK = '''Calling `login()` with "Admin" crashes:

Traceback (most recent call last):
  File "/srv/app/app/auth.py", line 13, in login
    record = users[key]
KeyError: 'Admin'
'''


def make_issue(title="Login fails for uppercase usernames", body=TRACEBACK, labels=("bug",)):
    return Issue(ref=REF, title=title, body=body, labels=list(labels))


# ---- классификатор -------------------------------------------------------------------------

def test_heuristic_classify_bug_with_traceback():
    c = heuristic_classify(make_issue())
    assert c.issue_type == "bug" and c.source == "heuristic"
    assert "login" in c.keywords and "normalize_username" not in c.keywords
    assert "/srv/app/app/auth.py" in c.mentioned_files or "srv/app/app/auth.py" in c.mentioned_files
    assert c.search_queries[0] == "Login fails for uppercase usernames"


@pytest.mark.parametrize(
    "title,labels,expected",
    [
        ("Add support for CSV export", [], "feature"),
        ("Typo in README", [], "documentation"),
        ("Why is parsing so slow?", [], "performance"),
        ("How do I configure retries?", [], "question"),
        ("Something", ["enhancement"], "feature"),
        ("Hello world", [], "other"),
    ],
)
def test_heuristic_issue_types(title, labels, expected):
    assert heuristic_classify(make_issue(title, "", labels)).issue_type == expected


def test_extract_helpers():
    assert extract_mentioned_files('File "C:\\proj\\pkg\\mod.py", line 3 and see utils/helpers.py') == [
        "C:/proj/pkg/mod.py", "utils/helpers.py",
    ]
    kw = extract_keywords("Crash in `Session.send`", "see get_user_by_id and HTTPAdapter, docs at github.com")
    assert "Session.send" in kw and "get_user_by_id" in kw and "HTTPAdapter" in kw
    assert not any("github" in k for k in kw)
    assert extract_keywords("Crashes on startup", "") == ["crashes", "startup"]


def test_llm_classification_is_used_and_validated():
    llm = FakeLLM(
        classification={
            "issue_type": "BUG",
            "summary": "Login is case sensitive",
            "keywords": ["normalize_username", "login", " "],
            "search_queries": ["normalize username before lookup"],
        }
    )
    c = TicketClassificationAgent(llm).classify(make_issue())
    assert (c.issue_type, c.source) == ("bug", "llm")
    assert c.keywords == ["normalize_username", "login"]
    assert c.mentioned_files  # пути берутся регулярками

    llm.classification = {"issue_type": "weird", "keywords": "not a list"}
    c = TicketClassificationAgent(llm).classify(make_issue())
    assert c.issue_type == "other" and c.keywords  # откат на эвристические ключевые слова


def test_llm_failure_falls_back_to_heuristics():
    c = TicketClassificationAgent(FakeLLM(fail=True)).classify(make_issue())
    assert c.source == "heuristic" and c.issue_type == "bug"
    assert TicketClassificationAgent(None).classify(make_issue()).source == "heuristic"


# ---- ретривер ------------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    s = VectorStore(str(tmp_path / "chroma"), HashEmbedder())
    chunks = [c for p, t in SAMPLE_REPO.items() for c in chunk_file(SourceFile(p, t))]
    s.index("o", "r", "abc1234", chunks)
    return s


def make_meta(issue=None, classification=None):
    issue = issue or make_issue()
    return IssueMetadata(issue, "abc1234", classification or heuristic_classify(issue))


def test_build_queries_dedup_and_limit(store):
    meta = make_meta()
    all_queries = CodeRetrieverAgent(store).build_queries(meta)
    assert len({q.lower() for q in all_queries}) == len(all_queries)  # дубликатов нет (заголовок не повторяется)
    assert all_queries[0] == meta.classification.search_queries[0]
    assert len(CodeRetrieverAgent(store, max_queries=2).build_queries(meta)) == 2


def test_retriever_without_llm(store):
    result = CodeRetrieverAgent(store, top_k_files=3).run(make_meta())
    paths = [f["path"] for f in result["files"]]
    assert paths[0] == "app/auth.py" and len(paths) <= 3
    top = result["files"][0]
    assert top["score"] == 1.0 and top["llm_relevant"] is None and result["llm_used"] is False
    assert top["url"] == "https://github.com/o/r/blob/abc1234/app/auth.py"
    chunk = top["chunks"][0]
    assert chunk["url"].startswith(top["url"] + "#L") and chunk["snippet"]
    assert "_code" not in chunk
    assert "mentioned in the issue" in top["reason"]


def test_retriever_with_llm_rerank(store):
    llm = FakeLLM(
        rerank={
            "files": [
                {"path": "app/auth.py", "relevant": True, "reason": "login() does not normalize the username"},
                {"path": "app/utils.py", "relevant": False, "reason": "unrelated helper"},
            ]
        }
    )
    result = CodeRetrieverAgent(store, llm, top_k_files=4).run(make_meta())
    assert result["llm_used"] is True
    first = result["files"][0]
    assert first["path"] == "app/auth.py" and first["reason"] == "login() does not normalize the username"
    flags = {f["path"]: f["llm_relevant"] for f in result["files"]}
    assert flags.get("app/utils.py") in (False, None)
    irrelevant = [i for i, f in enumerate(result["files"]) if f["llm_relevant"] is False]
    relevant = [i for i, f in enumerate(result["files"]) if f["llm_relevant"] is not False]
    assert all(r < i for r in relevant for i in irrelevant)  # «нерелевантные» в конце


def test_retriever_survives_llm_failure(store):
    result = CodeRetrieverAgent(store, FakeLLM(fail=True), top_k_files=2).run(make_meta())
    assert result["llm_used"] is False and result["files"][0]["path"] == "app/auth.py"


def test_mentioned_file_boost(store):
    issue = make_issue(title="Totals look odd", body="Seen in app/export.py and invoice totals", labels=[])
    cls = Classification("bug", "x", keywords=["invoice", "total"], search_queries=["invoice total tax"], mentioned_files=[])
    plain = CodeRetrieverAgent(store).run(make_meta(issue, cls))["files"][0]["path"]
    assert plain == "app/payments.py"
    cls.mentioned_files = ["app/export.py"]
    boosted = [f["path"] for f in CodeRetrieverAgent(store).run(make_meta(issue, cls))["files"]]
    assert "app/export.py" in boosted


def test_empty_index_returns_no_files(tmp_path):
    empty = VectorStore(str(tmp_path / "c"), HashEmbedder())
    assert CodeRetrieverAgent(empty).run(make_meta())["files"] == []

import math

import pytest

from issue_locator.chunker import chunk_file
from issue_locator.embeddings import HashEmbedder, make_embedder, tokenize
from issue_locator.models import SourceFile
from issue_locator.vector_store import VectorStore, collection_name

from fakes import SAMPLE_REPO


def test_tokenize_splits_identifiers():
    assert tokenize("normalizeUsername HTTPError get_user_by_id") == [
        "normalize", "username", "http", "error", "get", "user", "by", "id",
    ]


def test_hash_embedder_is_normalized_and_deterministic():
    emb = HashEmbedder()
    a, b = emb.embed(["parse the config file", "parse the config file"])
    assert a == b
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)


def test_hash_embedder_similarity_order():
    emb = HashEmbedder()
    q, close, far = emb.embed(["login username password", "def login(username, password)", "write csv report rows"])
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))
    assert dot(q, close) > dot(q, far)


def test_make_embedder():
    assert make_embedder("hash").name == "hash"
    assert make_embedder("default").name == "default"  # без загрузки модели
    with pytest.raises(ValueError):
        make_embedder("nope")


def test_collection_name_valid():
    for owner, repo in [("psf", "requests"), ("a", "b"), ("x" * 39, "y" * 100), ("o", "we..ird.name")]:
        name = collection_name(owner, repo)
        assert 3 <= len(name) <= 63 and name[0].isalnum() and name[-1].isalnum() and ".." not in name
    assert collection_name("A", "B") == collection_name("a", "b")


@pytest.fixture
def store(tmp_path):
    return VectorStore(str(tmp_path / "chroma"), HashEmbedder())


def _chunks():
    return [c for path, text in SAMPLE_REPO.items() for c in chunk_file(SourceFile(path, text))]


def test_index_query_and_cache(store):
    assert not store.is_indexed("o", "r", "sha1")
    n = store.index("o", "r", "sha1", _chunks())
    assert n == store.count("o", "r") > 0
    assert store.is_indexed("o", "r", "sha1")
    assert not store.is_indexed("o", "r", "sha2")  # другой коммит -> нужна переиндексация

    [hits_login, hits_csv] = store.query_many("o", "r", ["login with username and password", "write csv report"], 3)
    assert hits_login[0].chunk.path == "app/auth.py"
    assert hits_csv[0].chunk.path == "app/export.py"
    assert hits_login[0].similarity >= hits_login[-1].similarity
    first = hits_login[0].chunk
    assert first.start_line >= 1 and first.end_line >= first.start_line and first.code


def test_reindex_replaces_collection(store):
    store.index("o", "r", "sha1", _chunks())
    store.index("o", "r", "sha2", _chunks()[:2])
    assert store.count("o", "r") == 2 and store.is_indexed("o", "r", "sha2")


def test_query_on_missing_collection(store):
    assert store.query_many("o", "missing", ["anything"], 3) == [[]]

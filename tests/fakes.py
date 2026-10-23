"""Подставные GitHub и LLM для тестов: ни сеть, ни ключи не нужны."""

import io
import zipfile

from issue_locator.models import Issue, SourceFile
from issue_locator.github_client import extract_python_files

SAMPLE_REPO = {
    "app/auth.py": '''"""Authentication helpers."""
import hashlib


def normalize_username(name):
    """Lower-case and strip the username before lookup."""
    return name.strip()


def login(users, username, password):
    """Check credentials. Raises KeyError for unknown users."""
    key = normalize_username(username)
    record = users[key]
    digest = hashlib.sha256(password.encode()).hexdigest()
    return record["password_hash"] == digest
''',
    "app/payments.py": '''def invoice_total(items, tax_rate):
    """Sum the price of all items and add tax."""
    subtotal = sum(i["price"] * i["qty"] for i in items)
    return round(subtotal * (1 + tax_rate), 2)


class Refund:
    def __init__(self, invoice_id, amount):
        self.invoice_id = invoice_id
        self.amount = amount

    def approve(self):
        """Mark the refund as approved."""
        return {"invoice": self.invoice_id, "amount": self.amount, "status": "approved"}
''',
    "app/export.py": '''import csv


def write_report_csv(rows, path):
    """Write report rows to a CSV file."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        for row in rows:
            writer.writerow(row)
''',
    "app/utils.py": '''def chunked(seq, size):
    """Yield successive chunks of a list."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
''',
    "tests/test_auth.py": "def test_nothing():\n    assert True\n",
}


def build_zip(files: dict, root: str = "owner-repo-abc1234") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, content in files.items():
            zf.writestr(f"{root}/{path}", content)
    return buf.getvalue()


class FakeGitHub:
    """Тот же интерфейс, что у GitHubClient (get_issue / get_head_sha / download_python_files)."""

    def __init__(self, issue: Issue, files: dict | None = None, sha: str = "abc1234def5678"):
        self.issue = issue
        self.files = files if files is not None else SAMPLE_REPO
        self.sha = sha
        self.downloads = 0

    def get_issue(self, ref):
        return self.issue

    def get_head_sha(self, owner, repo):
        return self.sha

    def download_python_files(self, owner, repo, sha):
        self.downloads += 1
        return extract_python_files(build_zip(self.files))


class FakeLLM:
    """Отвечает по заготовленным JSON: отдельно для классификации и для проверки кандидатов."""

    def __init__(self, classification: dict | None = None, rerank: dict | None = None, fail: bool = False):
        self.classification = classification or {}
        self.rerank = rerank or {"files": []}
        self.fail = fail
        self.calls = []

    def chat_json(self, system, user, temperature=0.0):
        from issue_locator.llm import LLMError

        self.calls.append(system)
        if self.fail:
            raise LLMError("boom")
        return self.classification if "analyse GitHub issues" in system else self.rerank

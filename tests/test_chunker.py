from issue_locator.chunker import chunk_file
from issue_locator.models import SourceFile


def _by_symbol(chunks):
    return {c.symbol: c for c in chunks}


def test_functions_classes_and_module():
    code = '''"""Docstring."""
import os

LIMIT = 10


def helper(x):
    return x + 1


@decorator
async def fetch():
    return 1


class Small:
    def a(self):
        return 1

    def b(self):
        return 2


if __name__ == "__main__":
    helper(1)
'''
    chunks = chunk_file(SourceFile("m.py", code))
    sym = _by_symbol(chunks)
    assert set(sym) == {"<module>", "helper", "fetch", "Small"}
    assert sym["fetch"].start_line == 11  # включает декоратор
    assert sym["fetch"].kind == "function"
    assert sym["Small"].kind == "class" and "def b" in sym["Small"].code
    assert sym["helper"].code.startswith("def helper")
    assert all(c.id == f"m.py:{c.start_line}-{c.end_line}" for c in chunks)


def test_import_only_module_group_is_skipped():
    chunks = chunk_file(SourceFile("m.py", "import os\nfrom sys import path\n\ndef f():\n    pass\n"))
    assert [c.symbol for c in chunks] == ["f"]


def test_large_class_split_into_methods():
    methods = "\n".join(f"    def m{i}(self):\n        return {i}\n" for i in range(6))
    code = f"class Big:\n    attr = 1\n\n{methods}"
    chunks = chunk_file(SourceFile("b.py", code), max_lines=10)
    symbols = [c.symbol for c in chunks]
    assert symbols[0] == "Big" and "attr = 1" in chunks[0].code
    assert symbols[1:] == [f"Big.m{i}" for i in range(6)]
    assert chunks[1].kind == "method"


def test_long_function_split_with_overlap():
    body = "\n".join(f"    x{i} = {i}" for i in range(100))
    chunks = chunk_file(SourceFile("l.py", f"def long():\n{body}\n"), max_lines=40, overlap=10)
    assert len(chunks) >= 3
    assert chunks[0].symbol == "long#part1"
    assert chunks[1].start_line == chunks[0].end_line - 10 + 1  # перекрытие
    assert chunks[-1].end_line == 101
    assert len({c.id for c in chunks}) == len(chunks)


def test_syntax_error_falls_back_to_windows():
    chunks = chunk_file(SourceFile("py2.py", "print 'hello'\nprint 'world'\n"))
    assert len(chunks) == 1 and chunks[0].kind == "window" and chunks[0].symbol == "<file>"


def test_empty_file():
    assert chunk_file(SourceFile("e.py", "  \n\n")) == []

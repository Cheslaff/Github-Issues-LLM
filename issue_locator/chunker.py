"""Разбиение Python-файла на смысловые фрагменты (функции, классы, методы) через ast."""

from __future__ import annotations

import ast

from .models import CodeChunk, SourceFile

_FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)


def _start(node: ast.AST) -> int:
    """Первая строка узла с учётом декораторов."""
    lines = [node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]
    return min(lines)


def _class_spans(node: ast.ClassDef, max_lines: int, prefix: str = "") -> list[tuple[str, str, int, int]]:
    name = prefix + node.name
    start, end = _start(node), node.end_lineno
    members = [n for n in node.body if isinstance(n, _FUNC + (ast.ClassDef,))]
    # Небольшой класс (или без методов) — один фрагмент целиком.
    if end - start + 1 <= max_lines or not members:
        return [(name, "class", start, end)]
    # Большой класс: «шапка» (до первого метода) + отдельные методы.
    spans: list[tuple[str, str, int, int]] = []
    first = _start(members[0])
    if first - 1 >= start:
        spans.append((name, "class", start, first - 1))
    for member in members:
        if isinstance(member, ast.ClassDef):
            spans.extend(_class_spans(member, max_lines, prefix=name + "."))
        else:
            spans.append((f"{name}.{member.name}", "method", _start(member), member.end_lineno))
    return spans


def _spans(tree: ast.Module, max_lines: int) -> list[tuple[str, str, int, int]]:
    spans: list[tuple[str, str, int, int]] = []
    group: list[ast.stmt] = []

    def flush() -> None:
        # Подряд идущий «прочий» код верхнего уровня (docstring, константы, if __name__ ...).
        # Блоки из одних import-ов для поиска бесполезны — пропускаем.
        if group and not all(isinstance(g, (ast.Import, ast.ImportFrom)) for g in group):
            spans.append(("<module>", "module", group[0].lineno, group[-1].end_lineno))
        group.clear()

    for node in tree.body:
        if isinstance(node, _FUNC):
            flush()
            spans.append((node.name, "function", _start(node), node.end_lineno))
        elif isinstance(node, ast.ClassDef):
            flush()
            spans.extend(_class_spans(node, max_lines))
        else:
            group.append(node)
    flush()
    return spans


def _split(
    path: str, lines: list[str], symbol: str, kind: str, start: int, end: int, max_lines: int, overlap: int
) -> list[CodeChunk]:
    """Режет слишком длинный фрагмент на окна с перекрытием."""
    if end - start + 1 <= max_lines:
        ranges = [(start, end)]
    else:
        ranges, cur, step = [], start, max_lines - overlap
        while cur <= end:
            stop = min(cur + max_lines - 1, end)
            ranges.append((cur, stop))
            if stop == end:
                break
            cur += step
    chunks = []
    for i, (a, b) in enumerate(ranges):
        code = "\n".join(lines[a - 1 : b])
        if not code.strip():
            continue
        name = symbol if len(ranges) == 1 else f"{symbol}#part{i + 1}"
        chunks.append(CodeChunk(f"{path}:{a}-{b}", path, name, kind, a, b, code))
    return chunks


def chunk_file(source: SourceFile, max_lines: int = 80, overlap: int = 10) -> list[CodeChunk]:
    """Python-файл -> список фрагментов с метаданными (путь, символ, диапазон строк)."""
    lines = source.content.splitlines()
    if not source.content.strip():
        return []
    try:
        tree = ast.parse(source.content)
    except (SyntaxError, ValueError, RecursionError):
        # Файл не парсится (python 2, шаблон и т.п.) — режем просто по строкам.
        spans = [("<file>", "window", 1, len(lines))]
    else:
        spans = _spans(tree, max_lines)
    chunks: list[CodeChunk] = []
    for symbol, kind, start, end in spans:
        chunks.extend(_split(source.path, lines, symbol, kind, start, end, max_lines, overlap))
    return chunks

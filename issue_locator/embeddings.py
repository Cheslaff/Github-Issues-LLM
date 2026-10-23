"""Эмбеддеры: превращают текст в вектор."""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from typing import Protocol

from .models import CodeChunk


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def embedding_text(chunk: CodeChunk) -> str:
    """Текст, который реально эмбеддится: путь + символ + код."""
    return f"{chunk.path} {chunk.symbol}\n{chunk.code}"


_STOP = {
    "self", "cls", "def", "return", "import", "from", "class", "none", "true", "false", "pass",
    "the", "and", "for", "not", "with", "this", "that", "are", "was", "has", "have", "else", "elif",
    "try", "except", "raise", "yield", "lambda", "async", "await", "isinstance", "len",
}


def tokenize(text: str) -> list[str]:
    """Разбивает код/текст на слова: snake_case и CamelCase раскладываются на части."""
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [t for t in re.findall(r"[a-z]{2,}|\d+", text.lower()) if t not in _STOP]


class HashEmbedder:
    """Лёгкий оффлайн-эмбеддер (hashing trick, без нейросети и скачивания моделей).

    Ищет по совпадению слов, а не по смыслу. Нужен для тестов и быстрой отладки.
    """

    name = "hash"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        tokens = tokenize(text)
        features = Counter(tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])])
        vec = [0.0] * self.dim
        for feature, count in features.items():
            h = zlib.crc32(feature.encode())
            sign = 1.0 if (h >> 31) & 1 else -1.0
            vec[h % self.dim] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class DefaultEmbedder:
    """all-MiniLM-L6-v2 (ONNX) из ChromaDB. Модель (~80 МБ) скачивается при первом использовании."""

    name = "default"

    def __init__(self) -> None:
        self._fn = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._fn is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._fn = DefaultEmbeddingFunction()
        return [[float(x) for x in vec] for vec in self._fn(texts)]


def make_embedder(name: str) -> Embedder:
    if name == "default":
        return DefaultEmbedder()
    if name == "hash":
        return HashEmbedder()
    raise ValueError(f"Неизвестный эмбеддер: {name!r} (доступны: default, hash)")

"""Минимальный клиент для любого OpenAI-совместимого chat-completions API.

Подходит для DeepSeek (по умолчанию), OpenRouter, Groq, локального Ollama и т.п. —
меняются только LLM_BASE_URL / LLM_MODEL / LLM_API_KEY.
"""

from __future__ import annotations

import json
import logging
import re
import time

import requests

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> dict:
    """Достаёт JSON-объект из ответа модели (учитывает ```json ... ``` и лишний текст вокруг)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise LLMError("Модель вернула не JSON-объект.")


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: int = 90,
        retries: int = 2,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()

    def chat_json(self, system: str, user: str, temperature: float = 0.0) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"
        last_error: Exception | str = "unknown"
        for attempt in range(self.retries + 1):
            try:
                response = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
            else:
                code = response.status_code
                if code == 400 and "response_format" in payload:
                    # Часть провайдеров не знает JSON-режим — повторяем без него.
                    payload.pop("response_format")
                    continue
                if code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {code}"
                elif code >= 400:
                    raise LLMError(f"Ошибка LLM API {code}: {response.text[:200]}")
                else:
                    try:
                        content = response.json()["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, ValueError) as exc:
                        raise LLMError(f"Неожиданный формат ответа LLM API: {exc}") from exc
                    return extract_json(content)
            if attempt < self.retries:
                time.sleep(2**attempt)
        raise LLMError(f"LLM API недоступен после повторов: {last_error}")

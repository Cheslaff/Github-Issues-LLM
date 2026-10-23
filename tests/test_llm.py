import pytest
import requests

from issue_locator.llm import LLMClient, LLMError, extract_json


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": [1, 2]} hope it helps') == {"a": [1, 2]}
    for bad in ["no json", "[1, 2]", "{broken"]:
        with pytest.raises(LLMError):
            extract_json(bad)


class _Resp:
    def __init__(self, status, content=None, text=""):
        self.status_code, self.text = status, text
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _Session:
    def __init__(self, *responses):
        self.responses, self.payloads = list(responses), []

    def post(self, url, headers=None, json=None, timeout=None):
        self.payloads.append((url, dict(json)))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("issue_locator.llm.time.sleep", lambda s: None)


def test_chat_json_success():
    session = _Session(_Resp(200, '{"ok": true}'))
    client = LLMClient("key", base_url="https://x.test/v1/", model="m", session=session)
    assert client.chat_json("sys", "user") == {"ok": True}
    url, payload = session.payloads[0]
    assert url == "https://x.test/v1/chat/completions"
    assert payload["model"] == "m" and payload["response_format"] == {"type": "json_object"}


def test_retries_on_429_and_network_error():
    session = _Session(requests.ConnectionError("down"), _Resp(429), _Resp(200, '{"ok": 1}'))
    assert LLMClient("k", session=session).chat_json("s", "u") == {"ok": 1}
    assert len(session.payloads) == 3


def test_gives_up_after_retries():
    session = _Session(_Resp(503), _Resp(503), _Resp(503))
    with pytest.raises(LLMError, match="после повторов"):
        LLMClient("k", session=session).chat_json("s", "u")


def test_drops_response_format_on_400():
    session = _Session(_Resp(400, text="unknown field response_format"), _Resp(200, '{"ok": 1}'))
    assert LLMClient("k", session=session).chat_json("s", "u") == {"ok": 1}
    assert "response_format" not in session.payloads[1][1]


def test_client_error_is_not_retried():
    session = _Session(_Resp(401, text="bad key"))
    with pytest.raises(LLMError, match="401"):
        LLMClient("k", session=session).chat_json("s", "u")
    assert len(session.payloads) == 1

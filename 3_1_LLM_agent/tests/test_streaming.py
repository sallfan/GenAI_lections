from unittest.mock import MagicMock, patch, call

from llm_agent.streaming_client import StreamingLLMClient

class FakeStreamResp:
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    def iter_lines(self):
        for l in self._lines:
            yield l

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


def _make_client_local():
    return StreamingLLMClient(local=True, ollama_model="qwen3.5:0.8b")


def make_api_request_non_stream():
    client = _make_client_local()
    payload = {"model": "qwen3.5:0.8b", "messages": [{"role": "user", "content": "привет"}]}
    fake_json = {"choices": [{"message": {"content": "привет!"}}]}

    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_json
    mock_resp.raise_for_status.return_value = None

    with patch("llm_agent.streaming_client.requests.post", return_value=mock_resp) as mock_post:
        result = client._make_api_request(payload, stream=False)
        assert result == fake_json
        # Проверяем что stream не прокидывался
        args, kwargs = mock_post.call_args
        assert kwargs.get("stream") is None or kwargs.get("stream") is not True
        assert kwargs["json"] == payload


def streaming_yields_tokens():
    client = _make_client_local()
    payload = {"model": "qwen3.5:0.8b", "messages": [{"role": "user", "content": "скажи привет"}]}

    lines = [
        b'data: {"choices": [{"delta": {"content": "Hello"}}]}',
        b'data: {"choices": [{"delta": {"content": " world"}}]}',
        b'data: {"choices": [{"delta": {"content": "!"}}]}',
        b'data: [DONE]',
    ]

    with patch("llm_agent.streaming_client.requests.post", return_value=FakeStreamResp(lines)):
        tokens = list(client._make_api_request(payload, stream=True))
        assert tokens == ["Hello", " world", "!"]
        assert "".join(tokens) == "Hello world!"


def streaming_calls_on_token_stepwise():
    client = _make_client_local()
    payload = {"model": "qwen3.5:0.8b", "messages": [{"role": "user", "content": "думай вслух"}]}

    lines = [
        b'data: {"choices": [{"delta": {"content": "We"}}]}',
        b'data: {"choices": [{"delta": {"content": " need"}}]}',
        b'data: {"choices": [{"delta": {"content": " tools"}}]}',
        b'data: [DONE]',
    ]
    on_token = MagicMock()

    with patch("llm_agent.streaming_client.requests.post", return_value=FakeStreamResp(lines)):
        tokens = list(client._make_api_request(payload, stream=True, on_token=on_token))

    assert tokens == ["We", " need", " tools"]
    assert on_token.call_count == 3
    on_token.assert_has_calls([call("We"), call(" need"), call(" tools")], any_order=False)

def test_all_streaming_unit():
    make_api_request_non_stream()
    streaming_yields_tokens()
    streaming_calls_on_token_stepwise()

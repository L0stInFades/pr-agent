import json
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

import pr_agent.algo.ai_handlers.litellm_ai_handler as litellm_handler
import pr_agent.algo.ai_handlers.litellm_helpers as litellm_helpers
from pr_agent.algo import MAX_TOKENS
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler


MINIMAX_BASE = "https://api.minimax.io/v1"


class _Section:
    def __init__(self, **values):
        self._values = values
        for key, value in values.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return self._values.get(key, default)


class _Settings:
    def __init__(
        self,
        *,
        openai_key=None,
        openai_api_base=None,
        minimax_key=None,
        minimax_api_base=None,
        litellm_extra_body=None,
        minimax_reasoning_split=True,
        minimax_thinking=None,
        minimax_max_completion_tokens=-1,
    ):
        self.config = _Section(
            reasoning_effort=None,
            ai_timeout=30,
            custom_reasoning_model=False,
            max_model_tokens=1000000,
            verbosity_level=0,
            seed=-1,
        )
        self.litellm = _Section(extra_body=litellm_extra_body)
        self.openai = _Section(key=openai_key, api_base=openai_api_base)
        self.minimax = _Section(key=minimax_key, api_base=minimax_api_base)
        self._values = {
            "OPENAI.KEY": openai_key,
            "OPENAI.API_BASE": openai_api_base,
            "MINIMAX.KEY": minimax_key,
            "MINIMAX.API_BASE": minimax_api_base,
            "MINIMAX.REASONING_SPLIT": minimax_reasoning_split,
            "MINIMAX.THINKING": minimax_thinking,
            "MINIMAX.MAX_COMPLETION_TOKENS": minimax_max_completion_tokens,
        }

    def get(self, key, default=None):
        return self._values.get(key, default)


def _mock_response():
    response = MagicMock()
    response.__getitem__ = lambda self, key: {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
    }[key]
    response.dict.return_value = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    return response


@pytest.fixture(autouse=True)
def clean_litellm_state(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "api_base", None)


class TestMiniMaxM3:
    @pytest.mark.asyncio
    async def test_openai_compatible_minimax_m3_normalizes_and_uses_openai_key(self, monkeypatch):
        settings = _Settings(openai_key="minimax-key", openai_api_base=MINIMAX_BASE)
        monkeypatch.setattr(litellm_handler, "get_settings", lambda: settings)

        with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _mock_response()
            handler = LiteLLMAIHandler()
            await handler.chat_completion(model="MiniMax-M3", system="sys", user="usr")

        kwargs = mock_call.call_args[1]
        assert kwargs["model"] == "minimax/MiniMax-M3"
        assert kwargs["api_base"] == MINIMAX_BASE
        assert kwargs["api_key"] == "minimax-key"
        assert kwargs["reasoning_split"] is True

    @pytest.mark.asyncio
    async def test_openai_prefixed_minimax_m3_normalizes_when_base_url_is_minimax(self, monkeypatch):
        settings = _Settings(openai_key="minimax-key", openai_api_base=MINIMAX_BASE)
        monkeypatch.setattr(litellm_handler, "get_settings", lambda: settings)

        with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _mock_response()
            handler = LiteLLMAIHandler()
            await handler.chat_completion(model="openai/MiniMax-M3", system="sys", user="usr")

        assert mock_call.call_args[1]["model"] == "minimax/MiniMax-M3"

    @pytest.mark.asyncio
    async def test_minimax_section_key_and_base_are_used(self, monkeypatch):
        settings = _Settings(minimax_key="minimax-key", minimax_api_base=MINIMAX_BASE)
        monkeypatch.setattr(litellm_handler, "get_settings", lambda: settings)

        with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _mock_response()
            handler = LiteLLMAIHandler()
            await handler.chat_completion(model="minimax/MiniMax-M3", system="sys", user="usr")

        kwargs = mock_call.call_args[1]
        assert kwargs["api_base"] == MINIMAX_BASE
        assert kwargs["api_key"] == "minimax-key"

    @pytest.mark.asyncio
    async def test_litellm_extra_body_can_override_minimax_defaults(self, monkeypatch):
        extra_body = json.dumps({
            "reasoning_split": False,
            "thinking": {"type": "disabled"},
            "max_completion_tokens": 123,
            "top_p": 0.9,
        })
        settings = _Settings(
            openai_key="minimax-key",
            openai_api_base=MINIMAX_BASE,
            litellm_extra_body=extra_body,
            minimax_thinking="adaptive",
            minimax_max_completion_tokens=456,
        )
        monkeypatch.setattr(litellm_handler, "get_settings", lambda: settings)
        monkeypatch.setattr(litellm_helpers, "get_settings", lambda: settings)

        with patch("pr_agent.algo.ai_handlers.litellm_ai_handler.acompletion", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = _mock_response()
            handler = LiteLLMAIHandler()
            await handler.chat_completion(model="MiniMax-M3", system="sys", user="usr")

        kwargs = mock_call.call_args[1]
        assert kwargs["reasoning_split"] is False
        assert kwargs["thinking"] == {"type": "disabled"}
        assert kwargs["max_completion_tokens"] == 123
        assert kwargs["top_p"] == 0.9

    def test_minimax_m3_token_aliases_are_registered(self):
        assert MAX_TOKENS["MiniMax-M3"] == 1000000
        assert MAX_TOKENS["minimax/MiniMax-M3"] == 1000000
        assert MAX_TOKENS["openai/MiniMax-M3"] == 1000000

from pydantic import ValidationError
import pytest

from benchmark.agent.errors import UnsupportedLlmProviderError
from benchmark.agent.llm import create_llm_client
from benchmark.agent.llm.anthropic_client import AnthropicClient
from benchmark.agent.llm.mock_client import MockLlmClient
from benchmark.agent.llm.openai_compatible_client import OpenAICompatibleClient
from benchmark.agent.llm.openrouter_client import OpenRouterClient
from benchmark.agent.models import LlmConfig, LlmProvider


def test_create_llm_client_returns_mock_without_api_key() -> None:
    client = create_llm_client(LlmConfig(provider=LlmProvider.MOCK, model="mock"))

    assert isinstance(client, MockLlmClient)


def test_create_llm_client_maps_supported_providers() -> None:
    assert isinstance(
        create_llm_client(LlmConfig(provider=LlmProvider.OPENROUTER, model="model")),
        OpenRouterClient,
    )
    assert isinstance(
        create_llm_client(LlmConfig(provider=LlmProvider.OPENAI_COMPATIBLE, model="model")),
        OpenAICompatibleClient,
    )
    assert isinstance(
        create_llm_client(LlmConfig(provider=LlmProvider.ANTHROPIC, model="model")),
        AnthropicClient,
    )


def test_ollama_is_not_supported() -> None:
    with pytest.raises(ValidationError):
        LlmConfig(provider="ollama", model="llama")

    config = LlmConfig(provider=LlmProvider.MOCK, model="mock").model_copy(
        update={"provider": "ollama"}
    )
    with pytest.raises(UnsupportedLlmProviderError, match="ollama"):
        create_llm_client(config)


def test_unknown_provider_raises_unsupported_provider_error() -> None:
    config = LlmConfig(provider=LlmProvider.MOCK, model="mock").model_copy(
        update={"provider": "unknown"}
    )

    with pytest.raises(UnsupportedLlmProviderError, match="unknown"):
        create_llm_client(config)

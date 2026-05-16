from __future__ import annotations

from benchmark.agent.errors import UnsupportedLlmProviderError
from benchmark.agent.llm.anthropic_client import AnthropicClient
from benchmark.agent.llm.base import LlmClient
from benchmark.agent.llm.mock_client import MockLlmClient
from benchmark.agent.llm.openai_compatible_client import OpenAICompatibleClient
from benchmark.agent.llm.openrouter_client import OpenRouterClient
from benchmark.agent.models import LlmConfig, LlmProvider


def create_llm_client(config: LlmConfig) -> LlmClient:
    provider = config.provider

    if provider == LlmProvider.MOCK:
        return MockLlmClient()
    if provider == LlmProvider.OPENROUTER:
        return OpenRouterClient(config)
    if provider == LlmProvider.OPENAI_COMPATIBLE:
        return OpenAICompatibleClient(config)
    if provider == LlmProvider.ANTHROPIC:
        return AnthropicClient(config)

    raise UnsupportedLlmProviderError(f"Unsupported LLM provider: {provider}")

"""LLM 客户端工厂。

根据配置中的 provider 名称，映射到对应的 API base URL，并创建 OpenAICompatibleClient 实例。
所有支持的 LLM 提供商都使用 OpenAI 兼容的 /chat/completions 接口。
"""

from __future__ import annotations

from ycli.config import LlmConfig
from ycli.llm.openai_compatible import OpenAICompatibleClient

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"

# 国内 LLM 提供商的 API 地址映射（provider 别名 → base URL）
PROVIDER_BASE_URLS = {
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "kimi": "https://api.moonshot.cn/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "step": "https://api.stepfun.com/v1",
}

# 已知模型的上下文窗口大小（token 数）
MODEL_CONTEXT_WINDOWS = {
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000,
    "deepseek-coder": 128_000,
}


def create_llm_client(config: LlmConfig) -> OpenAICompatibleClient:
    """根据 LlmConfig 创建对应的 LLM 客户端。

    匹配优先级：deepseek → openai/compatible → 已知提供商 → 默认 fallback。
    """
    provider = config.provider.lower()
    if provider == "deepseek":
        base_url = config.base_url or DEEPSEEK_BASE_URL
        context = MODEL_CONTEXT_WINDOWS.get(config.model, 64_000)
        return OpenAICompatibleClient(
            provider_name="deepseek",
            model=config.model,
            api_key=config.api_key,
            base_url=base_url,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
            max_context_window=context,
            prompt_cache=True,  # DeepSeek 支持 prompt cache
        )
    if provider in {"openai", "openai-compatible", "compatible"}:
        return OpenAICompatibleClient(
            provider_name=provider,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url or OPENAI_BASE_URL,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
            max_context_window=128_000,
            prompt_cache=False,
        )
    if provider in PROVIDER_BASE_URLS:
        return OpenAICompatibleClient(
            provider_name=provider,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url or PROVIDER_BASE_URLS[provider],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
            max_context_window=128_000,
            prompt_cache=False,
        )
    # Fallback: 未知提供商，使用用户指定的 base_url 或默认 DeepSeek
    return OpenAICompatibleClient(
        provider_name=provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url or DEEPSEEK_BASE_URL,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout,
        max_context_window=64_000,
        prompt_cache=False,
    )

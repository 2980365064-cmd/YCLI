"""LLM 客户端协议定义。

LlmClient 是一个 Protocol（结构化子类型），任何实现了 chat() 方法的对象都可以作为 LLM 客户端。
Agent 循环通过此协议与 LLM 交互，不依赖具体的 API 实现。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from ycli.types import Message


class LlmClient(Protocol):
    """LLM 客户端协议。

    Attributes:
        model_name: 当前使用的模型标识。
        provider_name: 提供商名称（如 deepseek, openai）。
        max_context_window: 最大上下文窗口大小（token 数）。
    """

    model_name: str
    provider_name: str
    max_context_window: int

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        system_prompt: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式对话接口。

        Yields:
            事件字典，类型包括：
            - text_delta: 文本增量
            - thinking_delta: 思考过程增量
            - tool_call: 工具调用（含 name + arguments 增量）
            - usage: Token 用量
            - error: 错误信息
        """
        ...

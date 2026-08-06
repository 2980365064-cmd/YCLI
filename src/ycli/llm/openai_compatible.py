"""OpenAI 兼容 API 的 LLM 客户端实现。

通过 Server-Sent Events (SSE) 流式接收 LLM 响应，支持：
- 标准 /chat/completions 接口（DeepSeek、OpenAI、GLM、Kimi 等）
- 工具调用（function calling）的增量解析
- 思考过程（reasoning_content）的流式输出
- 多模态输入（图片）的自动降级处理
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from ycli.types import Message


@dataclass(slots=True)
class OpenAICompatibleClient:
    """OpenAI 兼容 API 客户端。

    所有支持 /chat/completions 接口的 LLM 提供商都可以使用此客户端。
    通过 SSE 流式返回事件，与 Agent 循环的事件驱动架构对齐。
    """

    provider_name: str
    model: str
    api_key: str
    base_url: str
    max_tokens: int = 8192
    temperature: float = 0.7
    timeout: float = 120.0
    max_context_window: int = 128_000
    prompt_cache: bool = False

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def supports_images(self) -> bool:
        """判断当前模型是否支持图片输入（通过模型名称启发式判断）。"""
        model = self.model.lower()
        provider = self.provider_name.lower()
        return any(marker in model for marker in ("vision", "image", "5v", "vl")) or (
            provider in {"glm", "zhipu"} and "5v" in model
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        system_prompt: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式对话主方法。通过 SSE 接收响应，逐块解析并 yield 事件。"""
        if not self.api_key:
            yield {
                "type": "error",
                "error": RuntimeError(
                    "YCLI_API_KEY is not configured. Set it in env, ~/.ycli/config.json, "
                    "or project .ycli/config.json."
                ),
            }
            return

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages, system_prompt),
            "stream": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
            "user-agent": "YCLI-Python/0.1.0",
        }
        url = self.base_url.rstrip("/") + "/chat/completions"

        yield {"type": "message_start", "model": self.model}
        # 使用 httpx 流式请求，逐 SSE 事件解析
        async with (
            httpx.AsyncClient(timeout=self.timeout, http2=False) as client,
            client.stream("POST", url, headers=headers, json=payload) as response,
        ):
            response.raise_for_status()
            async for event in _iter_sse(response):
                if event == "[DONE]":
                    break
                try:
                    chunk = json.loads(event)
                except json.JSONDecodeError:
                    continue
                async for parsed in self._parse_chunk(chunk):
                    yield parsed

    def _format_messages(self, messages: list[Message], system_prompt: str) -> list[dict[str, Any]]:
        """将内部 Message 列表转换为 OpenAI API 格式的消息数组。"""
        formatted: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for message in messages:
            if message.role == "tool":
                # 工具返回消息需要 tool_call_id
                formatted.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id or "",
                        "content": str(message.content),
                    }
                )
            elif message.role == "assistant":
                item: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
                if message.tool_calls:
                    item["tool_calls"] = message.tool_calls
                formatted.append(item)
            else:
                formatted.append(
                    {"role": message.role, "content": self._format_content(message.content)}
                )
        return formatted

    def _format_content(self, content: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
        """格式化消息内容。如果模型不支持图片，自动将图片引用降级为文本描述。"""
        if isinstance(content, str):
            return content
        if self.supports_images:
            # 支持图片：保留 image_url 部分，移除 metadata
            cleaned = []
            for part in content:
                item = {key: value for key, value in part.items() if key != "metadata"}
                cleaned.append(item)
            return cleaned
        # 不支持图片：将图片转为文本描述
        text_parts = []
        for part in content:
            if part.get("type") == "text":
                text_parts.append(str(part.get("text") or ""))
            elif part.get("type") == "image_url":
                metadata = part.get("metadata") or {}
                source = metadata.get("source", "remote image")
                width = metadata.get("width", "?")
                height = metadata.get("height", "?")
                text_parts.append(f"[Image omitted: {source}, {width}x{height}]")
        return "\n".join(text_parts)

    async def _parse_chunk(self, chunk: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """解析单个 SSE chunk，提取文本/思考/工具调用/用量等增量数据。"""
        choices = chunk.get("choices") or []
        if not choices:
            return
        choice = choices[0]
        delta = choice.get("delta") or {}

        # 思考过程（如 DeepSeek-R1 的 reasoning_content）
        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            yield {"type": "thinking_delta", "thinking": reasoning}

        # 正文内容
        content = delta.get("content")
        if isinstance(content, str) and content:
            yield {"type": "text_delta", "text": content}

        # 工具调用增量（index + function name + arguments 片段）
        tool_calls = delta.get("tool_calls") or []
        for tool_call in tool_calls:
            yield {"type": "tool_call_delta", "tool_call": tool_call}

        # 结束原因 → 统一的 stop_reason
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            yield {
                "type": "message_end",
                "stop_reason": _map_finish_reason(str(finish_reason)),
            }

        # Token 用量（部分提供商在最后一个 chunk 返回）
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            yield {
                "type": "usage",
                "usage": {
                    "input_tokens": int(usage.get("prompt_tokens") or 0),
                    "output_tokens": int(usage.get("completion_tokens") or 0),
                },
            }


async def _iter_sse(response: httpx.Response) -> AsyncIterator[str]:
    """从 HTTP 流式响应中解析 SSE 事件。

    SSE 协议：每个事件以双换行分隔，以 "data:" 开头的行携带数据。
    """
    buffer = ""
    async for text in response.aiter_text():
        buffer += text
        while "\n\n" in buffer:
            event, buffer = buffer.split("\n\n", 1)
            data_lines = []
            for line in event.splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())
            if data_lines:
                yield "\n".join(data_lines)
    # 处理最后一个未以双换行结尾的事件
    if buffer.strip():
        data_lines = []
        for line in buffer.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if data_lines:
            yield "\n".join(data_lines)


def _map_finish_reason(reason: str) -> str:
    """将各提供商的 finish_reason 映射为统一的 stop_reason。"""
    if reason in {"tool_calls", "tool_use"}:
        return "tool_use"
    if reason == "length":
        return "max_tokens"
    if reason == "content_filter":
        return "stop_sequence"
    return "end_turn"

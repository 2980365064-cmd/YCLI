"""ReAct（Reasoning + Acting）核心循环。

本模块是整个 Agent 系统的心脏，实现了经典的 ReAct 循环：
  1. 将用户消息和历史发给 LLM，流式接收响应
  2. 如果 LLM 决定调用工具，收集并解析工具调用请求
  3. 通过 ToolExecutor 执行工具，将结果作为 tool 消息回传给 LLM
  4. 重复上述过程，直到 LLM 不再请求工具调用或达到最大轮次

所有 agent 模式（单 agent、编排器、规划执行）最终都委托给本模块的 query() 函数。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ycli.config import YcliConfig
from ycli.image import parse_image_references
from ycli.llm.base import LlmClient
from ycli.tools.base import ToolContext
from ycli.tools.executor import ToolExecutor
from ycli.tools.registry import ToolRegistry
from ycli.types import Message


async def query(
    *,
    llm_client: LlmClient,
    tool_registry: ToolRegistry,
    system_prompt: str,
    user_message: str,
    history: list[Message] | None,
    cwd: str,
    config: YcliConfig,
    approval_callback=None,
    skill_context_buffer=None,
    max_turns: int = 20,
) -> AsyncIterator[dict[str, Any]]:
    """ReAct 主循环：驱动 LLM 与工具的交替推理，以 async generator 方式 yield 事件。

    参数:
        llm_client: LLM 客户端，负责流式调用模型
        tool_registry: 工具注册表，提供工具定义和执行器
        system_prompt: 系统提示词
        user_message: 用户输入（支持图片引用，会被解析为 data URL）
        history: 历史消息列表（可选）
        cwd: 当前工作目录，传递给工具上下文
        config: 全局配置
        approval_callback: 人工审批回调（HITL），工具执行前可能需要用户确认
        skill_context_buffer: 技能上下文缓冲区，load_skill 工具会往里写入内容
        max_turns: 最大推理轮次，防止死循环

    Yields:
        dict 事件，类型包括：
        - text_delta: LLM 的文本增量输出
        - thinking_delta: LLM 的思考过程增量（部分模型支持）
        - tool_call: 工具调用请求（包含名称和参数）
        - tool_result: 工具执行结果
        - usage: token 用量统计
        - turn_complete: 一轮 LLM 响应结束
        - done: 整个循环结束，包含总轮次、总 token 数和完整消息列表
        - error: 错误信息
    """
    # 将技能上下文注入用户消息（如果有待消费的 skill 内容）
    user_message = _prepend_skill_context(user_message, skill_context_buffer)
    # 构建消息列表：历史消息 + 当前用户消息（图片引用会被解析为 data URL）
    messages = [
        *(history or []),
        Message(role="user", content=parse_image_references(user_message, cwd)),
    ]
    tool_definitions = tool_registry.definitions()
    executor = ToolExecutor(tool_registry)
    context = ToolContext(
        cwd=cwd,
        config=config,
        approval_callback=approval_callback,
        skill_context_buffer=skill_context_buffer,
    )

    total_tokens = 0
    turn = 0

    while turn < max_turns:
        turn += 1
        text = ""
        thinking = ""
        stop_reason = "end_turn"
        usage_input = 0
        usage_output = 0
        # 收集流式 tool_call 增量，按 index 分组合并
        tool_states: dict[int, dict[str, Any]] = {}

        # 流式调用 LLM，逐块处理响应事件
        async for event in llm_client.chat(messages, tool_definitions, system_prompt=system_prompt):
            event_type = event.get("type")
            if event_type == "text_delta":
                delta = str(event.get("text") or "")
                text += delta
                yield {"type": "text_delta", "text": delta}
            elif event_type == "thinking_delta":
                delta = str(event.get("thinking") or "")
                thinking += delta
                yield {"type": "thinking_delta", "thinking": delta}
            elif event_type == "tool_call_delta":
                # 流式工具调用：增量合并到 tool_states 中
                _merge_tool_delta(tool_states, event["tool_call"])
            elif event_type == "message_end":
                stop_reason = str(event.get("stop_reason") or "end_turn")
            elif event_type == "usage":
                usage = event.get("usage") or {}
                usage_input += int(usage.get("input_tokens") or 0)
                usage_output += int(usage.get("output_tokens") or 0)
                yield {"type": "usage", "usage": usage}
            elif event_type == "error":
                yield {"type": "error", "error": event["error"]}
                return

        total_tokens += usage_input + usage_output
        # 将流式增量合并为完整的工具调用列表
        tool_calls = _finalize_tool_calls(tool_states)
        assistant_message = Message(role="assistant", content=text, tool_calls=tool_calls)
        if thinking and text:
            assistant_message.content = text
        elif thinking:
            assistant_message.content = ""
        messages.append(assistant_message)
        yield {"type": "turn_complete", "turn": turn, "stop_reason": stop_reason}

        # 如果 LLM 没有请求工具调用且 stop_reason 不是 tool_use，则结束循环
        if stop_reason != "tool_use" and not tool_calls:
            break

        # yield 工具调用事件，通知上层渲染器
        for call in tool_calls:
            name = call.get("function", {}).get("name", "unknown")
            yield {"type": "tool_call", "name": name, "input": _tool_input(call)}

        # 执行所有工具（并发执行只读工具，顺序执行写入工具）
        tool_results = await executor.execute_all(tool_calls, context)
        # 将工具结果作为 tool 消息追加到对话历史，并 yield 给上层
        for result in tool_results:
            yield {
                "type": "tool_result",
                "name": _tool_name_by_id(tool_calls, result.tool_use_id or ""),
                "result": result.content,
                "is_error": result.is_error,
            }
            messages.append(
                Message(
                    role="tool",
                    content=result.content,
                    tool_call_id=result.tool_use_id,
                )
            )

    # 循环结束，yield 最终统计信息
    yield {
        "type": "done",
        "total_turns": turn,
        "total_tokens": total_tokens,
        "messages": messages,
    }


def _merge_tool_delta(tool_states: dict[int, dict[str, Any]], delta: dict[str, Any]) -> None:
    """将流式 tool_call_delta 合并到对应 index 的工具状态中。

    LLM 流式输出工具调用时，id/name/arguments 会分多个 delta 到达，
    此函数按 index 分组，逐步拼接出完整的工具调用。
    """
    index = int(delta.get("index") or 0)
    state = tool_states.setdefault(
        index,
        {
            "id": delta.get("id") or f"tool_{index}",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        },
    )
    if delta.get("id"):
        state["id"] = delta["id"]
    function = delta.get("function") or {}
    if function.get("name"):
        state["function"]["name"] = function["name"]
    # arguments 是 JSON 字符串，流式分块到达，需要拼接
    if function.get("arguments"):
        state["function"]["arguments"] += function["arguments"]


def _finalize_tool_calls(tool_states: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """将合并后的工具状态转为有序的工具调用列表（按 index 排序）。"""
    calls = []
    for index in sorted(tool_states):
        state = tool_states[index]
        if state["function"]["name"]:
            calls.append(state)
    return calls


def _tool_input(call: dict[str, Any]) -> dict[str, Any]:
    """解析工具调用的 arguments JSON 字符串为字典。"""
    raw = call.get("function", {}).get("arguments") or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _tool_name_by_id(calls: list[dict[str, Any]], tool_call_id: str) -> str:
    """根据 tool_call_id 查找对应的工具名称。"""
    for call in calls:
        if call.get("id") == tool_call_id:
            return str(call.get("function", {}).get("name") or "unknown")
    return "unknown"


def _prepend_skill_context(user_message: str, skill_context_buffer) -> str:
    """如果技能缓冲区有待消费内容，将其注入到用户消息前面。

    load_skill 工具会将 SKILL.md 的内容写入缓冲区，
    在下一次 LLM 调用时自动注入到用户消息中。
    """
    if not skill_context_buffer or skill_context_buffer.is_empty():
        return user_message
    drained = skill_context_buffer.drain()
    if not drained:
        return user_message
    return f"{drained}\n\n---\nUser request:\n{user_message}"

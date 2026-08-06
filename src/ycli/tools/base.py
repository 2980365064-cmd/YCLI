"""工具基础定义。

定义了 Agent 工具系统的核心数据结构：
- Tool: 工具定义（名称、参数 schema、handler、安全级别）
- ToolResult: 工具执行结果
- ToolContext: 执行上下文（cwd、config、审批回调）
- ToolDecision: HITL 审批决策（approve/deny/skip）

工具是 Agent 与外部世界交互的唯一途径，所有文件操作、命令执行、
网络请求等都通过工具完成。工具执行受 Policy 层约束。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ycli.config import YcliConfig

# 危险级别：safe=只读操作, medium=可逆修改, high=不可逆操作
DangerLevel = Literal["safe", "medium", "high"]
# HITL（Human-in-the-Loop）审批决策
ToolDecision = Literal["approve", "deny", "skip"]


@dataclass(slots=True)
class ToolResult:
    """工具执行结果。"""

    content: str
    is_error: bool = False
    display_summary: str | None = None
    tool_use_id: str | None = None


@dataclass(slots=True)
class ToolContext:
    """工具执行时的上下文信息。

    Attributes:
        cwd: 当前工作目录（路径守卫以此为基准检查合法性）。
        config: 全局配置。
        approval_callback: HITL 审批回调函数，返回 approve/deny/skip。
        skill_context_buffer: 技能上下文缓冲区（load_skill 工具使用）。
    """

    cwd: str
    config: YcliConfig
    approval_callback: Callable[[dict[str, Any]], Awaitable[ToolDecision] | ToolDecision] | None = (
        None
    )
    skill_context_buffer: Any | None = None


@dataclass(slots=True)
class Tool:
    """工具定义。

    Attributes:
        name: 工具名称（如 read_file, bash），也是 LLM 调用时的标识符。
        description: 工具描述，会发送给 LLM 作为 function definition。
        parameters: JSON Schema 格式的参数定义。
        handler: 异步执行函数 (payload, context) → ToolResult。
        is_read_only: 是否为只读操作（影响并发策略）。
        is_concurrency_safe: 是否可并发执行。
        danger_level: 危险级别（safe/medium/high）。
        requires_approval: 是否需要人工审批。
        timeout: 执行超时秒数。
        required_keys: 必填参数名列表。
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]
    is_read_only: bool = True
    is_concurrency_safe: bool = True
    danger_level: DangerLevel = "safe"
    requires_approval: bool = False
    timeout: float = 60.0
    required_keys: list[str] = field(default_factory=list)

    def definition(self) -> dict[str, Any]:
        """生成 LLM function calling 格式的工具定义。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """校验工具参数（类型 + 必填项）。"""
        if not isinstance(payload, dict):
            raise ValueError(f'tool "{self.name}" input must be an object')
        for key in self.required_keys:
            if key not in payload:
                raise ValueError(f'tool "{self.name}" missing required input: {key}')
        return payload

    async def execute(self, payload: dict[str, Any], context: ToolContext) -> ToolResult:
        """执行工具，带超时保护。"""
        data = self.validate(payload)
        return await asyncio.wait_for(self.handler(data, context), timeout=self.timeout)


def object_schema(
    properties: dict[str, dict[str, Any]],
    required: list[str] | None = None,
) -> dict[str, Any]:
    """便捷函数：生成 object 类型的 JSON Schema。"""
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
    }

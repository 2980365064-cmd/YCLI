"""YCLI 核心类型定义。

定义了 Agent 循环中使用的基础数据结构：
- Message: LLM 对话消息（支持多模态 content）
- Usage: Token 用量统计
- QueryResult: 查询完成结果
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# 消息角色：system=系统提示, user=用户输入, assistant=模型回复, tool=工具返回
Role = Literal["system", "user", "assistant", "tool"]
# 停止原因：end_turn=正常结束, tool_use=需要调用工具, max_tokens=达到上限
StopReason = Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]


@dataclass(slots=True)
class Message:
    """对话消息。content 可以是纯文本(str)或多模态内容(list[dict])。"""

    role: Role
    content: str | list[dict[str, Any]]
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class Usage:
    """Token 用量统计（输入 + 输出）。"""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(slots=True)
class QueryResult:
    """Agent 查询完成后的结果摘要。"""

    text: str
    total_tokens: int
    turns: int

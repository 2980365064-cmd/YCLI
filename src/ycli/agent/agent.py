"""单 Agent 模式封装。

本模块提供 Agent 类，是系统中最简单的 agent 执行模式：
一个 LLM + 一组工具，直接驱动 ReAct 循环。
Agent 内部调用 query.py 的 query() 函数完成实际的推理-行动循环，
并在外层提供：
  - 历史消息管理（多轮对话）
  - 执行前后自动快照（可用于回滚）
  - 技能上下文缓冲
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

from ycli.config import YcliConfig
from ycli.llm.base import LlmClient
from ycli.skill import SkillContextBuffer
from ycli.snapshot import SnapshotService
from ycli.tools.registry import ToolRegistry
from ycli.types import Message, QueryResult

from .query import query


class Agent:
    """单 Agent 执行器，封装 LLM + 工具 + ReAct 循环。

    这是最基础的 agent 模式：单个 LLM 实例配合工具集，
    通过 query() 驱动推理-行动循环。支持多轮对话（通过 history 保持上下文），
    每次 run() 前后自动创建项目快照以便回滚。

    典型用法：
        agent = Agent(llm_client=..., tool_registry=..., ...)
        async for event in agent.run("帮我写一个函数"):
            # 处理事件（text_delta / tool_call / tool_result / done 等）
            ...
    """

    def __init__(
        self,
        *,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        system_prompt: str,
        cwd: str,
        config: YcliConfig,
        approval_callback=None,
        max_turns: int = 20,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.cwd = cwd
        self.config = config
        self.approval_callback = approval_callback
        self.max_turns = max_turns
        # 历史消息，在每次 run() 结束后更新，用于多轮对话
        self.history: list[Message] = []
        self.skill_context_buffer = SkillContextBuffer()

    async def run(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """流式执行一轮 ReAct 循环，yield 事件给调用方。

        执行前后自动创建快照（best-effort，失败不影响主流程）。
        循环结束后自动更新 self.history 以支持多轮对话。
        """
        snapshot = SnapshotService(self.cwd)
        # 执行前快照，用于回滚
        with suppress(Exception):
            snapshot.create("pre-turn")
        try:
            async for event in query(
                llm_client=self.llm_client,
                tool_registry=self.tool_registry,
                system_prompt=self.system_prompt,
                user_message=message,
                history=self.history,
                cwd=self.cwd,
                config=self.config,
                approval_callback=self.approval_callback,
                skill_context_buffer=self.skill_context_buffer,
                max_turns=self.max_turns,
            ):
                # 循环结束时，保存完整消息历史供后续轮次使用
                if event.get("type") == "done":
                    self.history = list(event.get("messages") or [])
                yield event
        finally:
            # 执行后快照
            with suppress(Exception):
                snapshot.create("post-turn")

    async def run_complete(self, message: str) -> QueryResult:
        """同步执行完整的 ReAct 循环，收集所有文本后返回聚合结果。

        不 yield 中间事件，适合不需要流式输出的场景。
        如果遇到 error 事件，直接抛出异常。
        """
        text = ""
        tokens = 0
        turns = 0
        async for event in self.run(message):
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "error":
                raise event["error"]
            elif event.get("type") == "done":
                tokens = int(event.get("total_tokens") or 0)
                turns = int(event.get("total_turns") or 0)
        return QueryResult(text=text, total_tokens=tokens, turns=turns)

    def clear_history(self) -> None:
        """清空对话历史和技能缓冲区，开始新的对话会话。"""
        self.history = []
        self.skill_context_buffer.clear()

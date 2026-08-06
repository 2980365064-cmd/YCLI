"""Plan-and-Execute（先规划后执行）模式。

本模块实现 DAG 驱动的任务执行流程：

  1. Planner 阶段：
     接收用户任务，通过 LLM 生成一个 ExecutionPlan（包含多个 Task 及其依赖关系，形成 DAG）。

  2. Executor 阶段：
     按 DAG 拓扑顺序执行 Task：
     - 无依赖的 Task 通过 asyncio.gather 并行执行
     - 有依赖的 Task 等待前置 Task 全部完成后再执行
     - 每个 Task 独立调用 ReAct 循环（query()），有自己的工具使用上下文
     - 已完成 Task 的结果会注入到下游 Task 的上下文中

  3. 汇总阶段：
     所有 Task 执行完毕后，汇总生成最终输出。

与 orchestrator.py 的区别：
  - orchestrator 有显式的 Reviewer 审查环节，Worker 可重试
  - plan_execute 更轻量：没有 Reviewer，每个 Task 只执行一次
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from ycli.agent.query import query
from ycli.config import YcliConfig
from ycli.llm.base import LlmClient
from ycli.plan import ExecutionPlan, Planner, Task, TaskStatus
from ycli.prompt import PromptAssembler
from ycli.skill import SkillContextBuffer
from ycli.snapshot import SnapshotService
from ycli.tools.registry import ToolRegistry
from ycli.types import Message


@dataclass(slots=True)
class TaskRunResult:
    """单个 Task 的执行结果，包含文本输出、token 用量和可选的异常信息。"""

    task: Task
    text: str
    tokens: int
    turns: int
    error: Exception | None = None


class PlanExecuteAgent:
    """先规划后执行的 Agent 模式。

    先由 Planner 生成 DAG 执行计划，再按拓扑顺序逐步执行。
    每个 Task 作为独立的 ReAct 循环运行，完成后结果注入下游 Task 上下文。
    支持并行执行无依赖的 Task 批次。

    参数:
        max_task_turns: 每个 Task 的 ReAct 最大轮次（默认 8，防止单任务失控）
    """

    def __init__(
        self,
        *,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        config: YcliConfig,
        cwd: str,
        approval_callback=None,
        planner: Planner | None = None,
        max_task_turns: int = 8,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.planner = planner or Planner(llm_client)
        self.max_task_turns = max_task_turns
        self.history: list[Message] = []
        self.skill_context_buffer = SkillContextBuffer()

    async def run(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """主入口：先规划，再执行，yield 事件流给调用方。"""
        snapshot = SnapshotService(self.cwd)
        with suppress(Exception):
            snapshot.create("pre-turn")
        total_tokens = 0
        total_turns = 0
        final_text = ""
        try:
            # 规划阶段：让 Planner 生成执行计划
            yield {"type": "text_delta", "text": f"Planning task: {message}\n\n"}
            plan = await self.planner.create_plan(message)
            yield {"type": "text_delta", "text": plan.summarize() + "\n\n"}

            # 执行阶段：按 DAG 拓扑顺序执行各 Task
            async for event in self._execute_plan(plan):
                if event.get("type") == "usage":
                    usage = event.get("usage") or {}
                    total_tokens += int(usage.get("input_tokens") or 0)
                    total_tokens += int(usage.get("output_tokens") or 0)
                elif event.get("type") == "plan_task_done":
                    total_turns += int(event.get("turns") or 0)
                    continue
                elif event.get("type") == "text_delta":
                    final_text += str(event.get("text") or "")
                yield event
            self.history = [
                Message(role="user", content=message),
                Message(role="assistant", content=final_text),
            ]
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "error": exc}
            return
        finally:
            with suppress(Exception):
                snapshot.create("post-turn")
        yield {
            "type": "done",
            "total_turns": total_turns,
            "total_tokens": total_tokens,
            "messages": self.history,
        }

    async def _execute_plan(self, plan: ExecutionPlan) -> AsyncIterator[dict[str, Any]]:
        """按 DAG 拓扑顺序执行计划中的所有 Task。

        每轮获取可执行的 Task（依赖已全部完成）：
        - 单个 Task：串行执行
        - 多个 Task：asyncio.gather 并行执行
        循环直到所有可执行 Task 处理完毕。
        """
        yield {"type": "text_delta", "text": "Executing plan...\n\n"}
        plan.mark_started()
        while True:
            executable = _executable_tasks_in_order(plan)
            if not executable:
                break
            if len(executable) == 1:
                result = await self._execute_task(plan, executable[0])
                async for event in self._apply_task_result(result):
                    yield event
                continue
            # 多个无依赖 Task，并行执行
            yield {
                "type": "text_delta",
                "text": (
                    f"Running parallel batch: {', '.join(task.id for task in executable)}\n\n"
                ),
            }
            results = await asyncio.gather(
                *(self._execute_task(plan, task) for task in executable),
                return_exceptions=False,
            )
            for result in results:
                async for event in self._apply_task_result(result):
                    yield event

        # 判断计划最终状态
        if plan.has_failed():
            plan.mark_failed()
            yield {"type": "text_delta", "text": "Plan partially completed with failed tasks.\n\n"}
        elif plan.is_all_completed():
            plan.mark_completed()
            yield {"type": "text_delta", "text": _build_plan_result(plan)}
        else:
            # 部分 Task 因依赖未满足而无法执行
            plan.mark_failed()
            yield {
                "type": "text_delta",
                "text": "Plan stalled because dependencies were not satisfied.\n\n",
            }

    async def _apply_task_result(self, result: TaskRunResult) -> AsyncIterator[dict[str, Any]]:
        """将 Task 执行结果应用到计划中，并 yield 相应事件。"""
        if result.error:
            result.task.mark_failed(str(result.error))
            yield {"type": "text_delta", "text": f"Failed [{result.task.id}]: {result.error}\n\n"}
            return
        result.task.mark_completed(result.text)
        yield {
            "type": "text_delta",
            "text": f"Completed [{result.task.id}]: {_preview(result.text)}\n\n",
        }
        yield {
            "type": "usage",
            "usage": {"input_tokens": result.tokens, "output_tokens": 0},
        }
        yield {"type": "plan_task_done", "turns": result.turns, "tokens": result.tokens}

    async def _execute_task(self, plan: ExecutionPlan, task: Task) -> TaskRunResult:
        """执行单个 Task：构建上下文后委托给 ReAct 循环（query()）。

        每个 Task 使用空 history，避免跨 Task 上下文污染。
        上下文包含任务描述和已完成依赖 Task 的结果摘要。
        """
        task.mark_started()
        text = ""
        tool_results: list[str] = []
        tokens = 0
        turns = 0
        try:
            async for event in query(
                llm_client=self.llm_client,
                tool_registry=self.tool_registry,
                system_prompt=self._task_system_prompt(task),
                user_message=_task_context(plan, task),
                history=[],  # 每个 Task 独立对话上下文
                cwd=self.cwd,
                config=self.config,
                approval_callback=self.approval_callback,
                skill_context_buffer=self.skill_context_buffer,
                max_turns=self.max_task_turns,
            ):
                if event.get("type") == "text_delta":
                    text += str(event.get("text") or "")
                elif event.get("type") == "tool_result":
                    content = str(event.get("result") or "")
                    if content:
                        tool_results.append(content)
                elif event.get("type") == "usage":
                    usage = event.get("usage") or {}
                    tokens += int(usage.get("input_tokens") or 0)
                    tokens += int(usage.get("output_tokens") or 0)
                elif event.get("type") == "done":
                    turns += int(event.get("total_turns") or 0)
                elif event.get("type") == "error":
                    raise event["error"]
            # 优先使用 LLM 文本输出，否则拼接工具结果
            result_text = text.strip() or "\n".join(tool_results).strip()
            return TaskRunResult(task=task, text=result_text, tokens=tokens, turns=turns)
        except Exception as exc:  # noqa: BLE001
            return TaskRunResult(task=task, text="", tokens=tokens, turns=turns, error=exc)

    def _task_system_prompt(self, task: Task) -> str:
        """为单个 Task 生成系统提示词：基础提示 + Task 执行指令。"""
        base = PromptAssembler(
            config=self.config,
            cwd=self.cwd,
            tool_names=self.tool_registry.list_names(),
            model=self.llm_client.model_name,
            provider=self.llm_client.provider_name,
        ).build()
        return (
            base
            + "\n\nYou are executing one task inside a Plan-and-Execute DAG.\n"
            + f"Task id: {task.id}\nTask type: {task.type.value}\n"
            + "Complete this task concretely. Use tools when needed."
        )


def _executable_tasks_in_order(plan: ExecutionPlan) -> list[Task]:
    """获取当前可执行的 Task 列表，按 plan 中的 execution_order 排序。"""
    executable_ids = {task.id for task in plan.executable_tasks()}
    return [plan.tasks[task_id] for task_id in plan.execution_order() if task_id in executable_ids]


def _task_context(plan: ExecutionPlan, task: Task) -> str:
    """构建 Task 执行上下文：目标、当前任务描述、已完成依赖 Task 的结果摘要。"""
    lines = [
        f"Goal: {plan.goal}",
        f"Current task [{task.id}]: {task.description}",
        "",
        "Completed dependency results:",
    ]
    for dep_id in task.dependencies:
        dep = plan.get_task(dep_id)
        if dep and dep.status == TaskStatus.COMPLETED:
            # 依赖结果截断到 800 字，避免上下文过长
            lines.append(f"- [{dep.id}] {dep.description}: {_preview(dep.result, 800)}")
    return "\n".join(lines)


def _build_plan_result(plan: ExecutionPlan) -> str:
    """构建计划完成后的汇总文本，列出所有 Task 的状态和结果。"""
    lines = ["Plan execution completed.", "", "Task summary:"]
    for task in plan.all_tasks():
        lines.append(f"- [{task.id}] {task.status.value}: {task.description}")
        if task.result:
            lines.append(f"  Result: {_preview(task.result)}")
    return "\n".join(lines) + "\n"


def _preview(text: str, max_len: int = 160) -> str:
    """截断文本到指定长度，超长部分用 ... 替代。"""
    value = (text or "").replace("\r\n", "\n").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."

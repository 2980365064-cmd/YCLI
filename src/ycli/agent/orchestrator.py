"""多 Agent 编排器（Orchestrator）模式。

本模块实现 Planner → Workers → Reviewer 三阶段协作流程：

  Phase 1 — Planner：
    接收用户任务，生成结构化的执行计划（JSON 格式的 step 列表）。

  Phase 2 — Workers + Reviewer：
    Worker 逐步执行计划中的每个 step（支持依赖关系，可并行执行无依赖的 step）。
    每个 step 执行完成后，由独立的 Reviewer 审查结果质量。
    如果 Reviewer 不通过，Worker 会带着审查反馈重试（最多 max_retries_per_step 次）。

  Phase 3 — 汇总：
    所有 step 完成后，汇总各 step 的结果生成最终输出。

关键类:
  - AgentRole: Agent 角色枚举（PLANNER / WORKER / REVIEWER）
  - AgentMessage: Agent 间通信的消息格式
  - ExecutionStep: 执行计划中的单个步骤
  - SubAgent: 子 Agent，承载具体角色的 LLM 调用
  - AgentOrchestrator: 编排器主体，协调各子 Agent
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from ycli.agent.query import query
from ycli.config import YcliConfig
from ycli.llm.base import LlmClient
from ycli.prompt import PromptAssembler
from ycli.skill import SkillContextBuffer
from ycli.snapshot import SnapshotService
from ycli.tools.registry import ToolRegistry
from ycli.types import Message


class AgentRole(StrEnum):
    """Agent 角色枚举，决定子 Agent 的系统提示词和行为模式。"""

    PLANNER = "PLANNER"
    WORKER = "WORKER"
    REVIEWER = "REVIEWER"


class AgentMessageType(StrEnum):
    """Agent 间通信的消息类型。"""

    TASK = "TASK"  # 任务下发（编排器 → Worker）
    RESULT = "RESULT"  # 执行结果（Worker → 编排器）
    FEEDBACK = "FEEDBACK"  # 审查反馈
    APPROVAL = "APPROVAL"  # 审查通过
    REJECTION = "REJECTION"  # 审查驳回
    ERROR = "ERROR"  # 执行出错


class StepStatus(StrEnum):
    """执行步骤的状态流转：PENDING → RUNNING → COMPLETED / FAILED。"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(slots=True)
class AgentMessage:
    """Agent 间的消息载体，包含发送者、角色、内容和消息类型。"""

    from_agent: str
    from_role: AgentRole | None
    content: str
    type: AgentMessageType

    @classmethod
    def task(cls, from_agent: str, content: str) -> AgentMessage:
        """创建任务下发消息。"""
        return cls(from_agent, None, content, AgentMessageType.TASK)

    @classmethod
    def result(cls, from_agent: str, role: AgentRole, content: str) -> AgentMessage:
        """创建执行结果消息。"""
        return cls(from_agent, role, content, AgentMessageType.RESULT)

    @classmethod
    def error(cls, from_agent: str, role: AgentRole, content: str) -> AgentMessage:
        """创建错误消息。"""
        return cls(from_agent, role, content, AgentMessageType.ERROR)


@dataclass(slots=True)
class ExecutionStep:
    """执行计划中的单个步骤，包含描述、类型、依赖关系和执行结果。

    步骤之间通过 dependencies 形成 DAG，编排器据此决定执行顺序。
    """

    id: str
    description: str
    type: str
    dependencies: list[str]
    result: str = ""
    status: StepStatus = StepStatus.PENDING

    def with_result(self, result: str) -> ExecutionStep:
        """返回带执行结果的新实例（标记为 COMPLETED）。"""
        return replace(self, result=result, status=StepStatus.COMPLETED)

    def with_failed(self, result: str) -> ExecutionStep:
        """返回带失败结果的新实例（标记为 FAILED）。"""
        return replace(self, result=result, status=StepStatus.FAILED)

    def started(self) -> ExecutionStep:
        """返回已启动状态的新实例。"""
        return replace(self, status=StepStatus.RUNNING)


class SubAgent:
    """子 Agent，承载特定角色的 LLM 调用。

    Worker 角色会走完整的 ReAct 循环（可以使用工具）；
    Planner 和 Reviewer 角色仅做纯 LLM 对话（不使用工具），
    返回结构化的 JSON 输出。
    """

    def __init__(
        self,
        *,
        name: str,
        role: AgentRole,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        config: YcliConfig,
        cwd: str,
        approval_callback=None,
        skill_context_buffer: SkillContextBuffer | None = None,
    ):
        self.name = name
        self.role = role
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.skill_context_buffer = skill_context_buffer or SkillContextBuffer()
        self.history: list[Message] = []

    async def execute(self, task: AgentMessage, context: str = "") -> AgentMessage:
        """执行任务：Worker 走 ReAct 循环，其他角色走纯 LLM 对话。"""
        content = f"{context}\n\nCurrent task:\n{task.content}".strip() if context else task.content
        if self.role == AgentRole.WORKER:
            return await self._execute_worker(content)
        return await self._execute_without_tools(content)

    async def review(self, original_task: str, execution_result: str) -> AgentMessage:
        """审查执行结果：将原始任务和执行结果拼在一起，让 Reviewer 评判。"""
        return await self.execute(
            AgentMessage.task(
                "orchestrator",
                f"Original task:\n{original_task}\n\nExecution result:\n{execution_result}",
            )
        )

    def clear_history(self) -> None:
        """清空对话历史，避免跨任务的上下文污染。"""
        self.history = []

    async def _execute_worker(self, content: str) -> AgentMessage:
        """Worker 执行路径：通过 ReAct 循环调用 LLM + 工具完成任务。"""
        text = ""
        tool_results: list[str] = []
        try:
            async for event in query(
                llm_client=self.llm_client,
                tool_registry=self.tool_registry,
                system_prompt=self._system_prompt(),
                user_message=content,
                history=self.history,
                cwd=self.cwd,
                config=self.config,
                approval_callback=self.approval_callback,
                skill_context_buffer=self.skill_context_buffer,
                max_turns=8,  # Worker 单次任务限制轮次，防止失控
            ):
                if event.get("type") == "text_delta":
                    text += str(event.get("text") or "")
                elif event.get("type") == "tool_result":
                    tool_results.append(str(event.get("result") or ""))
                elif event.get("type") == "done":
                    self.history = list(event.get("messages") or [])
                elif event.get("type") == "error":
                    raise event["error"]
        except Exception as exc:  # noqa: BLE001
            return AgentMessage.error(self.name, self.role, str(exc))
        # 优先使用 LLM 的文本输出，否则拼接工具结果
        result = text.strip() or "\n".join(item for item in tool_results if item).strip()
        return AgentMessage.result(self.name, self.role, result)

    async def _execute_without_tools(self, content: str) -> AgentMessage:
        """Planner/Reviewer 执行路径：仅调用 LLM 对话，不使用工具。"""
        text = ""
        messages = [*self.history, Message(role="user", content=content)]
        try:
            async for event in self.llm_client.chat(
                messages,
                [],  # 不提供工具定义
                system_prompt=self._system_prompt(),
            ):
                if event.get("type") == "text_delta":
                    text += str(event.get("text") or "")
                elif event.get("type") == "error":
                    raise event["error"]
        except Exception as exc:  # noqa: BLE001
            return AgentMessage.error(self.name, self.role, str(exc))
        self.history = [*messages, Message(role="assistant", content=text)]
        return AgentMessage.result(self.name, self.role, text)

    def _system_prompt(self) -> str:
        """根据角色生成系统提示词：基础提示 + 角色特定指令。"""
        base = PromptAssembler(
            config=self.config,
            cwd=self.cwd,
            tool_names=self.tool_registry.list_names(),
            model=self.llm_client.model_name,
            provider=self.llm_client.provider_name,
        ).build()
        role_prompt = {
            AgentRole.PLANNER: (
                "You are the Planner in a multi-agent workflow. Return only JSON with a "
                "steps array. Each step needs id, description, type, and dependencies."
            ),
            AgentRole.WORKER: (
                "You are the Worker in a multi-agent workflow. Execute only the assigned "
                "step. Use tools when needed and return the concrete result."
            ),
            AgentRole.REVIEWER: (
                "You are the Reviewer in a multi-agent workflow. Return JSON only: "
                '{"approved": true|false, "summary": "...", "issues": []}.'
            ),
        }[self.role]
        return f"{base}\n\n{role_prompt}\nAgent name: {self.name}"


class AgentOrchestrator:
    """多 Agent 编排器，协调 Planner → Workers → Reviewer 完成复杂任务。

    执行流程：
      1. Planner 分析用户任务，输出 JSON 格式的执行计划（步骤 + 依赖关系）
      2. 按 DAG 拓扑顺序执行步骤：
         - 无依赖的步骤并行执行（通过 worker 池调度）
         - 每个步骤执行完后由独立 Reviewer 审查
         - 审查不通过则 Worker 重试（最多 max_retries_per_step 次）
      3. 汇总所有步骤结果，生成最终输出

    Worker 池通过 asyncio.Queue 管理，任务完成后 Worker 归还池中复用。
    """

    max_retries_per_step = 2  # 每个步骤最多重试次数

    def __init__(
        self,
        *,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        config: YcliConfig,
        cwd: str,
        approval_callback=None,
        worker_count: int = 2,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.skill_context_buffer = SkillContextBuffer()
        self.planner = self._subagent("planner", AgentRole.PLANNER)
        # 创建 Worker 池，多个 Worker 可并行执行不同步骤
        self.workers = [
            self._subagent(f"worker-{index}", AgentRole.WORKER)
            for index in range(1, max(1, worker_count) + 1)
        ]
        self.reviewer = self._subagent("reviewer", AgentRole.REVIEWER)
        self.history: list[Message] = []

    async def run(self, message: str) -> AsyncIterator[dict[str, Any]]:
        """编排器主入口：依次执行规划阶段和执行阶段，yield 事件流。"""
        snapshot = SnapshotService(self.cwd)
        with suppress(Exception):
            snapshot.create("pre-turn")
        final_text = ""
        try:
            # Phase 1: 让 Planner 生成分步执行计划
            yield {"type": "text_delta", "text": "Phase 1: planner\n\n"}
            plan_result = await self.planner.execute(
                AgentMessage.task("orchestrator", f"Create an execution plan for:\n{message}")
            )
            self.planner.clear_history()
            if plan_result.type == AgentMessageType.ERROR:
                raise RuntimeError(f"planner failed: {plan_result.content}")
            steps = self.parse_plan(plan_result.content)
            if not steps:
                raise ValueError(f"planner output could not be parsed:\n{plan_result.content}")
            yield {"type": "text_delta", "text": self.summarize_steps(steps) + "\n"}

            # Phase 2: Workers 执行各步骤，Reviewer 逐步审查
            yield {"type": "text_delta", "text": "Phase 2: workers and reviewer\n\n"}
            for event in await self._execute_steps(
                steps, lambda text: {"type": "text_delta", "text": text}
            ):
                yield event
            # Phase 3: 汇总所有步骤结果
            final_text = self.build_final_result(steps)
            yield {"type": "text_delta", "text": final_text}
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
        yield {"type": "done", "total_turns": 0, "total_tokens": 0, "messages": self.history}

        async def _execute_steps(
            self,
            steps: list[ExecutionStep],
            event_factory,
        ) -> list[dict[str, Any]]:
            """按依赖关系逐步执行所有步骤，无依赖的步骤并行处理。"""
            events: list[dict[str, Any]] = []
            retry_count: dict[str, int] = {}
            # 通过 Queue 管理 Worker 池，空闲 Worker 入队等待分配
            worker_queue: asyncio.Queue[SubAgent] = asyncio.Queue()
            for worker in self.workers:
                worker_queue.put_nowait(worker)

            while True:
                # 获取当前可执行的步骤（依赖已全部完成）
                executable = self.get_executable_steps(steps)
                if not executable:
                    break
                if len(executable) > 1:
                    events.append(
                        event_factory(
                            f"Parallel batch: {', '.join(step.id for step in executable)}\n\n"
                        )
                    )
                # 并行执行当前批次的所有步骤
                await asyncio.gather(
                    *(
                        self._run_step_with_worker_queue(
                            step,
                            steps,
                            retry_count,
                            worker_queue,
                        )
                        for step in executable
                    )
                )
            return events

    async def _run_step_with_worker_queue(
        self,
        step: ExecutionStep,
        steps: list[ExecutionStep],
        retry_count: dict[str, int],
        worker_queue: asyncio.Queue[SubAgent],
    ) -> None:
        """从 Worker 池取出一个 Worker 执行步骤，执行完毕后归还。

        每个步骤使用独立的 Reviewer 实例，避免审查历史交叉污染。
        """
        worker = await worker_queue.get()
        try:
            reviewer = self._subagent(f"reviewer-{step.id}", AgentRole.REVIEWER)
            await self._run_step(step, steps, retry_count, worker, reviewer)
        finally:
            # 清空 Worker 历史后归还池中，防止跨步骤上下文泄漏
            worker.clear_history()
            worker_queue.put_nowait(worker)

    async def _run_step(
        self,
        step: ExecutionStep,
        steps: list[ExecutionStep],
        retry_count: dict[str, int],
        worker: SubAgent,
        reviewer: SubAgent,
    ) -> None:
        """执行单个步骤：Worker 执行 → Reviewer 审查 → 不通过则重试。

        重试时会把 Reviewer 的反馈（issues）注入上下文，
        让 Worker 知道上次被驳回的原因。
        """
        self._update_step(steps, step.id, step.started())
        # 构建上下文：包含已完成依赖步骤的结果摘要
        context = self.build_step_context(steps, step)
        task_msg = AgentMessage.task("orchestrator", step.description)
        result = await worker.execute(task_msg, context)
        if result.type == AgentMessageType.ERROR or not result.content.strip():
            self._update_step(steps, step.id, step.with_failed(result.content or "empty result"))
            return

        accepted_result = result.content
        review = await reviewer.review(step.description, accepted_result)
        reviewer.clear_history()
        approved = self.parse_review_approval(review.content)
        issues = self.parse_review_issues(review.content)
        retries = retry_count.get(step.id, 0)
        # 审查不通过时重试，直到通过或达到最大重试次数
        while not approved and retries < self.max_retries_per_step:
            retries += 1
            retry_count[step.id] = retries
            # 将审查反馈注入上下文，让 Worker 知道上次失败原因
            retry_context = context + f"\n\nReviewer rejected the previous result:\n{issues}"
            retry_result = await worker.execute(task_msg, retry_context)
            if retry_result.type == AgentMessageType.ERROR or not retry_result.content.strip():
                issues = retry_result.content or "empty retry result"
                continue
            accepted_result = retry_result.content
            retry_review = await reviewer.review(step.description, accepted_result)
            reviewer.clear_history()
            approved = self.parse_review_approval(retry_review.content)
            issues = self.parse_review_issues(retry_review.content)

        self._update_step(steps, step.id, step.with_result(accepted_result))

    def parse_plan(self, plan_json: str) -> list[ExecutionStep]:
        """解析 Planner 输出的 JSON 执行计划为 ExecutionStep 列表。

        支持 {"steps": [...]} 或 {"tasks": [...]} 两种格式。
        会对 step id 做归一化（统一为 step_1, step_2, ...），
        同时将依赖关系中的原始 id 映射为归一化后的 id。
        """
        try:
            data = _parse_json_object(plan_json)
        except (json.JSONDecodeError, ValueError):
            return []
        nodes = data.get("steps") or data.get("tasks") or []
        if not isinstance(nodes, list) or not nodes:
            return []
        # 第一遍：创建 step 并建立 id 映射
        id_mapping: dict[str, str] = {}
        steps: list[ExecutionStep] = []
        for index, node in enumerate(nodes, start=1):
            if not isinstance(node, dict):
                continue
            original_id = str(node.get("id") or f"step_{index}")
            new_id = f"step_{index}"
            id_mapping[original_id] = new_id
            steps.append(
                ExecutionStep(
                    id=new_id,
                    description=str(node.get("description") or original_id),
                    type=str(node.get("type") or "COMMAND"),
                    dependencies=[],
                )
            )
        # 第二遍：填充依赖关系（将原始 id 映射为归一化 id）
        for index, node in enumerate(nodes, start=1):
            if not isinstance(node, dict) or index > len(steps):
                continue
            raw_deps = node.get("dependencies") or []
            if not isinstance(raw_deps, list):
                continue
            steps[index - 1].dependencies = [
                id_mapping.get(str(dep), str(dep)) for dep in raw_deps if str(dep)
            ]
        return steps

    def get_executable_steps(self, steps: list[ExecutionStep]) -> list[ExecutionStep]:
        """获取当前可执行的步骤：状态为 PENDING 且所有依赖已 COMPLETED。"""
        status = {step.id: step.status for step in steps}
        return [
            step
            for step in steps
            if step.status == StepStatus.PENDING
            and all(status.get(dep) == StepStatus.COMPLETED for dep in step.dependencies)
        ]

    def parse_review_approval(self, review_content: str | None) -> bool:
        """解析 Reviewer 的输出，判断是否通过审查。

        优先解析 JSON 中的 approved 字段，
        JSON 解析失败时回退到关键词匹配（支持中英文）。
        """
        if not review_content:
            return False
        try:
            data = _parse_json_object(review_content)
            if "approved" not in data:
                return False
            return bool(data.get("approved"))
        except (json.JSONDecodeError, ValueError):
            # JSON 解析失败时，用关键词兜底判断
            lower = review_content.lower()
            negative = ["未通过", "不通过", "不合格", "有问题", '"approved": false']
            positive = ["通过", "合格", '"approved": true']
            if any(item in lower for item in negative):
                return False
            return any(item in lower for item in positive)

    def parse_review_issues(self, review_content: str | None) -> str:
        """从 Reviewer 输出中提取问题列表（issues/suggestions 字段）。"""
        if not review_content:
            return ""
        try:
            data = _parse_json_object(review_content)
        except (json.JSONDecodeError, ValueError):
            return "review rejected the result"
        for key in ("issues", "suggestions"):
            value = data.get(key)
            if isinstance(value, list) and value:
                return "\n".join(f"- {item}" for item in value)
        return str(data.get("summary") or "review rejected the result")

    def build_step_context(self, steps: list[ExecutionStep], current_step: ExecutionStep) -> str:
        """构建步骤执行上下文：包含已完成依赖步骤的结果摘要（截断到 500 字）。"""
        lines = ["Overall task context:"]
        for step in steps:
            if step.id in current_step.dependencies and step.status == StepStatus.COMPLETED:
                lines.append(f"[{step.id}] {step.description}")
                if step.result:
                    lines.append(f"Result: {_preview(step.result, 500)}")
        return "\n".join(lines)

    def summarize_steps(self, steps: list[ExecutionStep]) -> str:
        """生成执行计划的文本摘要，展示每个步骤的 id、描述、类型和依赖。"""
        lines = ["Execution plan:"]
        for step in steps:
            deps = ", ".join(step.dependencies) if step.dependencies else "none"
            lines.append(f"- [{step.id}] {step.description} ({step.type}, deps: {deps})")
        return "\n".join(lines)

    def build_final_result(self, steps: list[ExecutionStep]) -> str:
        """汇总所有步骤状态和结果，生成最终输出文本。"""
        all_completed = all(step.status == StepStatus.COMPLETED for step in steps)
        failed = any(step.status == StepStatus.FAILED for step in steps)
        if all_completed:
            header = "Multi-Agent task completed."
        elif failed:
            header = "Multi-Agent task did not fully complete; failed steps remain."
        else:
            header = "Multi-Agent task partially completed; pending steps remain."
        lines = [header, "", "Execution summary:"]
        for step in steps:
            icon = {
                StepStatus.COMPLETED: "COMPLETED",
                StepStatus.FAILED: "FAILED",
                StepStatus.PENDING: "PENDING",
                StepStatus.RUNNING: "RUNNING",
            }[step.status]
            lines.append(f"- [{step.id}] {icon}: {step.description}")
            if step.result:
                lines.append(f"  Result: {_preview(step.result)}")
        return "\n".join(lines) + "\n"

    def _subagent(self, name: str, role: AgentRole) -> SubAgent:
        """创建一个指定名称和角色的子 Agent 实例。"""
        return SubAgent(
            name=name,
            role=role,
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
            skill_context_buffer=self.skill_context_buffer,
        )

    def _update_step(
        self,
        steps: list[ExecutionStep],
        step_id: str,
        updated: ExecutionStep,
    ) -> None:
        """通过 id 查找并原地更新步骤（dataclass 是不可变替换，需要写回列表）。"""
        for index, step in enumerate(steps):
            if step.id == step_id:
                steps[index] = updated
                return


def _parse_json_object(text: str) -> dict[str, Any]:
    """从 LLM 输出中提取并解析 JSON 对象（自动去除 markdown 代码块）。"""
    cleaned = re.sub(r"```(?:json)?\s*", "", text or "").replace("```", "").strip()
    if not cleaned:
        raise ValueError("empty JSON")
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def _preview(text: str, max_len: int = 160) -> str:
    """截断文本到指定长度，超长部分用 ... 替代。"""
    value = (text or "").replace("\r\n", "\n").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."

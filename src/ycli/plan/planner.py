"""规划器模块。

将用户的自然语言目标转换为结构化的执行计划（Task DAG）。
Planner 通过 LLM 生成分步计划（JSON 格式），解析后构建 ExecutionPlan 对象。
对于简单任务（如"列出文件"、"搜索 xxx"），会自动短路生成单任务计划，
避免不必要的 LLM 调用开销。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from ycli.llm.base import LlmClient
from ycli.plan.models import ExecutionPlan, Task, TaskType
from ycli.types import Message

# 规划器的系统提示词，要求 LLM 输出符合指定 JSON Schema 的执行计划
PLANNER_PROMPT = """You are YCLI's planner.
Create a compact executable DAG for the user's task.
Return only JSON with this shape:
{
  "summary": "short summary",
  "tasks": [
    {
      "id": "stable_source_id",
      "description": "concrete executable step",
      "type": "FILE_READ|FILE_WRITE|COMMAND|ANALYSIS|VERIFICATION",
      "dependencies": ["stable_source_id"]
    }
  ]
}
Use independent tasks when they can run in parallel.
"""


class Planner:
    """执行计划规划器。

    通过 LLM 将用户目标转换为 ExecutionPlan（Task DAG）。
    - create_plan(): 根据目标创建执行计划（简单任务自动短路）。
    - replan(): 基于失败计划重新规划，附带失败原因和已完成任务信息。
    - parse_plan(): 解析 LLM 输出的 JSON 文本为 ExecutionPlan 对象。
    """

    def __init__(self, llm_client: LlmClient):
        self.llm_client = llm_client

    async def create_plan(self, goal: str) -> ExecutionPlan:
        """创建执行计划。简单目标（<30 字且不含多步提示词）会自动短路为单任务计划。"""
        if _is_simple_goal(goal):
            return _minimal_plan(goal)
        text = await _collect_text(
            self.llm_client,
            [Message(role="user", content=f"Please create an execution plan for:\n{goal}")],
            system_prompt=PLANNER_PROMPT,
        )
        return self.parse_plan(goal, text)

    async def replan(self, failed_plan: ExecutionPlan, failure_reason: str) -> ExecutionPlan:
        """基于失败计划重新规划，将失败原因和已完成任务信息注入新的目标描述中。"""
        completed = "\n".join(
            f"- {task.id}: {task.description}"
            for task in failed_plan.all_tasks()
            if task.result and not task.error
        )
        return await self.create_plan(
            f"{failed_plan.goal}\nFailure reason: {failure_reason}\nCompleted tasks:\n{completed}"
        )

    def parse_plan(self, goal: str, plan_json: str) -> ExecutionPlan:
        """将 LLM 输出的 JSON 文本解析为 ExecutionPlan。

        流程：提取 JSON → 遍历 tasks 数组创建 Task 对象 → 映射依赖关系 → 拓扑排序验证无环。
        任务 id 会被统一重命名为 task_1, task_2, ... 以确保一致性。
        """
        data = _parse_json_object(plan_json)
        task_nodes = data.get("tasks") or data.get("steps") or []
        if not isinstance(task_nodes, list) or not task_nodes:
            raise ValueError("planner output did not contain a non-empty tasks/steps array")

        plan = ExecutionPlan(id=f"plan_{int(time.time() * 1000)}", goal=goal)
        plan.summary = str(data.get("summary") or "")
        id_mapping: dict[str, str] = {}

        for index, node in enumerate(task_nodes, start=1):
            if not isinstance(node, dict):
                continue
            original_id = str(node.get("id") or f"task_{index}")
            new_id = f"task_{index}"
            id_mapping[original_id] = new_id
            plan.add_task(
                Task(
                    id=new_id,
                    description=str(node.get("description") or original_id),
                    type=_parse_task_type(str(node.get("type") or "ANALYSIS")),
                )
            )

        for index, node in enumerate(task_nodes, start=1):
            if not isinstance(node, dict):
                continue
            task = plan.get_task(f"task_{index}")
            if not task:
                continue
            dependencies = node.get("dependencies") or []
            if not isinstance(dependencies, list):
                continue
            for raw_dep in dependencies:
                dep_id = id_mapping.get(str(raw_dep), str(raw_dep))
                if dep_id in plan.tasks:
                    task.add_dependency(dep_id)
                    plan.tasks[dep_id].add_dependent(task.id)

        if not plan.compute_execution_order():
            raise ValueError("plan contains a cyclic dependency")
        return plan


async def _collect_text(
    llm_client: LlmClient,
    messages: list[Message],
    *,
    system_prompt: str,
) -> str:
    """从 LLM 流式响应中收集完整的文本输出。"""
    text = ""
    async for event in llm_client.chat(messages, [], system_prompt=system_prompt):
        event_type = event.get("type")
        if event_type == "text_delta":
            text += str(event.get("text") or "")
        elif event_type == "error":
            raise event["error"]
    return text


def _parse_json_object(text: str) -> dict[str, Any]:
    """从 LLM 输出文本中提取 JSON 对象（去除 markdown 代码块包裹）。"""
    cleaned = re.sub(r"```(?:json)?\s*", "", text or "").replace("```", "").strip()
    if not cleaned:
        raise ValueError("empty planner output")
    return json.loads(cleaned)


def _parse_task_type(value: str) -> TaskType:
    """解析任务类型字符串，无法识别时默认返回 ANALYSIS。"""
    normalized = value.upper()
    try:
        return TaskType(normalized)
    except ValueError:
        return TaskType.ANALYSIS


def _is_simple_goal(goal: str | None) -> bool:
    """简单任务短路判断。

    满足以下条件的目标被视为简单任务，直接生成单任务计划而不调用 LLM：
    - 非空且长度 ≤ 30 字符。
    - 不包含多步骤提示词（如"然后"、"并且"、"先"等）。
    - 包含简单动作提示词（如"列出"、"查看"、"运行"等）。
    """
    normalized = (goal or "").strip()
    if not normalized or len(normalized) > 30:
        return False
    multi_step_cues = ["然后", "并且", "再", "最后", "同时", "先", "之后", "接着", "以及"]
    if any(cue in normalized for cue in multi_step_cues):
        return False
    simple_cues = ["列出", "查看", "读取", "显示", "执行", "运行", "搜索", "当前目录", "文件"]
    return any(cue in normalized for cue in simple_cues)


def _minimal_plan(goal: str) -> ExecutionPlan:
    """为简单任务生成单任务最小计划，跳过 LLM 调用。"""
    normalized = goal.strip()
    plan = ExecutionPlan(id=f"plan_{int(time.time() * 1000)}", goal=normalized)
    plan.summary = f"直接执行简单任务：{normalized}"
    plan.add_task(Task(id="task_1", description=normalized, type=_infer_simple_type(normalized)))
    plan.compute_execution_order()
    return plan


def _infer_simple_type(goal: str) -> TaskType:
    """根据目标文本中的关键词推断简单任务的类型。"""
    if any(token in goal for token in ["读取", "打开", "查看"]) and "文件" in goal:
        return TaskType.FILE_READ
    if any(token in goal for token in ["写入", "修改", "创建文件"]):
        return TaskType.FILE_WRITE
    if any(token in goal for token in ["分析", "总结", "解释"]):
        return TaskType.ANALYSIS
    if any(token in goal for token in ["验证", "检查"]):
        return TaskType.VERIFICATION
    return TaskType.COMMAND

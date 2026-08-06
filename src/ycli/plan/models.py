"""执行计划模型模块（Task DAG）。

定义任务执行计划的核心数据结构：
- Task: 单个可执行任务，支持依赖关系（DAG 结构）。
- ExecutionPlan: 任务执行计划，管理一组 Task 的生命周期。
- TaskType / TaskStatus / PlanStatus: 相关枚举类型。

ExecutionPlan 提供 DAG 遍历能力：拓扑排序（compute_execution_order）、
执行批次计算（execution_batches）、进度查询等，
供 PlanExecuteAgent 和 AgentOrchestrator 使用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum


class TaskType(StrEnum):
    """任务类型枚举，描述任务的执行方式。"""

    PLANNING = "PLANNING"
    FILE_READ = "FILE_READ"
    FILE_WRITE = "FILE_WRITE"
    COMMAND = "COMMAND"
    ANALYSIS = "ANALYSIS"
    VERIFICATION = "VERIFICATION"


class TaskStatus(StrEnum):
    """任务状态枚举，描述任务在执行周期中的当前阶段。"""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class PlanStatus(StrEnum):
    """执行计划整体状态枚举。"""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(slots=True)
class Task:
    """执行计划中的单个任务节点。

    通过 dependencies（前置依赖）和 dependents（后续依赖）构成 DAG 结构。
    只有当所有前置依赖都已完成（COMPLETED）时，任务才是可执行的（is_executable）。
    """

    id: str
    description: str
    type: TaskType = TaskType.ANALYSIS
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    start_time: float = 0.0
    end_time: float = 0.0

    def add_dependency(self, task_id: str) -> None:
        """添加一个前置依赖任务（当前任务必须在该任务完成后才能执行）。"""
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)

    def add_dependent(self, task_id: str) -> None:
        """添加一个后续依赖任务（该任务必须在当前任务完成后才能执行）。"""
        if task_id not in self.dependents:
            self.dependents.append(task_id)

    def mark_started(self) -> None:
        self.status = TaskStatus.RUNNING
        self.start_time = time.time()

    def mark_completed(self, result: str) -> None:
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.end_time = time.time()

    def mark_failed(self, error: str) -> None:
        self.status = TaskStatus.FAILED
        self.error = error
        self.end_time = time.time()

    def mark_skipped(self) -> None:
        self.status = TaskStatus.SKIPPED
        self.end_time = time.time()

    def is_executable(self, all_tasks: dict[str, Task]) -> bool:
        """判断当前任务是否可执行（状态为 PENDING 且所有前置依赖已完成）。"""
        if self.status != TaskStatus.PENDING:
            return False
        return all(
            dep_id in all_tasks and all_tasks[dep_id].status == TaskStatus.COMPLETED
            for dep_id in self.dependencies
        )


@dataclass(slots=True)
class ExecutionPlan:
    """任务执行计划，管理一组有依赖关系的 Task。

    核心能力：
    - add_task(): 添加任务并自动维护双向依赖关系。
    - compute_execution_order(): 拓扑排序，检测循环依赖。
    - execution_batches(): 计算可并行执行的批次（同批次内任务互不依赖）。
    - progress() / is_all_completed() / has_failed(): 查询执行状态。
    - summarize(): 生成计划的文本摘要。
    """

    id: str
    goal: str
    tasks: dict[str, Task] = field(default_factory=dict)
    status: PlanStatus = PlanStatus.CREATED
    summary: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    _execution_order: list[str] = field(default_factory=list)

    def add_task(self, task: Task) -> None:
        """添加任务到计划中，并自动维护前置/后续依赖的双向引用。"""
        self.tasks[task.id] = task
        for dep_id in task.dependencies:
            dep = self.tasks.get(dep_id)
            if dep:
                dep.add_dependent(task.id)
        for existing in self.tasks.values():
            if task.id in existing.dependencies:
                task.add_dependent(existing.id)
        self._execution_order.clear()

    def get_task(self, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    def all_tasks(self) -> list[Task]:
        return list(self.tasks.values())

    def root_tasks(self) -> list[Task]:
        """返回所有无前置依赖的根任务（DAG 的起点）。"""
        return [task for task in self.tasks.values() if not task.dependencies]

    def executable_tasks(self) -> list[Task]:
        """返回当前所有可立即执行的任务（PENDING 且依赖已全部完成）。"""
        return [task for task in self.tasks.values() if task.is_executable(self.tasks)]

    def compute_execution_order(self) -> bool:
        """通过 DFS 拓扑排序计算任务执行顺序。如果存在循环依赖则返回 False。"""
        order: list[str] = []
        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(task: Task) -> bool:
            if task.id in visiting:
                return False
            if task.id in visited:
                return True
            visiting.add(task.id)
            for dep_id in task.dependencies:
                dep = self.tasks.get(dep_id)
                if dep and not visit(dep):
                    return False
            visiting.remove(task.id)
            visited.add(task.id)
            order.append(task.id)
            return True

        for task in self.tasks.values():
            if task.id not in visited and not visit(task):
                return False
        self._execution_order = order
        return True

    def execution_order(self) -> list[str]:
        if not self._execution_order:
            self.compute_execution_order()
        return list(self._execution_order)

    def execution_batches(self) -> list[list[Task]]:
        """将任务划分为可并行执行的批次。

        每个批次内的任务互不依赖，可以并发执行；
        批次之间按依赖顺序串行推进。
        """
        remaining = set(self.tasks)
        completed: set[str] = set()
        batches: list[list[Task]] = []
        while remaining:
            batch_ids = [
                task_id
                for task_id in self.execution_order()
                if task_id in remaining
                and all(dep_id in completed for dep_id in self.tasks[task_id].dependencies)
            ]
            if not batch_ids:
                break
            batch = [self.tasks[task_id] for task_id in batch_ids]
            batches.append(batch)
            completed.update(batch_ids)
            remaining.difference_update(batch_ids)
        return batches

    def progress(self) -> float:
        """返回执行进度（0.0 ~ 1.0），即已完成任务占总任务的比例。"""
        if not self.tasks:
            return 1.0
        completed = sum(1 for task in self.tasks.values() if task.status == TaskStatus.COMPLETED)
        return completed / len(self.tasks)

    def is_all_completed(self) -> bool:
        return all(task.status == TaskStatus.COMPLETED for task in self.tasks.values())

    def has_failed(self) -> bool:
        return any(task.status == TaskStatus.FAILED for task in self.tasks.values())

    def mark_started(self) -> None:
        self.status = PlanStatus.RUNNING
        self.start_time = time.time()

    def mark_completed(self) -> None:
        self.status = PlanStatus.COMPLETED
        self.end_time = time.time()

    def mark_failed(self) -> None:
        self.status = PlanStatus.FAILED
        self.end_time = time.time()

    def summarize(self) -> str:
        """生成执行计划的文本摘要，包含任务数、批次数、首末批次信息。"""
        batches = self.execution_batches()
        first_batch = ", ".join(task.id for task in batches[0]) if batches else "none"
        final_batch = ", ".join(task.id for task in batches[-1]) if batches else "none"
        return (
            f"Plan {self.id}: {self.summary or self.goal}\n"
            f"Tasks: {len(self.tasks)} | Parallel batches: {len(batches)} | "
            f"Executable now: {len(self.executable_tasks())}\n"
            f"First batch: {first_batch}\n"
            f"Final convergence: {final_batch}"
        )

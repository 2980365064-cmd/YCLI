"""系统提示词（System Prompt）组装模块。

根据当前配置、工作目录、可用工具列表、模型信息等上下文，
动态拼装发送给 LLM 的 system prompt。拼装内容包括：
- 基础身份描述（YCLI 角色定义）
- 当前时间、工作目录、模型信息、可用工具列表
- 行为指引（Guidelines）
- 项目级记忆文件（YAI.md）和 SQLite 长期记忆
- Skill 索引文本（列出已启用的技能摘要）
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ycli.config import YcliConfig
from ycli.memory import MemoryManager
from ycli.skill import SkillRegistry


class PromptAssembler:
    """系统提示词组装器。

    根据当前运行上下文（配置、工作目录、工具列表、模型信息）
    动态构建发送给 LLM 的 system prompt。
    通过 build() 方法将各部分拼装为完整的字符串。
    """

    def __init__(
        self,
        config: YcliConfig,
        cwd: str,
        tool_names: list[str],
        model: str,
        provider: str,
    ):
        self.config = config
        self.cwd = str(Path(cwd).resolve())
        self.tool_names = tool_names
        self.model = model
        self.provider = provider

    def build(self) -> str:
        """组装完整的 system prompt 字符串。

        拼装顺序：基础身份描述 → 运行时上下文 → 行为指引 → 项目记忆 → Skill 索引。
        最终文本以换行连接，总长度不做截断（由各输入源自行控制上限）。
        """
        parts = [
            "You are YCLI, a powerful AI coding assistant running in a terminal.",
            f"Current time: {datetime.now().isoformat(timespec='seconds')}",
            f"Working directory: {self.cwd}",
            f"Model: {self.model} ({self.provider})",
            f"Available tools: {', '.join(self.tool_names)}",
            "",
            "Guidelines:",
            "- Be concise, direct, and implementation-oriented.",
            "- Use tools to inspect files, search code, and verify behavior when needed.",
            "- Prefer deterministic local tools before guessing.",
            "- When writing files, use write_file and keep changes scoped.",
            "- Preserve URLs and user-provided identifiers exactly unless a tool result proves "
            "otherwise.",
            "- Ask a clarifying question only when proceeding would be risky.",
        ]
        project_memory = self._project_memory()
        if project_memory:
            parts.extend(["", "Project memory:", project_memory])
        skill_index = SkillRegistry(self.cwd).index_text() if self.config.features.skill else ""
        if skill_index:
            parts.extend(["", skill_index])
        return "\n".join(parts)

    def _project_memory(self) -> str:
        """收集项目级记忆上下文。

        依次检查以下位置的 YAI.md 文件（每个最多读取 4000 字符）：
        项目根目录和 .ycli/ 下的 YAI.md 与 YAI.local.md。
        如果启用了长期记忆（SQLite），还会追加最近 8 条记忆。
        所有内容拼接后截断到 8000 字符。
        """
        memory_files = [
            Path(self.cwd) / "YAI.md",
            Path(self.cwd) / ".ycli" / "YAI.md",
            Path(self.cwd) / "YAI.local.md",
            Path(self.cwd) / ".ycli" / "YAI.local.md",
        ]
        chunks = []
        for path in memory_files:
            if path.exists():
                try:
                    chunks.append(path.read_text(encoding="utf-8")[:4000])
                except OSError:
                    continue
        if self.config.features.memory and self.config.memory.long_term_enabled:
            manager = MemoryManager(self.config.memory.long_term_db_path, scope=self.cwd)
            memories = manager.list(limit=8)
            if memories:
                chunks.append("\n".join(f"- {item.content}" for item in memories))
        return "\n\n".join(chunks)[:8000]

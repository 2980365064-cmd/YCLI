"""工具注册表 — 工具名称到 Tool 实例的映射。

ToolRegistry 是工具的存储中心，Agent 启动时通过它获取所有可用工具，
并在每轮对话中将工具定义发送给 LLM。
"""

from __future__ import annotations

from ycli.tools.base import Tool


class ToolRegistry:
    """工具注册表。管理所有已注册的工具，提供按名称查找和批量导出功能。"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_all(self, tools: list[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return sorted(self._tools)

    def definitions(self) -> list[dict]:
        """导出所有工具的 LLM function calling 格式定义。"""
        return [self._tools[name].definition() for name in self.list_names()]

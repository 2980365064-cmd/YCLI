"""YCLI 引导模块 — 工具注册表的初始化入口。

负责将内置工具(Builtin Tools)和 MCP 远程工具统一注册到 ToolRegistry 中，
是 Agent 启动流程的第一步：先有工具，才能执行任务。
"""

from __future__ import annotations

from ycli.config import YcliConfig
from ycli.mcp import McpClientManager
from ycli.tools import ToolRegistry, get_builtin_tools


async def build_tool_registry(
    *,
    config: YcliConfig,
    cwd: str,
) -> tuple[ToolRegistry, McpClientManager | None]:
    """构建完整的工具注册表。

    流程：
    1. 注册所有内置工具（read_file, write_file, bash, web_search 等）
    2. 如果启用了 MCP，连接 MCP server 并注册远程工具

    Returns:
        (registry, manager) — manager 为 None 表示未启用 MCP。
    """
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    manager: McpClientManager | None = None
    if config.features.mcp:
        manager = McpClientManager(cwd)
        registry.register_all(await manager.load_tools())
    return registry, manager

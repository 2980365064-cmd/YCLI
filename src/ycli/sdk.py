"""YCLI 公共 SDK 接口。

提供 create_default_engine() 工厂函数，用于以编程方式创建 QueryEngine。
这是一个 Facade（外观模式），封装了配置加载、LLM 客户端创建、工具注册等初始化流程。

用法示例::

    from ycli.sdk import create_default_engine
    engine = create_default_engine(cwd="/path/to/project")
    result = engine.complete("帮我分析这个项目")
"""

from __future__ import annotations

from pathlib import Path

from ycli.agent import QueryEngine
from ycli.config import load_config
from ycli.llm import create_llm_client
from ycli.tools import ToolRegistry, get_builtin_tools


def create_default_engine(cwd: str | None = None) -> QueryEngine:
    """创建一个配置好的 QueryEngine 实例。

    Args:
        cwd: 工作目录，默认为当前目录。

    Returns:
        已就绪的 QueryEngine，可直接调用 complete() / ask_complete()。
    """
    root = str(Path(cwd or ".").resolve())
    config = load_config(project_root=root)
    client = create_llm_client(config.llm)
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    return QueryEngine(llm_client=client, tool_registry=registry, config=config, cwd=root)


__all__ = ["QueryEngine", "create_default_engine"]

"""YCLI 配置系统。

采用层级合并策略（优先级从低到高）：
  默认值 → ~/.ycli/config.json (用户级) → .ycli/config.json (项目级)
  → .env 文件 → overrides 参数 → 环境变量

环境变量前缀为 YCLI_（如 YCLI_API_KEY, YCLI_PROVIDER 等）。
各 LLM 提供商的专属 Key（DEEPSEEK_API_KEY 等）在 YCLI_API_KEY 未设置时自动回退。
"""

from __future__ import annotations

import json
import os
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# YCLI 用户级配置根目录（~/.ycli/）
YCLI_HOME = Path.home() / ".ycli"

__all__ = [
    "YCLI_HOME",
    "YcliConfig",
    "LlmConfig",
    "ToolsConfig",
    "McpConfig",
    "MemoryConfig",
    "PolicyConfig",
    "PromptConfig",
    "FeatureConfig",
    "load_config",
    "get_config_paths",
    "config_to_public_dict",
]


def _home() -> Path:
    return Path.home()


@dataclass(slots=True)
class LlmConfig:
    """LLM 模型配置。"""

    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 8192
    temperature: float = 0.7
    timeout: float = 120.0


@dataclass(slots=True)
class ToolsConfig:
    """工具执行配置。"""

    enabled: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    timeout: float = 60.0
    batch_timeout: float = 90.0
    max_concurrent_read: int = 4


@dataclass(slots=True)
class McpConfig:
    """MCP（Model Context Protocol）服务器配置。"""

    servers: list[dict[str, Any]] = field(default_factory=list)
    auto_start: bool = True


@dataclass(slots=True)
class MemoryConfig:
    """记忆系统配置。"""

    max_conversation_history: int = 100
    long_term_enabled: bool = True
    long_term_db_path: str = "~/.ycli/memory.db"
    token_budget_mode: str = "balanced"
    compression_threshold: float = 0.8


@dataclass(slots=True)
class PolicyConfig:
    """安全策略配置（HITL 审批模式 + 危险命令黑名单 + 路径守卫）。"""

    hitl_mode: str = "auto"  # never=全放行, auto=按需审批, always=全审批
    path_guard_enabled: bool = True
    command_blacklist: list[str] = field(
        default_factory=lambda: [
            "sudo",
            "rm -rf /",
            "rm -rf ~",
            "mkfs",
            "dd if=/dev/zero",
            ":(){:|:&};:",  # fork 炸弹
            "chmod -R 777 /",
            "curl | sh",
            "curl|sh",
            "shutdown",
            "reboot",
        ]
    )
    audit_log_path: str = "~/.ycli/audit.jsonl"


@dataclass(slots=True)
class PromptConfig:
    """系统提示配置。"""

    personality: str = "default"
    agent_mode: str = "react"
    custom_prompt_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FeatureConfig:
    """功能开关（可按需关闭不需要的子系统）。"""

    mcp: bool = True
    skill: bool = True
    memory: bool = True
    audit_log: bool = True
    context_compression: bool = True
    code_index: bool = True


@dataclass(slots=True)
class YcliConfig:
    """YCLI 顶层配置，聚合所有子系统配置。"""

    llm: LlmConfig = field(default_factory=LlmConfig)
    render_mode: str = "inline"
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)


def load_config(
    project_root: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    env: dict[str, str | None] | None = None,
) -> YcliConfig:
    """加载配置（层级合并）。

    优先级（从低到高）：默认值 → 用户 JSON → 项目 JSON → .env → overrides → 环境变量。
    """
    env_map = env if env is not None else os.environ
    data = _config_to_dict(YcliConfig())

    user_config = _read_json(_home() / ".ycli" / "config.json")
    if user_config:
        data = _deep_merge(data, user_config)

    root = Path(project_root).resolve() if project_root else None
    if root:
        project_config = _read_json(root / ".ycli" / "config.json")
        if project_config:
            data = _deep_merge(data, project_config)
        project_env = _read_env(root / ".env")
        if project_env:
            data = _apply_env(data, project_env)

    if overrides:
        data = _deep_merge(data, overrides)

    data = _apply_env(data, env_map)
    config = _dict_to_config(data)
    config.memory.long_term_db_path = _expand_home(config.memory.long_term_db_path)
    config.policy.audit_log_path = _expand_home(config.policy.audit_log_path)
    return config


def get_config_paths(project_root: str | Path | None = None) -> list[Path]:
    paths = [_home() / ".ycli" / "config.json"]
    if project_root:
        paths.append(Path(project_root).resolve() / ".ycli" / "config.json")
    return paths


def config_to_public_dict(config: YcliConfig) -> dict[str, Any]:
    data = _config_to_dict(config)
    if data.get("llm", {}).get("api_key"):
        data["llm"]["api_key"] = "***"
    return data


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _read_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        result[key] = value
    return result


def _apply_env(data: dict[str, Any], env: dict[str, str | None]) -> dict[str, Any]:
    result = deepcopy(data)
    llm = result.setdefault("llm", {})
    features = result.setdefault("features", {})
    policy = result.setdefault("policy", {})

    mappings: list[tuple[str, str, Any]] = [
        ("YCLI_API_KEY", "api_key", str),
        ("YCLI_PROVIDER", "provider", str),
        ("YCLI_MODEL", "model", str),
        ("YCLI_BASE_URL", "base_url", str),
        ("YCLI_MAX_TOKENS", "max_tokens", int),
        ("YCLI_TEMPERATURE", "temperature", float),
    ]
    for env_key, config_key, caster in mappings:
        raw = env.get(env_key)
        if raw not in (None, ""):
            with suppress(TypeError, ValueError):
                llm[config_key] = caster(raw)

    provider = str(llm.get("provider") or "").lower()
    if not llm.get("api_key"):
        provider_key_map = {
            "deepseek": "DEEPSEEK_API_KEY",
            "glm": "GLM_API_KEY",
            "zhipu": "GLM_API_KEY",
            "step": "STEP_API_KEY",
            "kimi": "KIMI_API_KEY",
            "moonshot": "KIMI_API_KEY",
            "freellmapi": "FREELLMAPI_API_KEY",
            "xfyun": "XFYUN_API_KEY",
            "agnes": "AGNES_API_KEY",
        }
        provider_key = provider_key_map.get(provider)
        if provider_key and env.get(provider_key):
            llm["api_key"] = env[provider_key]

    provider_model_key = f"{provider.upper()}_MODEL" if provider else ""
    provider_base_url_key = f"{provider.upper()}_BASE_URL" if provider else ""
    if provider_model_key and env.get(provider_model_key):
        llm["model"] = env[provider_model_key]
    if provider_base_url_key and env.get(provider_base_url_key):
        llm["base_url"] = env[provider_base_url_key]

    render_mode = env.get("YCLI_RENDER_MODE") or env.get("YCLI_RENDERER")
    if render_mode in {"plain", "inline"}:
        result["render_mode"] = render_mode

    if env.get("YCLI_TUI") == "true":
        result["render_mode"] = "inline"

    for env_key, feature_key in [
        ("YCLI_MCP", "mcp"),
        ("YCLI_SKILL", "skill"),
        ("YCLI_MEMORY", "memory"),
    ]:
        raw = env.get(env_key)
        if raw == "false":
            features[feature_key] = False
        elif raw == "true":
            features[feature_key] = True

    hitl = env.get("YCLI_HITL")
    if hitl in {"always", "auto", "never"}:
        policy["hitl_mode"] = hitl

    return result


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(target)
    for key, value in source.items():
        if value is None:
            continue
        old = result.get(key)
        if isinstance(old, dict) and isinstance(value, dict):
            result[key] = _deep_merge(old, value)
        else:
            result[key] = deepcopy(value)
    return result


def _config_to_dict(config: YcliConfig) -> dict[str, Any]:
    return asdict(config)


def _dict_to_config(data: dict[str, Any]) -> YcliConfig:
    return YcliConfig(
        llm=LlmConfig(**data.get("llm", {})),
        render_mode=data.get("render_mode", "inline"),
        tools=ToolsConfig(**data.get("tools", {})),
        mcp=McpConfig(**data.get("mcp", {})),
        memory=MemoryConfig(**data.get("memory", {})),
        policy=PolicyConfig(**data.get("policy", {})),
        prompt=PromptConfig(**data.get("prompt", {})),
        features=FeatureConfig(**data.get("features", {})),
    )


def _expand_home(path: str) -> str:
    return str(Path(path).expanduser())

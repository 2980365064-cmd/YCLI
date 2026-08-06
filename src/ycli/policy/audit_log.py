"""审计日志（Audit Log）——以 JSONL 格式记录工具调用的执行事件。

本模块提供 :class:`AuditLog`，每次工具调用都会以追加方式写入一条 JSON 记录，
包含时间戳、工具名、输入参数、执行结果、审批人以及当前工作目录。

**敏感信息脱敏**：写入前会通过 :meth:`AuditLog._redact` 递归扫描输入参数，
将键名命中 :data:`SENSITIVE_KEYS` 中任一关键字（如 ``token``、``password``、
``authorization`` 等）的字段值替换为 ``"***"``，避免密钥、密码等敏感数据被
明文落盘。

**日志格式**：每行一条合法的 JSON 对象，形如::

    {
        "timestamp": "2026-08-06T12:34:56.789012+00:00",
        "tool_name":  "write_file",
        "input":      {"path": "/workspace/foo.py", "content": "..."},
        "outcome":    "approved",
        "approver":   "user",
        "cwd":        "/workspace"
    }
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# 敏感字段关键字列表：输入参数中键名（小写后）包含这些字符串时，值会被替换为 "***"
SENSITIVE_KEYS = ("token", "key", "password", "secret", "authorization", "bearer")


class AuditLog:
    """基于 JSONL 的审计日志记录器。

    每个 :class:`AuditLog` 实例绑定一个日志文件路径。调用 :meth:`record` 时，
    会以 JSON Lines 格式在该文件末尾追加一条事件记录；调用 :meth:`tail` 可
    读取最近的若干条记录，方便在 REPL 中回看审计历史。
    """

    def __init__(self, path: str | Path):
        # expanduser() 支持形如 "~/.ycli/audit.jsonl" 的路径写法
        self.path = Path(path).expanduser()

    def record(
        self,
        *,
        tool_name: str,
        input_data: dict[str, Any],
        outcome: str,
        approver: str,
        cwd: str,
    ) -> None:
        """向审计日志追加一条工具调用事件。

        Parameters
        ----------
        tool_name:
            被调用的工具名称，例如 ``"write_file"``、``"run_shell"``。
        input_data:
            工具的输入参数字典；写入前会经过 :meth:`_redact` 脱敏处理。
        outcome:
            执行结果描述，例如 ``"approved"``、``"rejected"``、``"error"``。
        approver:
            审批人标识，例如 ``"user"``（人工审批）或 ``"auto"``（自动放行）。
        cwd:
            工具执行时的当前工作目录。
        """
        # 确保日志文件所在目录存在（首次写入时自动创建）
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.now(UTC).isoformat(),  # UTC 时间戳，便于跨时区排序
            "tool_name": tool_name,
            "input": self._redact(input_data),  # 对敏感字段做脱敏处理
            "outcome": outcome,
            "approver": approver,
            "cwd": cwd,
        }
        # 以追加模式写入，每条记录独占一行（JSONL 格式）
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def tail(self, limit: int = 20) -> list[dict[str, Any]]:
        """读取日志文件末尾最近的若干条记录。

        Parameters
        ----------
        limit:
            返回的最大记录条数，默认为 20。

        Returns
        -------
        list[dict[str, Any]]
            按文件顺序排列的事件字典列表；若日志文件不存在则返回空列表。
            解析失败的脏行会被静默跳过，不影响其他记录的返回。
        """
        if not self.path.exists():
            return []
        # 取文件最后 limit 行，避免一次性把整个日志加载进内存
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        events = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # 跳过格式损坏的行，保证尾部读取的健壮性
                continue
        return events

    def _redact(self, value: Any) -> Any:
        """递归地对 ``value`` 中的敏感字段进行脱敏处理。

        脱敏规则：若字典键名（忽略大小写）包含 :data:`SENSITIVE_KEYS` 中的任一
        关键字（例如 ``token``、``password``、``authorization`` 等），则将该键
        对应的值替换为 ``"***"``；否则递归处理子结构。

        Parameters
        ----------
        value:
            待脱敏的值，可以是字典、列表或基本类型。

        Returns
        -------
        Any
            脱敏后的同结构副本，原值不会被修改。
        """
        if isinstance(value, dict):
            redacted = {}
            for key, item in value.items():
                # 检查键名是否命中敏感关键字列表（忽略大小写）
                if any(marker in key.lower() for marker in SENSITIVE_KEYS):
                    redacted[key] = "***"  # 命中则用占位符替换
                else:
                    # 未命中则递归处理子值，保证嵌套结构也能被脱敏
                    redacted[key] = self._redact(item)
            return redacted
        if isinstance(value, list):
            # 列表中每个元素都需递归检查
            return [self._redact(item) for item in value]
        # 基本类型（str / int / bool / None 等）直接返回
        return value

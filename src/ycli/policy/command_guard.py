"""命令守卫（Command Guard）——在 HITL（人机协同）审批之前快速拦截明显危险的 shell 命令。

本模块维护两类过滤机制：

1. **黑名单关键字**（``blacklist``）：由调用方自定义的字符串列表，只要命令中
   包含其中任一子串即被拒绝；
2. **内置正则模式**（``patterns``）：针对典型的破坏性命令（如 ``rm -rf /``、
   ``mkfs``、fork 炸弹等）预先编写的正则表达式。

两层检查任一命中，即抛出 :class:`CommandPolicyError`，阻止命令进入执行阶段。
"""

from __future__ import annotations

import re


class CommandPolicyError(ValueError):
    """命令被策略拒绝时抛出的错误。

    继承自 :class:`ValueError`，便于上层统一捕获策略类异常。
    """

    pass


class CommandGuard:
    """在进入 HITL 审批流程之前，快速拦截明显危险 / 破坏性的 shell 命令。

    构造时可传入自定义黑名单 ``blacklist``（字符串列表），同时内部会初始化一组
    预定义的正则 ``patterns``，覆盖常见的系统破坏性命令。

    调用 :meth:`validate` 时按以下顺序检查：

    1. 将命令规范化（合并多余空白）；
    2. 逐个匹配 ``blacklist`` 中的关键字子串，命中即拒绝；
    3. 逐个匹配 ``patterns`` 中的正则表达式，命中即拒绝；
    4. 全部通过则不抛出异常，允许命令继续流转。
    """

    def __init__(self, blacklist: list[str] | None = None):
        # 用户自定义的黑名单关键字，为空字符串或 None 时跳过
        self.blacklist = blacklist or []
        # 内置破坏性命令的正则模式
        self.patterns = [
            # rm -rf /：递归强制删除根目录
            re.compile(r"\brm\s+-[^\n]*r[^\n]*f\s+/(?:\s|$)"),
            # rm -rf ~：递归强制删除用户主目录
            re.compile(r"\brm\s+-[^\n]*r[^\n]*f\s+~(?:\s|$)"),
            # mkfs：格式化磁盘
            re.compile(r"\bmkfs(?:\s|$)"),
            # dd if=/dev/zero：用零覆盖磁盘
            re.compile(r"\bdd\s+if=/dev/zero"),
            # :(){ :|:& };：经典的 bash fork 炸弹
            re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:"),
            # chmod -R 777 /：将根目录权限开放为 777
            re.compile(r"\bchmod\s+-R\s+777\s+/"),
            # shutdown / reboot：关机 / 重启
            re.compile(r"\bshutdown(?:\s|$)"),
            re.compile(r"\breboot(?:\s|$)"),
            # find /：扫描整棵根文件系统，常是误操作
            re.compile(r"\bfind\s+/(\s|$)"),
        ]

    def validate(self, command: str) -> None:
        """校验 ``command`` 是否违反策略，命中即抛出 :class:`CommandPolicyError`。

        Parameters
        ----------
        command:
            待校验的 shell 命令字符串。

        Raises
        ------
        CommandPolicyError
            当命令命中黑名单关键字或内置破坏性模式时抛出。
        """
        # 规范化：去除首尾空白，把连续空白压缩成单个空格，便于稳定匹配
        normalized = " ".join(command.strip().split())
        # 第一层：用户自定义的黑名单关键字子串匹配
        for blocked in self.blacklist:
            if blocked and blocked in normalized:
                raise CommandPolicyError(f"command rejected by policy: {blocked}")
        # 第二层：内置正则模式匹配破坏性命令
        for pattern in self.patterns:
            if pattern.search(normalized):
                raise CommandPolicyError("command rejected by destructive-command policy")

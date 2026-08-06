"""路径守卫（Path Guard）——限制文件读写操作必须在工作区（workspace）内。

本模块提供 :class:`PathGuard`，通过把目标路径解析成绝对路径后与工作区根目录做
前缀比对，阻止任何"逃出"工作区的文件访问请求。
一旦检测到路径越界，便抛出 :class:`PathPolicyError`，由上层策略层统一处理。
"""

from __future__ import annotations

from pathlib import Path


class PathPolicyError(ValueError):
    """路径越界时抛出的策略错误。

    继承自 :class:`ValueError`，以便调用方可以用同一 except 分支同时捕获
    常规的值错误与策略错误。
    """

    pass


class PathGuard:
    """把文件工具的操作范围限制在当前工作区目录树内。

    构造时传入工作区根目录（``root``），后续每次调用 :meth:`validate` 都会：

    1. 将传入路径转换为 :class:`~pathlib.Path` 对象；
    2. 若为相对路径，则基于 ``root`` 拼接成绝对路径；
    3. 通过 :meth:`~pathlib.Path.resolve` 解析符号链接，得到真实绝对路径；
    4. 调用 :meth:`~pathlib.Path.relative_to` 检查该路径是否仍在 ``root`` 之下；
       若抛出 :class:`ValueError`，说明路径已逃逸出工作区，随即抛出
       :class:`PathPolicyError`。

    这样即便攻击者通过 ``../`` 或符号链接试图越界，也会被拦截。
    """

    def __init__(self, root: str | Path):
        # 将根目录解析为规范化的绝对路径，消除符号链接和多余的斜杠
        self.root = Path(root).resolve()

    def validate(self, value: str | Path) -> Path:
        """校验 ``value`` 所指向的路径是否位于工作区内，并返回解析后的绝对路径。

        Parameters
        ----------
        value:
            待校验的路径，支持字符串或 :class:`~pathlib.Path` 对象，
            可以是相对路径或绝对路径。

        Returns
        -------
        Path
            解析后的规范绝对路径（已确认位于工作区内）。

        Raises
        ------
        PathPolicyError
            当解析后的路径逃逸出工作区根目录时抛出。
        """
        candidate = Path(value)
        # 相对路径以工作区根为基准拼接成绝对路径
        if not candidate.is_absolute():
            candidate = self.root / candidate
        # resolve() 会展开符号链接，防止通过 symlink 逃逸
        resolved = candidate.resolve()
        try:
            # relative_to 只有在 resolved 确实是 root 的子路径时才会成功
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise PathPolicyError(f"path escapes workspace: {value}") from exc
        return resolved

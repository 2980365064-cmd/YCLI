"""LSP 诊断模块。

提供轻量级的代码诊断功能。当前实现通过 Python 标准库的 py_compile
对 .py 文件做语法检查，非 Python 文件直接返回空列表。
后续可扩展为接入完整的 LSP 协议以支持更多语言和诊断类型。
"""

from __future__ import annotations

import py_compile
from pathlib import Path


def diagnose_file(path: str | Path) -> list[str]:
    """对 Python 文件执行语法诊断。

    使用 py_compile 编译指定文件，如果有语法错误则返回错误信息列表。
    非 .py 文件直接返回空列表（无诊断信息）。
    """
    file_path = Path(path)
    if file_path.suffix != ".py":
        return []
    try:
        py_compile.compile(str(file_path), doraise=True)
    except py_compile.PyCompileError as exc:
        return [str(exc)]
    return []

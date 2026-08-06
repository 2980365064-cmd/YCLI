"""本地代码索引模块。

基于 SQLite 为当前项目构建轻量级全文代码索引。
rebuild() 遍历项目中所有文本类型的源文件（按后缀名过滤），
将每个非空行存入 ``code_chunks`` 表；search() 采用多词元 AND 匹配
对索引内容进行关键词搜索。

该索引定位为「够用就好」的本地方案，适合快速定位代码片段，
不依赖外部服务或重型搜索引擎。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".java",
    ".kt",
    ".go",
    ".rs",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".xml",
    ".html",
    ".css",
    ".scss",
    ".sql",
    ".sh",
}

SKIP_DIRS = {".git", ".venv", "node_modules", "dist", "build", "target", "__pycache__"}


@dataclass(slots=True)
class CodeSearchResult:
    """代码搜索结果的数据类。

    Attributes:
        path: 匹配行所在文件的相对路径。
        line: 匹配行的行号（从 1 开始）。
        snippet: 匹配行的文本内容（已去除首尾空白）。
    """

    path: str
    line: int
    snippet: str


class CodeIndex:
    """本地代码全文索引。

    核心接口：
    - rebuild(path): 遍历指定路径下所有文本源文件，将每个非空行存入 SQLite。
      如果传入 None 则重建整个项目索引；传入具体文件/子目录则只重建该范围。
    - search(query, limit): 将查询拆分为多个词元，先在 SQLite 中用第一个词元
      做 LIKE 预筛选（最多 500 行），再在内存中对所有词元做 AND 精确匹配。
    """

    def __init__(self, root: str | Path, db_path: str | Path | None = None):
        self.root = Path(root).resolve()
        self.db_path = (
            Path(db_path).expanduser() if db_path else self.root / ".ycli" / "code_index.sqlite3"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def rebuild(self, path: str | Path | None = None) -> int:
        """重建代码索引。

        遍历指定路径（默认为项目根目录）下的所有文本源文件，
        将每个非空行以 (root, path, line, content) 的形式存入 SQLite。
        重建前会先删除当前 root 下的旧数据。返回插入的总行数。
        """
        base = self._resolve(path or self.root)
        files = [base] if base.is_file() else list(self._iter_files(base))
        with self._connect() as conn:
            # 先清除当前 root 的旧索引数据
            conn.execute("delete from code_chunks where root = ?", (str(self.root),))
            count = 0
            for file_path in files:
                rel = str(file_path.relative_to(self.root))
                try:
                    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                except OSError:
                    continue
                for line_number, line in enumerate(lines, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue  # 跳过空行，节省存储空间
                    conn.execute(
                        """
                        insert into code_chunks(root, path, line, content)
                        values (?, ?, ?, ?)
                        """,
                        (str(self.root), rel, line_number, stripped),
                    )
                    count += 1
            return count

    def search(self, query: str, limit: int = 20) -> list[CodeSearchResult]:
        """在索引中搜索与 query 匹配的代码行。

        采用两步策略：先用第一个词元在 SQLite 中做 LIKE 预筛选（最多 500 行），
        再在内存中要求所有词元都出现在行内容中（AND 语义）。
        """
        terms = [term.lower() for term in query.split() if term.strip()]
        if not terms:
            return []
        rows: list[tuple[str, int, str]]
        with self._connect() as conn:
            like = f"%{terms[0]}%"
            rows = conn.execute(
                """
                select path, line, content
                from code_chunks
                where root = ? and lower(content) like ?
                order by path, line
                limit 500
                """,
                (str(self.root), like),
            ).fetchall()
        results: list[CodeSearchResult] = []
        for path, line, content in rows:
            lowered = content.lower()
            if all(term in lowered for term in terms):
                results.append(CodeSearchResult(path, int(line), content))
            if len(results) >= limit:
                break
        return results

    def _iter_files(self, base: Path):
        """递归遍历目录，产出所有文本类型源文件的路径（跳过 SKIP_DIRS）。"""
        for path in base.rglob("*"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                yield path

    def _resolve(self, value: str | Path) -> Path:
        """解析路径并确保其位于项目根目录之内（防止路径逃逸）。"""
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        resolved = path.resolve()
        # 如果路径不在 root 下，relative_to 会抛出 ValueError
        resolved.relative_to(self.root)
        return resolved

    def _ensure_schema(self) -> None:
        """初始化 code_chunks 表和索引（如不存在）。"""
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists code_chunks (
                    id integer primary key autoincrement,
                    root text not null,
                    path text not null,
                    line integer not null,
                    content text not null
                )
                """
            )
            conn.execute(
                "create index if not exists idx_code_chunks_root_path on code_chunks(root, path)"
            )
            conn.execute(
                "create index if not exists idx_code_chunks_root_content "
                "on code_chunks(root, content)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

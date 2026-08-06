"""SQLite 长期记忆管理模块。

提供基于 SQLite 的轻量级持久化记忆功能，用于在多次会话间保存项目相关的上下文信息。
每条记忆按 scope（通常是项目 cwd）分组，支持保存、列表查询和关键词搜索。
底层存储为单表 ``memories``，通过 ``(scope, id)`` 联合索引加速按 scope 的查询。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(slots=True)
class MemoryEntry:
    """单条记忆条目的数据类。

    Attributes:
        id: 自增主键，全局唯一。
        scope: 记忆所属的作用域（通常为项目根目录路径）。
        content: 记忆文本内容。
        created_at: 创建时间，ISO 8601 格式字符串。
    """

    id: int
    scope: str
    content: str
    created_at: str


class MemoryManager:
    """SQLite 长期记忆管理器。

    通过 save / list / search 三个核心接口管理记忆条目：
    - save(content): 将文本内容存储为一条新记忆，返回新条目的 id。
    - list(limit): 按创建时间倒序返回最近 N 条记忆。
    - search(query, limit): 基于关键词的简单全文搜索，
      将查询拆分为多个词元，要求所有词元都在内容中出现（AND 语义）。
    所有操作都限定在当前 scope 内，不同项目之间互不干扰。
    """

    def __init__(self, db_path: str | Path, scope: str):
        self.db_path = Path(db_path).expanduser()
        self.scope = scope
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def save(self, content: str) -> int:
        """保存一条新的记忆条目，返回其自增 id。"""
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "insert into memories(scope, content, created_at) values (?, ?, ?)",
                (self.scope, content.strip(), created_at),
            )
            return int(cursor.lastrowid)

    def list(self, limit: int = 20) -> list[MemoryEntry]:
        """按 id 倒序返回当前 scope 下最近的 ``limit`` 条记忆。"""
        with self._connect() as conn:
            rows = conn.execute(
                """
                select id, scope, content, created_at
                from memories
                where scope = ?
                order by id desc
                limit ?
                """,
                (self.scope, limit),
            ).fetchall()
        return [MemoryEntry(*row) for row in rows]

    def search(self, query: str, limit: int = 10) -> list[MemoryEntry]:
        """基于关键词的简单记忆搜索。

        将 query 拆分为多个词元（按空白分词），然后对最近 200 条记忆做
        内存中的全词元 AND 匹配。如果 query 为空则退化为 list()。
        """
        terms = [term for term in query.lower().split() if term]
        if not terms:
            return self.list(limit)
        with self._connect() as conn:
            # 先拉取最近 200 条到内存中，再做多词元匹配，避免在 SQL 中拼多个 LIKE
            rows = conn.execute(
                """
                select id, scope, content, created_at
                from memories
                where scope = ?
                order by id desc
                limit 200
                """,
                (self.scope,),
            ).fetchall()
        matches = []
        for row in rows:
            content = str(row[2]).lower()
            # 要求所有词元都出现在内容中（AND 语义）
            if all(term in content for term in terms):
                matches.append(MemoryEntry(*row))
            if len(matches) >= limit:
                break
        return matches

    def clear(self) -> int:
        """清空当前 scope 下的所有记忆，返回被删除的条目数。"""
        with self._connect() as conn:
            cursor = conn.execute("delete from memories where scope = ?", (self.scope,))
            return int(cursor.rowcount)

    def _ensure_schema(self) -> None:
        """初始化数据库表和索引（如不存在）。"""
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists memories (
                    id integer primary key autoincrement,
                    scope text not null,
                    content text not null,
                    created_at text not null
                )
                """
            )
            conn.execute("create index if not exists idx_memories_scope on memories(scope, id)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

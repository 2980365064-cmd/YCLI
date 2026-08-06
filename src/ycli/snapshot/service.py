"""项目快照服务模块。

为 Agent 的工具调用回合（turn）提供项目级别的快照和恢复能力。
每次快照会将整个项目目录（跳过 .git / node_modules / .venv 等）完整复制到
``~/.ycli/snapshots/<sha256-of-cwd>/`` 下，并在 ``index.jsonl`` 中追加一条索引记录。

典型流程：
- Agent 开始一个 turn 前调用 create("pre-turn") 保存快照。
- turn 结束后调用 create("post-turn") 保存结束状态。
- 需要回滚时调用 restore(ref)，会先自动创建 "pre-restore" 快照，再用目标快照覆盖项目目录。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "node_modules", "dist", "build", "target", "__pycache__"}


@dataclass(slots=True)
class SnapshotRecord:
    """单条快照记录的数据类。

    Attributes:
        id: 快照唯一标识，格式为 ``{phase}_{YYYYMMDDHHMMSSffffff}``。
        phase: 快照阶段标记（如 "pre-turn"、"post-turn"、"pre-restore"）。
        created_at: 快照创建时间，ISO 8601 格式。
        path: 快照在磁盘上的存储路径。
    """

    id: str
    phase: str
    created_at: str
    path: Path


class SnapshotService:
    """项目快照服务。

    核心流程：
    - create(phase): 将当前项目目录复制到快照存储区，并在 JSONL 索引文件中追加一条记录。
    - list(limit): 从 JSONL 索引文件中读取最近的快照记录列表。
    - restore(ref): 通过快照编号或索引恢复项目状态。
      恢复前会自动创建一次 "pre-restore" 快照以保护当前状态。
    - clean(): 清除所有快照数据和索引。

    索引文件 ``index.jsonl`` 采用追加写入的 JSON Lines 格式，
    每行是一条快照的元信息（id / phase / created_at / path）。
    """

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        # 用项目根目录路径的 SHA-256 前 16 位作为存储子目录名，确保不同项目隔离
        digest = hashlib.sha256(str(self.project_root).encode("utf-8")).hexdigest()[:16]
        self.root = Path.home() / ".ycli" / "snapshots" / digest
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"

    def create(self, phase: str) -> SnapshotRecord:
        """创建一个新快照，将项目目录完整复制到快照存储区。

        Args:
            phase: 快照阶段标记，如 "pre-turn"、"post-turn"。

        Returns:
            本次快照的记录对象。
        """
        snapshot_id = f"{phase}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
        target = self.root / snapshot_id
        target.mkdir(parents=True, exist_ok=True)
        self._copy_tree(self.project_root, target)
        record = SnapshotRecord(
            id=snapshot_id,
            phase=phase,
            created_at=datetime.now(UTC).isoformat(),
            path=target,
        )
        # 以追加模式写入 JSONL 索引，每次一条 JSON 记录
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "id": record.id,
                        "phase": record.phase,
                        "created_at": record.created_at,
                        "path": str(record.path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return record

    def list(self, limit: int = 20) -> list[SnapshotRecord]:
        """从 JSONL 索引文件中读取最近的快照记录列表（按时间倒序）。"""
        if not self.index_path.exists():
            return []
        records = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(
                SnapshotRecord(
                    id=item["id"],
                    phase=item["phase"],
                    created_at=item["created_at"],
                    path=Path(item["path"]),
                )
            )
        return records[-limit:][::-1]

    def restore(self, snapshot_ref: str) -> SnapshotRecord:
        """根据快照编号或序号恢复项目目录。

        支持两种引用方式：纯数字序号（从 1 开始）或完整的快照 id。
        恢复前会先自动创建一次 "pre-restore" 快照以保护当前状态。

        Raises:
            ValueError: 找不到指定快照时抛出。
        """
        records = self.list(limit=200)
        record = None
        if snapshot_ref.isdigit():
            index = int(snapshot_ref) - 1
            if 0 <= index < len(records):
                record = records[index]
        else:
            record = next((item for item in records if item.id == snapshot_ref), None)
        if not record:
            raise ValueError(f"snapshot not found: {snapshot_ref}")
        self.create("pre-restore")
        self._restore_tree(record.path, self.project_root)
        return record

    def clean(self) -> int:
        """清除所有快照数据和索引文件，返回被清除的快照数量。"""
        count = len(self.list(limit=10_000))
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        return count

    def _copy_tree(self, source: Path, target: Path) -> None:
        """递归复制目录树，跳过 SKIP_DIRS 中的目录。"""
        for item in source.iterdir():
            if _skip(item):
                continue
            destination = target / item.name
            if item.is_dir():
                shutil.copytree(item, destination, ignore=_ignore)
            elif item.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, destination)

    def _restore_tree(self, source: Path, target: Path) -> None:
        """用快照内容覆盖目标目录（保留 SKIP_DIRS 中的目录不动）。"""
        for item in target.iterdir():
            if _skip(item):
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        self._copy_tree(source, target)


def _skip(path: Path) -> bool:
    """判断文件或目录是否应被跳过（名称在 SKIP_DIRS 集合中）。"""
    return path.name in SKIP_DIRS


def _ignore(_directory: str, names: list[str]) -> set[str]:
    """shutil.copytree 的 ignore 回调，用于过滤 SKIP_DIRS 中的子目录。"""
    return {name for name in names if name in SKIP_DIRS}

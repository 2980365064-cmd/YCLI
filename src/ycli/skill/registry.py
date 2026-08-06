"""技能注册表模块。

管理 YCLI 的 Skill 系统：从三个来源加载 SKILL.md 文件，
提供技能的查询、启用/禁用、索引文本生成等功能。

SKILL.md 加载流程：
1. 按优先级遍历三个来源目录：builtin（内置）→ user（~/.ycli/skills/）→ project（.ycli/skills/）。
2. 在每个来源下扫描 ``*/SKILL.md`` 文件，解析 YAML frontmatter 获取元数据。
3. 同名技能高优先级来源覆盖低优先级来源。
4. 通过 SkillStateStore 持久化启用/禁用状态。
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Skill:
    """单个技能的数据表示。

    Attributes:
        name: 技能名称（取自 frontmatter 的 name 字段或所在目录名）。
        description: 技能描述，用于注入 system prompt 供 LLM 决策是否加载。
        path: SKILL.md 文件在磁盘上的路径。
        content: SKILL.md 的完整内容（含 frontmatter）。
        source: 来源类型，"builtin" / "user" / "project"。
        version: 技能版本号。
        tags: 技能标签列表。
        enabled: 是否启用。
    """

    name: str
    description: str
    path: Path
    content: str
    source: str = "project"
    version: str = ""
    tags: list[str] = field(default_factory=list)
    enabled: bool = True

    @property
    def body(self) -> str:
        """返回去除 frontmatter 后的技能正文内容。"""
        return _strip_frontmatter(self.content).strip()


class SkillContextBuffer:
    """技能上下文缓冲区。

    当 LLM 调用 load_skill 工具时，技能内容被推入此缓冲区，
    在下一次用户消息发送前通过 drain() 一次性取出并拼接到消息中。
    采用 OrderedDict 实现 LRU 语义，超过容量限制时自动淘汰最早的条目。
    """

    def __init__(self, limit: int = 3):
        self.limit = limit
        self._items: OrderedDict[str, str] = OrderedDict()

    def push(self, name: str | None, body: str | None) -> None:
        """推入一个技能内容。如果同名技能已存在则更新（移到队尾），超过 limit 时淘汰最早的。"""
        if not name or not body:
            return
        if name in self._items:
            del self._items[name]
        self._items[name] = body
        while len(self._items) > self.limit:
            self._items.popitem(last=False)

    def drain(self) -> str:
        """取出并清空缓冲区中所有技能内容，返回拼接后的文本。"""
        if not self._items:
            return ""
        chunks = [
            f"## Loaded Skill: {name}\n{body.strip()}"
            for name, body in self._items.items()
            if body.strip()
        ]
        self._items.clear()
        return "\n\n".join(chunks)

    def clear(self) -> None:
        self._items.clear()

    def is_empty(self) -> bool:
        return not self._items

    def size(self) -> int:
        return len(self._items)


class SkillStateStore:
    """技能启用/禁用状态的持久化存储。

    通过 JSON 文件（默认 ~/.ycli/skills.json）记录被禁用的技能名称列表。
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or Path.home() / ".ycli" / "skills.json").expanduser()

    def disabled(self) -> set[str]:
        """从磁盘读取被禁用的技能名称集合。"""
        if not self.path.exists():
            return set()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        values = data.get("disabled") if isinstance(data, dict) else None
        if not isinstance(values, list):
            return set()
        return {str(item) for item in values if str(item).strip()}

    def disable(self, name: str) -> None:
        """将指定技能标记为禁用并持久化到磁盘。"""
        values = self.disabled()
        values.add(name)
        self._write(values)

    def enable(self, name: str) -> None:
        """将指定技能标记为启用并持久化到磁盘。"""
        values = self.disabled()
        values.discard(name)
        self._write(values)

    def _write(self, disabled: set[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"disabled": sorted(disabled)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class SkillRegistry:
    """技能注册表，从内置、用户和项目三个来源加载 SKILL.md 技能文件。

    加载优先级：builtin → user → project，同名技能高优先级覆盖低优先级。
    所有技能在首次访问时懒加载并缓存，reload() 可清除缓存触发重新加载。
    """

    def __init__(
        self,
        project_root: str | Path,
        *,
        builtin_root: str | Path | None = None,
        user_root: str | Path | None = None,
        state_store: SkillStateStore | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        package_root = Path(__file__).resolve().parents[1]
        self.builtin_root = Path(builtin_root or package_root / "builtin_skills")
        self.user_root = Path(user_root or Path.home() / ".ycli" / "skills")
        self.project_skill_root = self.project_root / ".ycli" / "skills"
        self.state_store = state_store or SkillStateStore()
        self._skills: dict[str, Skill] | None = None

    def reload(self) -> None:
        """清除缓存，下次访问时重新从磁盘加载所有技能。"""
        self._skills = None

    def list(self) -> list[Skill]:
        """返回所有已启用的技能列表（等同于 enabled_skills()）。"""
        return self.enabled_skills()

    def all_skills(self) -> list[Skill]:
        """返回所有技能（包括已禁用的），按名称排序。"""
        skills = self._load_all()
        return [skills[name] for name in sorted(skills)]

    def enabled_skills(self) -> list[Skill]:
        """返回所有已启用的技能列表。"""
        return [skill for skill in self.all_skills() if skill.enabled]

    def load(self, name: str, *, include_disabled: bool = False) -> Skill | None:
        """按名称加载指定技能。默认只返回已启用的技能。"""
        skill = self._load_all().get(name)
        if not skill:
            return None
        if not include_disabled and not skill.enabled:
            return None
        return skill

    def enable(self, name: str) -> bool:
        """启用指定技能，返回是否成功。"""
        if not self.load(name, include_disabled=True):
            return False
        self.state_store.enable(name)
        self.reload()
        return True

    def disable(self, name: str) -> bool:
        """禁用指定技能，返回是否成功。"""
        if not self.load(name, include_disabled=True):
            return False
        self.state_store.disable(name)
        self.reload()
        return True

    def index_text(self, max_chars: int = 4000, max_skills: int = 20) -> str:
        """生成技能索引文本，注入 system prompt 供 LLM 决策。

        列出所有已启用技能的名称和描述摘要，LLM 可据此决定是否调用 load_skill。
        """
        skills = self.enabled_skills()[:max_skills]
        if not skills:
            return ""
        lines = [
            "Available skills:",
            "Load a skill with load_skill(name) when its description matches the task.",
        ]
        for skill in skills:
            description = " ".join(skill.description.split())
            if len(description) > 500:
                description = description[:497] + "..."
            lines.append(f"- {skill.name}: {description}")
        text = "\n".join(lines)
        return text[:max_chars]

    def _load_all(self) -> dict[str, Skill]:
        """从三个来源加载所有 SKILL.md 文件，结果会被缓存。

        遍历顺序：builtin → user → project。同名技能后加载的会覆盖先加载的，
        因此 project 优先级最高，builtin 最低。
        """
        if self._skills is not None:
            return self._skills
        disabled = self.state_store.disabled()
        skills: dict[str, Skill] = {}
        for source, root in [
            ("builtin", self.builtin_root),
            ("user", self.user_root),
            ("project", self.project_skill_root),
        ]:
            if not root.exists():
                continue
            for skill_file in sorted(root.glob("*/SKILL.md")):
                skill = self._load_skill_file(skill_file, source, disabled)
                if skill:
                    skills[skill.name] = skill
        self._skills = skills
        return skills

    def _load_skill_file(self, path: Path, source: str, disabled: set[str]) -> Skill | None:
        """解析单个 SKILL.md 文件，提取 frontmatter 元数据并构造 Skill 对象。"""
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return None
        metadata = _parse_frontmatter(content)
        name = metadata.get("name") or path.parent.name
        description = metadata.get("description") or ""
        tags = _parse_tags(metadata.get("tags", ""))
        return Skill(
            name=name,
            description=description,
            version=metadata.get("version") or "",
            tags=tags,
            source=source,
            path=path,
            content=content,
            enabled=name not in disabled,
        )


def _parse_frontmatter(content: str) -> dict[str, str]:
    """解析 SKILL.md 文件开头的 YAML frontmatter（--- 包裹的元数据区域）。"""
    if not content.startswith("---"):
        return {}
    match = re.match(r"^---\s*\n(.*?)\n---\s*", content, re.S)
    if not match:
        return {}
    lines = match.group(1).splitlines()
    metadata: dict[str, str] = {}
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        if ":" not in raw_line:
            index += 1
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "|":
            index += 1
            block: list[str] = []
            while index < len(lines) and (lines[index].startswith(" ") or not lines[index].strip()):
                block.append(lines[index].strip())
                index += 1
            metadata[key] = " ".join(part for part in block if part)
            continue
        metadata[key] = value.strip().strip('"').strip("'")
        index += 1
    return metadata


def _strip_frontmatter(content: str) -> str:
    """去除文本开头的 YAML frontmatter 区域，返回纯正文内容。"""
    if not content.startswith("---"):
        return content
    return re.sub(r"^---\s*\n.*?\n---\s*", "", content, count=1, flags=re.S)


def _parse_tags(raw: str) -> list[str]:
    """将 frontmatter 中的 tags 字符串解析为标签列表（支持 [a, b, c] 和 a, b, c 两种格式）。"""
    value = raw.strip()
    if not value:
        return []
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return [item.strip().strip('"').strip("'") for item in value.split(",") if item.strip()]

"""图片处理模块。

支持在用户消息中使用 ``@image:<path_or_url>`` 语法引用图片。
parse_image_references() 会扫描消息文本，将所有 @image: 引用替换为
符合多模态 LLM API 格式的 image_url 内容块（data URL 或远程 URL）。

对于本地图片，会自动进行缩放（最大边 1568px）和格式转换（统一为 JPEG），
以控制 token 消耗。不支持多模态的模型会在上层自动降级（丢弃图片部分）。
"""

from __future__ import annotations

import base64
import io
import mimetypes
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image

# 匹配消息中的 @image:<引用> 语法，引用可以是文件路径或 URL
IMAGE_PATTERN = re.compile(r"@image:([^\s]+)")


def parse_image_references(message: str, cwd: str) -> str | list[dict]:
    """解析消息中的 @image: 引用，将其转换为多模态 API 内容块。

    如果消息中不包含任何 @image: 引用，直接返回原始字符串。
    否则返回一个包含 text 和 image_url 类型内容块的列表，
    可直接作为多模态 LLM API 的 message content 使用。

    Args:
        message: 用户输入的原始消息文本。
        cwd: 当前工作目录，用于解析相对路径的图片引用。
    """
    matches = list(IMAGE_PATTERN.finditer(message))
    if not matches:
        return message

    parts: list[dict] = []
    cursor = 0
    for match in matches:
        before = message[cursor : match.start()]
        if before:
            parts.append({"type": "text", "text": before})
        reference = match.group(1)
        parts.append(_image_part(reference, cwd))
        cursor = match.end()
    tail = message[cursor:]
    if tail:
        parts.append({"type": "text", "text": tail})
    return parts


def _image_part(reference: str, cwd: str) -> dict:
    """将单个 @image: 引用转换为 API 内容块（支持远程 URL 和本地文件）。"""
    if reference.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": reference}}
    path = _resolve_image_path(reference, cwd)
    data_url, width, height = _encode_image(path)
    return {
        "type": "image_url",
        "image_url": {"url": data_url},
        "metadata": {"source": str(path), "width": width, "height": height},
    }


def _resolve_image_path(reference: str, cwd: str) -> Path:
    """解析图片引用路径，支持 file:// URI 和普通路径，确保路径在项目目录内。"""
    if reference.startswith("file://"):
        parsed = urlparse(reference)
        path = Path(unquote(parsed.path))
    else:
        path = Path(reference)
    if not path.is_absolute():
        path = Path(cwd).resolve() / path
    path = path.resolve()
    path.relative_to(Path(cwd).resolve())
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _encode_image(path: Path, max_side: int = 1568) -> tuple[str, int, int]:
    """将本地图片压缩、转换为 base64 data URL。

    - 等比缩放使最大边不超过 max_side（默认 1568px，适配主流多模态模型）。
    - 将 RGBA/LA 模式合成白底 RGB，其他非 RGB 模式直接转换。
    - 输出为 JPEG 格式（quality=85），兼顾质量和体积。

    Returns:
        (data_url, width, height) 三元组。
    """
    with Image.open(path) as image:
        image.thumbnail((max_side, max_side))
        if image.mode in {"RGBA", "LA"}:
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        return f"data:{mime};base64,{encoded}", image.width, image.height

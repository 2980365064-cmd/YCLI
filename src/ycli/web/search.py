"""DuckDuckGo 网页搜索模块。

通过请求 DuckDuckGo 的 HTML 版本实现网页搜索，无需 API Key。
解析返回的 HTML 页面提取搜索结果的标题、URL 和摘要。
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, quote, unquote, urlparse

import httpx


@dataclass(slots=True)
class SearchResult:
    """单条搜索结果的数据类。

    Attributes:
        title: 搜索结果标题。
        url: 搜索结果链接（已从 DuckDuckGo 跳转链接还原为实际 URL）。
        snippet: 搜索结果摘要文本。
    """

    title: str
    url: str
    snippet: str


async def search_web(query: str, max_results: int = 5, timeout: float = 15.0) -> list[SearchResult]:
    """通过 DuckDuckGo HTML 版执行网页搜索。

    构造 DuckDuckGo HTML 搜索 URL，发起异步请求后解析返回的 HTML 提取结果。
    返回最多 max_results 条搜索结果。
    """
    url = f"https://duckduckgo.com/html/?q={quote(query)}"
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url, headers={"user-agent": "YCLI-Python/0.1.0"})
        response.raise_for_status()
    return _parse_duckduckgo(response.text)[:max_results]


def _parse_duckduckgo(raw_html: str) -> list[SearchResult]:
    """从 DuckDuckGo HTML 响应中解析搜索结果列表。"""
    results: list[SearchResult] = []
    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)</a>'
        r"[\s\S]*?"
        r'<a[^>]+class="result__snippet"[^>]*>([\s\S]*?)</a>',
        re.I,
    )
    for match in pattern.finditer(raw_html):
        results.append(
            SearchResult(
                title=_clean(match.group(2)),
                url=_normalize_duckduckgo_url(html.unescape(match.group(1))),
                snippet=_clean(match.group(3)),
            )
        )
    return results


def _clean(value: str) -> str:
    """清理 HTML 片段：去除标签、反转义实体、合并空白。"""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def _normalize_duckduckgo_url(url: str) -> str:
    """将 DuckDuckGo 的跳转 URL 还原为实际目标 URL（提取 uddg 参数）。"""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    if "uddg" in params and params["uddg"]:
        return unquote(params["uddg"][0])
    return url

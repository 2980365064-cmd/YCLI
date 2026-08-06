"""URL 抓取与 SSRF 防护模块。

提供异步 URL 抓取功能，内置 SSRF（Server-Side Request Forgery）防护。
fetch_url() 在发起 HTTP 请求前会先验证目标 URL 必须为公网地址，
拒绝所有私有 IP、环回地址、链路本地地址和组播地址。
对于域名会先做 DNS 解析，逐一检查解析结果中的每个 IP 地址。
"""

from __future__ import annotations

import html
import ipaddress
import re
import socket
from urllib.parse import urlparse

import httpx


class NetworkPolicyError(ValueError):
    """网络策略违规异常，当 URL 违反 SSRF 防护规则时抛出。"""

    pass


async def fetch_url(url: str, max_length: int = 10_000, timeout: float = 15.0) -> str:
    """抓取指定 URL 的文本内容，内置 SSRF 防护。

    流程：
    1. 调用 _validate_public_url() 进行 SSRF 检查（协议、域名、IP 地址）。
    2. 使用 httpx 异步发起 GET 请求。
    3. 如果响应是 HTML，调用 extract_text_from_html() 提取纯文本。
    4. 超过 max_length 时截断并追加截断标记。

    Raises:
        NetworkPolicyError: URL 违反安全策略时抛出。
    """
    _validate_public_url(url)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url, headers={"user-agent": "YCLI-Python/0.1.0"})
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        text = response.text
        if "html" in content_type:
            text = extract_text_from_html(text)
        if len(text) > max_length:
            text = text[:max_length] + "\n... [truncated]"
        return text or "(empty page)"


def extract_text_from_html(raw_html: str) -> str:
    """从 HTML 中提取纯文本：移除 script/style 标签、去除所有 HTML 标签、反转义 HTML 实体。"""
    text = re.sub(r"<script[\s\S]*?</script>", " ", raw_html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _validate_public_url(url: str) -> None:
    """SSRF 防护核心：验证 URL 指向公网地址。

    检查步骤：
    1. 协议必须是 http 或 https。
    2. 如果主机名本身就是 IP 地址，直接检查是否为私有/环回/链路本地/组播地址。
    3. 如果主机名是域名，先做 DNS 解析，然后逐一检查解析出的每个 IP 地址。
       任何一个 IP 是私有地址都会拒绝，防止通过 DNS rebinding 绕过检查。
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise NetworkPolicyError("only http/https URLs are allowed")
    if not parsed.hostname:
        raise NetworkPolicyError("URL must include a hostname")
    host = parsed.hostname
    try:
        ip = ipaddress.ip_address(host)
        _reject_private_ip(ip)
        return
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise NetworkPolicyError(f"cannot resolve host: {host}") from exc
    for info in infos:
        address = info[4][0]
        try:
            _reject_private_ip(ipaddress.ip_address(address))
        except ValueError:
            continue


def _reject_private_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    """拒绝私有/环回/链路本地/组播 IP 地址。

    这些地址类型包括：
    - is_private: RFC 1918 私有地址（如 10.x.x.x, 192.168.x.x）
    - is_loopback: 环回地址（127.0.0.1, ::1）
    - is_link_local: 链路本地地址（169.254.x.x, fe80::）
    - is_multicast: 组播地址（224.x.x.x, ff00::）
    """
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
        raise NetworkPolicyError("URL resolves to a private or local address")

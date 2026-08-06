"""YCLI CLI 入口 — 基于 Typer 的命令行应用。

主命令：ycli（进入 REPL 或执行单次 prompt）
子命令：
- doctor: 环境诊断
- serve: 启动 Runtime API HTTP 服务器
- mcp serve: 启动 MCP 服务器
- mcp init-chrome: 生成 Chrome DevTools MCP 配置
- mcp list: 列出已配置的 MCP 服务器
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from ycli import __version__
from ycli.agent import QueryEngine
from ycli.bootstrap import build_tool_registry
from ycli.config import get_config_paths, load_config
from ycli.entrypoints.repl import start_repl
from ycli.llm import create_llm_client
from ycli.mcp import load_mcp_server_specs, serve_http, serve_stdio, write_chrome_devtools_config
from ycli.runtime import RuntimeApiServer
from ycli.runtime.api import runtime_api_key

app = typer.Typer(
    name="ycli",
    help="YCLI — Terminal AI Agent in Python",
    invoke_without_command=True,
    no_args_is_help=False,
)
mcp_app = typer.Typer(help="MCP server management")
app.add_typer(mcp_app, name="mcp")
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"ycli {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    prompt: Annotated[
        str | None,
        typer.Option("-p", "--prompt", help="Print mode: single prompt, non-interactive"),
    ] = None,
    model: Annotated[str | None, typer.Option("-m", "--model", help="Override model name")] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Override LLM provider"),
    ] = None,
    plain: Annotated[bool, typer.Option("--plain", help="Use plain text rendering")] = False,
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version"),
    ] = False,
) -> None:
    _ = version
    if ctx.invoked_subcommand is not None:
        return
    root = (cwd or Path.cwd()).resolve()
    overrides: dict = {}
    if provider or model or plain:
        overrides = {
            "llm": {"provider": provider, "model": model},
            "render_mode": "plain" if plain else None,
        }
    config = load_config(project_root=root, overrides=overrides)
    if plain:
        config.render_mode = "plain"
    if prompt is not None:
        asyncio.run(_run_prompt(prompt, str(root), config))
    else:
        asyncio.run(start_repl(str(root), config))


@app.command("doctor")
def doctor(
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
) -> None:
    root = (cwd or Path.cwd()).resolve()
    config = load_config(project_root=root)
    checks = {
        "python": sys.version.split()[0],
        "uv": shutil.which("uv") or "missing",
        "node": _version_of("node"),
        "npx": shutil.which("npx") or "missing",
        "rg": shutil.which("rg") or "missing",
        "api_key": "configured" if config.llm.api_key else "missing",
        "provider": config.llm.provider,
        "model": config.llm.model,
        "cwd": str(root),
        "config_paths": [str(path) for path in get_config_paths(root)],
    }
    console.print_json(json.dumps(checks, ensure_ascii=False))


@app.command("serve")
def runtime_serve(
    http: Annotated[bool, typer.Option("--http", help="Serve Runtime API over HTTP")] = True,
    port: Annotated[int, typer.Option("--port", help="HTTP port")] = 8080,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="Runtime API key. Defaults to YCLI_RUNTIME_API_KEY."),
    ] = None,
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
) -> None:
    _ = http
    root = (cwd or Path.cwd()).resolve()
    config = load_config(project_root=root)
    try:
        key = runtime_api_key(api_key)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    RuntimeApiServer(cwd=str(root), config=config, api_key=key, port=port).serve_forever()


@mcp_app.command("serve")
def mcp_serve(
    transport: Annotated[
        str,
        typer.Option("--transport", help="Transport type: stdio or http"),
    ] = "stdio",
    port: Annotated[int, typer.Option("--port", help="HTTP port")] = 3000,
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
) -> None:
    root = str((cwd or Path.cwd()).resolve())
    if transport == "http":
        serve_http(port=port, cwd=root)
    elif transport == "stdio":
        asyncio.run(serve_stdio(cwd=root))
    else:
        raise typer.BadParameter("transport must be stdio or http")


@mcp_app.command("init-chrome")
def mcp_init_chrome(
    scope: Annotated[
        str,
        typer.Option("--scope", help="Config scope: user or project"),
    ] = "project",
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
    browser_url: Annotated[
        str | None,
        typer.Option("--browser-url", help="Connect to an existing Chrome remote debugging URL"),
    ] = None,
    headless: Annotated[bool, typer.Option("--headless", help="Start Chrome headless")] = False,
    slim: Annotated[bool, typer.Option("--slim", help="Use Chrome DevTools slim mode")] = False,
) -> None:
    if scope not in {"user", "project"}:
        raise typer.BadParameter("scope must be user or project")
    root = None if scope == "user" else (cwd or Path.cwd()).resolve()
    path = write_chrome_devtools_config(
        scope_root=root,
        browser_url=browser_url,
        headless=headless,
        slim=slim,
    )
    typer.echo(f"Wrote Chrome DevTools MCP config to {path}")


@mcp_app.command("list")
def mcp_list(
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
) -> None:
    root = (cwd or Path.cwd()).resolve()
    specs = load_mcp_server_specs(root)
    if not specs:
        typer.echo("No MCP servers configured.")
        return
    for spec in specs.values():
        target = spec.url or f"{spec.command} {' '.join(spec.args)}".strip()
        typer.echo(f"{spec.name}\t{spec.type}\t{target}")


async def _run_prompt(prompt: str, cwd: str, config) -> None:
    config.render_mode = "plain"
    if not config.llm.api_key:
        typer.echo(
            "Fatal error: YCLI_API_KEY is not configured. Set it in env, "
            "~/.ycli/config.json, or project .ycli/config.json.",
            err=True,
        )
        raise typer.Exit(1)
    registry, manager = await build_tool_registry(config=config, cwd=cwd)
    if manager and manager.last_errors:
        for name, error in manager.last_errors.items():
            typer.echo(f"MCP server {name} failed to load: {error}", err=True)
    engine = QueryEngine(
        llm_client=create_llm_client(config.llm),
        tool_registry=registry,
        config=config,
        cwd=cwd,
    )
    try:
        result = await engine.ask_complete_async(prompt)
    except Exception as exc:  # noqa: BLE001 - CLI should report model/config errors cleanly
        typer.echo(f"Fatal error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(result.text)


def _version_of(command: str) -> str:
    if not shutil.which(command):
        return "missing"
    try:
        result = subprocess.run(
            [command, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:  # noqa: BLE001
        return "unknown"
    return (result.stdout or result.stderr).strip() or "unknown"

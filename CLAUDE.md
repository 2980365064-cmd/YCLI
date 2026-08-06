# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install / sync
uv sync --extra dev

# Run CLI
uv run ycli                    # interactive REPL
uv run ycli -p "your prompt"   # single-shot
uv run ycli doctor --cwd .     # env check
uv run ycli --plain -p hello   # plain render

# Lint & format
uv run python -m ruff check .
uv run python -m ruff format --check .

# Test
uv run python -m pytest                  # all tests
uv run python -m pytest tests/test_tools.py          # one file
uv run python -m pytest tests/test_tools.py::test_read_file_inside_workspace  # one test

# Build
uv build
```

## Architecture

**Entry points** → `src/ycli/entrypoints/cli.py` (Typer app, `main` callback dispatches to REPL or single-shot) → `entrypoints/repl.py` (interactive loop with prompt-toolkit + Rich).

**Agent loop** — three execution modes, all built on the same ReAct core (`agent/query.py`):
- `agent/agent.py::Agent` — single-agent ReAct with tool call loop, yields typed event dicts (`text_delta`, `thinking_delta`, `tool_call`, `tool_result`, `usage`, `done`, `error`)
- `agent/plan_execute.py::PlanExecuteAgent` — Planner generates a DAG of `Task`s, executor runs them in dependency batches with `asyncio.gather` for parallel tasks
- `agent/orchestrator.py::AgentOrchestrator` — multi-agent: Planner → Worker pool (queue-based dispatch) → Reviewer per step, with retry loop (`max_retries_per_step=2`)

`agent/query_engine.py::QueryEngine` is the high-level facade that wraps all three modes; also exposed via `sdk.py::create_default_engine`.

**Tools** — `tools/base.py::Tool` dataclass (name, params JSON schema, async handler, danger_level, is_read_only, requires_approval). `tools/registry.py::ToolRegistry` holds them; `tools/executor.py::ToolExecutor` splits calls into concurrent (read-only + concurrency-safe) vs sequential, with HITL approval and audit logging per call. Builtin tool handlers live in `tools/builtins.py`. MCP tools are registered as `mcp__<server>__<tool>`.

**LLM** — single `LlmClient` protocol (`llm/base.py`), one implementation `OpenAICompatibleClient` (`llm/openai_compatible.py`) that streams SSE and emits the same event types the agent loop consumes. `llm/factory.py::create_llm_client` maps provider name → base URL (DeepSeek, GLM/Zhipu, Kimi/Moonshot, Step, OpenAI-compatible).

**Config layering** (`config.py::load_config`): defaults → `~/.ycli/config.json` → `.ycli/config.json` (project) → `.env` (project) → overrides dict → env vars. Provider-specific keys (`DEEPSEEK_API_KEY`, `GLM_API_KEY`, etc.) are resolved when `YCLI_API_KEY` is unset. The config is a nested dataclass tree (`LlmConfig`, `ToolsConfig`, `McpConfig`, `MemoryConfig`, `PolicyConfig`, `PromptConfig`, `FeatureConfig`).

**Prompt assembly** — `prompt/assembler.py::PromptAssembler` builds the system prompt from config, cwd, tool names, model, and provider. Skills index text and AGENTS.md files are injected here.

**Skill system** (`skill/registry.py::SkillRegistry`) — loads `SKILL.md` files with YAML frontmatter from three sources in priority order: builtin (`builtin_skills/`), user (`~/.ycli/skills/`), project (`.ycli/skills/`). Skills can be enabled/disabled via `SkillStateStore` (`~/.ycli/skills.json`). The `load_skill` tool drains a `SkillContextBuffer` that gets prepended to the next user message.

**Memory** (`memory/manager.py::MemoryManager`) — SQLite at `~/.ycli/memory.db`, scoped by cwd. Keyword search over recent entries. Used by both the `save_memory` tool and `/memory` slash commands.

**Snapshots** (`snapshot/service.py::SnapshotService`) — full project copy (minus `.git`/`.venv`/`node_modules`/etc.) to `~/.ycli/snapshots/<sha256-of-cwd>/`. Created `pre-turn` and `post-turn` by the Agent. Index is JSONL. Restore creates a `pre-restore` snapshot first.

**Policy** (`policy/`) — `PathGuard` blocks writes outside cwd; `CommandGuard` blacklists dangerous shell patterns; `AuditLog` appends JSONL to `~/.ycli/audit.jsonl`. HITL modes: `never` (auto-approve all), `auto` (approve unless `requires_approval`), `always` (prompt for everything).

**MCP** (`mcp/`) — client side: `McpClientManager` in `bootstrap.py` loads tools from stdio and HTTP MCP servers; server side: `ycli mcp serve` exposes YCLI's own tools. Chrome DevTools config helper via `ycli mcp init-chrome`.

**Runtime API** (`runtime/`) — HTTP server (`RuntimeApiServer`) exposing threads, turns, events, and durable background tasks (SQLite-backed via `DurableTaskManager`).

**Other modules**: `rag/code_index.py` (local code search index), `web/` (DuckDuckGo search + URL fetch with SSRF guards), `image/processor.py` (resize/compress/convert to data URL, auto-degrade for non-multimodal models), `lsp/diagnostics.py` (file diagnostics after `write_file`), `render/` (RichRenderer for REPL, plain renderer for scripts).

## Key patterns

- All agent modes yield `dict[str, Any]` events — the renderer consumes them, never the agent returning a monolithic result.
- Tool handlers are `async (payload, ToolContext) -> ToolResult`. `ToolContext` carries cwd, config, approval callback, and skill buffer.
- `ToolExecutor` is the single enforcement point for approval + audit; tool handlers themselves do not check policy.
- Snapshots are best-effort (`suppress(Exception)`) — they must never break the agent loop.
- The `query()` function in `agent/query.py` is the shared ReAct core; all three agent modes delegate to it.

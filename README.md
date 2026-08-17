# YCLI

基于多 Agent 编排的终端 AI 编程助手。

## 核心特性

**三种 Agent 执行模式**
- **单 Agent ReAct**：经典的推理-行动循环，流式事件驱动
- **Plan-Execute DAG**：Planner 生成有向无环图，按依赖批次并行执行
- **Multi-Agent 编排**：Planner-Worker-Reviewer 三角协作，支持自纠错重试

**工程化能力**
- Git Side-History 快照系统，不污染用户提交历史
- MCP 协议集成，统一内置工具与远程工具
- 智能并发调度，按工具属性自动分流并发/串行
- 分层配置系统（5 级覆盖）+ 三级 Skill 加载
- 混合记忆架构（短期会话 + 长期 SQLite + 静态文件）
- 策略层安全沙箱（路径守卫 + 命令黑名单 + 审计日志）

## 架构亮点

**流式事件驱动**

所有 Agent 模式统一 yield 事件流（text_delta / tool_call / tool_result / usage / done / error），解耦执行逻辑与渲染层，支持富文本和纯文本两种渲染模式。

**智能并发调度**

ToolExecutor 根据工具的读写属性和安全级别自动决定并发或串行执行。只读且并发安全的工具自动并发，写入或需审批的工具串行，在保证安全的前提下最大化执行效率。

**Git Side-History 快照**

通过独立的 orphan branch 记录 Agent 的每次修改，支持 pre-turn / post-turn / pre-restore 三种相位，不污染用户的 git log，不干扰 rebase / merge 操作。

**MCP 协议统一抽象**

将 MCP 远程工具与内置工具统一为 Tool 抽象，命名格式 `mcp__<server>__<tool>`，对 LLM 透明。支持 stdio 和 HTTP 两种传输协议。

**混合记忆系统**

- **短期记忆**：内存中的会话历史，支持自动截断
- **长期记忆**：SQLite 持久化，按项目隔离，支持关键词搜索
- **静态记忆**：YAI.md 文件，可 commit 到 git，团队共享

## 快速开始

```bash
# 克隆项目
git clone https://github.com/itwanger/YCLI-Python.git
cd YCLI-Python

# 安装依赖
uv sync --extra dev

# 启动交互模式
uv run ycli

# 单次查询
uv run ycli -p "帮我总结这个项目"

# 环境检查
uv run ycli doctor --cwd .
```

## 配置

YCLI 采用五级配置覆盖机制：

1. 内置默认配置
2. 用户配置 `~/.ycli/config.json`
3. 项目配置 `.ycli/config.json`
4. 环境变量文件 `.env`
5. CLI 参数和环境变量

### API Key 配置

在项目根目录创建 `.env` 文件：

```bash
# 方式一：Provider-specific Key
YCLI_PROVIDER=deepseek
YCLI_MODEL=deepseek-v4-flash
DEEPSEEK_API_KEY=your_key_here

# 方式二：通用 Key
YCLI_API_KEY=your_key_here
```

支持的 Provider-specific API Key：
- `DEEPSEEK_API_KEY`
- `GLM_API_KEY`
- `STEP_API_KEY`
- `KIMI_API_KEY`

### 连接本地模型

```bash
YCLI_PROVIDER=openai-compatible \
YCLI_BASE_URL=http://127.0.0.1:11434/v1 \
YCLI_MODEL=qwen2.5-coder \
uv run ycli -p "解释这个仓库"
```

## 使用方式

### 交互模式

```bash
uv run ycli
```

进入 REPL 后，可用命令：

```text
/help                    # 显示帮助
/exit                    # 退出
/clear                   # 清空历史
/context                 # 查看上下文信息

# 记忆管理
/memory                  # 列出记忆
/memory search <query>   # 搜索记忆
/memory clear            # 清空记忆
/save <fact>             # 保存记忆

# Agent 模式
/plan <task>             # Plan-Execute 模式
/team <task>             # Multi-Agent 编排模式

# 工具与策略
/tools                   # 列出工具
/hitl on|off|always|auto|never  # HITL 模式
/policy                  # 查看策略

# 快照与恢复
/snapshot                # 列出快照
/restore <id>            # 恢复快照
/snapshot clean          # 清理快照

# 其他
/config                  # 查看配置
/model                   # 查看/切换模型
/skill                   # Skill 管理
/mcp                     # MCP 管理
/task                    # 后台任务管理
```

### 单次查询

```bash
uv run ycli -p "帮我分析这个项目的架构"
```

### SDK 调用

```python
from ycli.sdk import create_default_engine

engine = create_default_engine(cwd=".")

# 单 Agent 模式
result = engine.ask_complete("解释这个项目")
print(result.text)

# Plan-Execute 模式
plan_result = engine.plan_complete("先读取 README，再总结项目结构")

# Multi-Agent 模式
team_result = engine.team_complete("让多个 Agent 并行检查核心模块")
```

## 内置工具

YCLI 内置 14+ 个本地工具和联网工具：

**文件操作**
- `read_file` - 读取文件
- `write_file` - 写入文件
- `list_dir` - 列出目录

**搜索**
- `glob` / `glob_files` - 通配符搜索文件
- `grep` / `grep_code` - 正则搜索内容
- `search_code` - 语义搜索代码

**执行**
- `bash` / `execute_command` - 执行 Shell 命令

**网络**
- `web_search` - DuckDuckGo 搜索
- `web_fetch` - 抓取网页内容

**系统**
- `save_memory` - 保存长期记忆
- `load_skill` - 加载 Skill
- `revert_turn` - 恢复快照

所有写入操作都经过策略层安全检查，支持 HITL 人工审批。

## MCP 集成

### 连接 MCP Server

YCLI 可以连接 MCP Server，自动注册远程工具：

```bash
# 初始化 Chrome DevTools MCP
uv run ycli mcp init-chrome --scope project

# 查看已配置的 MCP Server
uv run ycli mcp list
```

MCP 工具会自动命名为 `mcp__<server>__<tool>` 格式。

### 作为 MCP Server

YCLI 自身也可以作为 MCP Server 暴露内置工具：

```bash
# stdio 模式
uv run ycli mcp serve --transport stdio

# HTTP 模式
uv run ycli mcp serve --transport http --port 3000
```

## Runtime API

YCLI 内置轻量级 HTTP API，适合外部系统接入：

```bash
# 启动服务
YCLI_RUNTIME_API_KEY=dev-key uv run ycli serve --http --port 8080

# 创建线程
curl -X POST http://127.0.0.1:8080/v1/threads \
  -H 'x-api-key: dev-key'

# 发送消息
curl -X POST http://127.0.0.1:8080/v1/threads/<thread_id>/turns \
  -H 'x-api-key: dev-key' \
  -H 'content-type: application/json' \
  -d '{"message":"总结这个项目"}'

# 创建后台任务
curl -X POST http://127.0.0.1:8080/v1/tasks \
  -H 'x-api-key: dev-key' \
  -H 'content-type: application/json' \
  -d '{"message":"后台分析代码"}'
```

## 图片输入

支持在 prompt 中引用图片：

```text
分析这张截图 @image:./screenshots/page.png
```

支持本地图片、绝对路径和远程 URL。本地图片会自动压缩和缩放，如果模型不支持多模态输入会自动降级为文本元信息。

## 环境要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- 可选：`rg`（更快的本地搜索）
- 可选：Node.js 20.19.0+（Chrome DevTools MCP）

## 开发

```bash
# 安装开发依赖
uv sync --extra dev

# 代码检查
uv run python -m ruff check .
uv run python -m ruff format --check .

# 运行测试
uv run python -m pytest

# 构建
uv build
```

## 技术栈

- **Python 3.11+**
- **asyncio** - 异步并发
- **SQLite** - 持久化存储
- **prompt-toolkit** - 交互式终端
- **Rich** - 富文本渲染
- **MCP Protocol** - 工具协议
- **OpenAI API** - LLM 调用

## License

MIT

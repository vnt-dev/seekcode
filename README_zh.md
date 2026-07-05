# SeekCode

SeekCode 是一个基于 Tauri、React 和 Rust 构建的桌面端编程助手。它提供面向工作区的 AI 对话、流式推理与工具调用展示、持久化会话、模型供应商配置，以及适合代码任务的本地命令执行能力。

English documentation is available in [README.md](README.md).

## 功能特性

- 工作区与会话管理，数据通过 SQLite 持久化保存。
- 流式展示助手回复、推理内容和工具调用进度。
- 默认支持 DeepSeek，也可以配置其他 OpenAI-compatible 模型供应商。
- 每个会话可单独选择模型、开启/关闭 thinking mode，并设置 reasoning effort。
- 统计上下文窗口使用量，并在长会话中自动进行上下文压缩。
- 会话看板展示模型调用次数、token 用量、缓存命中和耗时统计。
- 内置 `run_command` 工具，可在当前工作区执行非交互式命令。
- 用户设置保存在本机用户目录下，便于长期使用。

## 技术栈

- 前端：React 19、Vite、Tauri JavaScript APIs、lucide-react。
- 桌面端：Tauri 2。
- 后端：Rust workspace，基于 Tokio 的异步服务。
- 存储：SQLite 与 `sqlx`。
- 模型客户端：DeepSeek/OpenAI-compatible chat completions API。

## 环境要求

- Node.js 和 npm。
- Rust stable 工具链。
- Tauri 2 CLI：

```bash
cargo install tauri-cli --version "^2" --locked
```

如果本机缺少原生构建工具链或 WebView 运行时，请按 Tauri 官方文档完成对应操作系统的环境配置。

## 快速开始

安装 JavaScript 依赖：

```bash
npm ci
```

以开发模式运行桌面应用：

```bash
cargo tauri dev
```

只启动 Vite 前端：

```bash
npm run dev
```

开发环境前端地址为 `http://127.0.0.1:1420`。

## 配置

可以在应用内设置页配置默认供应商和额外供应商。配置文件保存位置：

- Windows：`%USERPROFILE%\.seekcode\config.toml`
- macOS/Linux：`$HOME/.seekcode/config.toml`

默认配置使用 DeepSeek：

```toml
base_url = "https://api.deepseek.com"
api_key = ""
title_model = "deepseek-v4-flash"
context_window = "1M"

[[models]]
id = "deepseek-v4-pro"
label = "DeepSeek V4 Pro"

[[models]]
id = "deepseek-v4-flash"
label = "DeepSeek V4 Flash"
```

额外供应商需要提供 OpenAI-compatible 接口：

- `GET /models`：用于获取模型列表。
- `POST /chat/completions`：用于聊天补全。

会话数据保存位置：

- Windows：`%USERPROFILE%\.seekcode\seekcode.sqlite`
- macOS/Linux：`$HOME/.seekcode/seekcode.sqlite`

## 开发命令

```bash
# 前端测试
npm test

# Rust 测试
cargo test --workspace

# 构建前端资源
npm run build

# 构建桌面端安装包
cargo tauri build
```

Windows 发布流程会使用下面的命令构建 NSIS 和 MSI 安装包：

```bash
cargo tauri build --bundles nsis,msi
```

## 项目结构

```text
.
|-- src/                         # React UI
|   |-- components/              # UI 组件
|   |-- lib/                     # 前端状态、消息与格式化辅助逻辑
|   `-- styles/                  # 按 UI 区域拆分的样式文件
|-- src-tauri/                   # Tauri 应用、命令、配置与日志
|-- crates/
|   |-- agent-core/              # Agent 任务生命周期与运行器
|   |-- app-kernel/              # 应用服务编排层
|   |-- common/                  # 共享 ID、错误、DTO 与 telemetry
|   |-- deepseek-client/         # DeepSeek/OpenAI-compatible 模型客户端
|   |-- storage/                 # SQLite 存储层与 migrations
|   `-- tool-system/             # 工具注册表与内置系统工具
|-- package.json                 # 前端脚本与依赖
`-- Cargo.toml                   # Rust workspace manifest
```

## 本地工具安全说明

SeekCode 会向模型暴露 `run_command` 工具。命令会通过当前平台 shell 执行，并默认以所选工作区作为工作目录。对于可能修改、删除或移动文件的任务，请仔细检查提示词和模型行为，避免影响重要文件。

## 测试

项目同时包含前端单元测试和 Rust crate 测试：

```bash
npm test
cargo test --workspace
```

建议在提交 PR 或发布本地构建前运行以上测试。

## License

SeekCode 使用 MIT License。详见 [LICENSE](LICENSE)。

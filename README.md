# KnotLiEdge（阶段一：知识底座 MVP）

目标：把本地 PDF 文献加工为 **Markdown Vault + ChromaDB 向量索引**，并通过本地 MCP Server 给 Cursor 提供「检索 → 取更长上下文 / 定位文件」的两段式工具。

## 目录结构（仓库内）

- `configs/`：配置（默认 [`configs/default.yaml`](configs/default.yaml)；可复制 [`configs/default.example.yaml`](configs/default.example.yaml) 作为模板）
- `data/01_raw_pdf/`：原始 PDF（仅 `.gitkeep` 占位；正文不进 Git）
- `data/02_markdown_vault/`：解析后的 Markdown（同上）
- `data/02_markdown_vault/assets/`：解析附件（同上）
- `data/03_chroma_db/`：ChromaDB 持久化目录（同上）
- `output/`：脚本与工作流产物（同上）
- `sandbox/`：沙盒配置（`sandbox/configs/` 可跟踪；`sandbox/data/` 忽略）
- `scripts/`：脚本入口（`python -m scripts.xxx`）
- `src/`：核心代码（绝对导入：`from knotliedge...`）
- `templates/`：LLM 模板
- `.knotliedge/`：运行时工作目录（MinerU 日志等；忽略）

内部协作文档（`docs/`、`PROJECT_STATUS.md`、`.cursorrules`）默认不参与本公开快照；说明以本 README 为准。

## 环境准备（Windows）

约定 Conda 环境 **`agent`**（可选用 venv）：

```bash
conda run -n agent python -m pip install -r requirements.txt
conda run -n agent python -m pip install -e .
```

### 手动安装：MinerU

- 本项目通过 `mineru` 命令调用 MinerU；验证：`mineru --help`。

### Torch / 本地 Embedding（BGE-M3）

- 自行安装与机器匹配的 **PyTorch**（CPU/GPU）。
- 将模型放到配置中的目录（默认相对路径 `models/bge-m3`，见 `configs/default.yaml`）。
- 模型下载建议走 **ModelScope**（例如 `BAAI/bge-m3`）；勿依赖 HuggingFace 直连下载。

### 编辑配置（必做）

1. **`embedding.model_name_or_path`**：指向本机模型目录。
2. **`embedding.device`**：`cuda` 或 `cpu`。
3. **`openalex.mailto`**：填入你的联系邮箱（OpenAlex Polite Pool 必填）。

可选：复制 `.env.example` 为 `.env`，填写 `DASHSCOPE_API_KEY` / `OPENAI_API_KEY`（用于兼容 Chat Completions 的关键词解析等；见脚本说明）。

## Chroma（HTTP，必须先于入库）

本项目的向量访问通过 **`chromadb.HttpClient`** 连接**独立 Chroma 进程**；`paths.chroma_db_dir` 由 **`chroma run --path`** 使用，须与 YAML 一致。

在一个终端启动（端口与 `chroma.http_port` 一致，默认 `37651`）：

```bash
conda run -n agent python -m scripts.run_chroma_http_server --config configs/default.yaml
```

若未安装 CLI，可等价执行：`chroma run --path "<yaml 中的 chroma_db_dir 绝对路径>" --port 37651`。

## 使用流程（MVP）

### 1) 放入 PDF

将 PDF 放到 `data/01_raw_pdf/`。

### 2) PDF → Markdown（MinerU）

```bash
conda run -n agent python -m scripts.pdf_to_md --config configs/default.yaml
```

### 3) Markdown → ChromaDB（Chunk + Embedding + 入库）

保持 **Chroma HTTP 服务**已启动，且嵌入 IPC 可用（一般首次入库会自动拉起 `127.0.0.1:60123` 上的嵌入服务）：

```bash
conda run -n agent python -m scripts.index_markdown --config configs/default.yaml
```

增量维护：

```bash
conda run -n agent python -m scripts.index_markdown --config configs/default.yaml --mode incremental
```

从 vault 删除文档后的索引清理：

```bash
conda run -n agent python -m scripts.index_markdown --config configs/default.yaml --mode purge-missing
```

### 4) 启动 MCP Server

```bash
conda run -n agent python -m scripts.run_mcp_server --config configs/default.yaml
```

（可将 `--config` 换为配置文件的**绝对路径**，便于 Cursor 从任意 `cwd` 启动。）

## Embedding IPC（避免多进程重复加载 BGE-M3）

默认由 **`scripts.run_embedding_server`** 提供本地 IPC（`127.0.0.1:60123`），`index_markdown` / MCP 检索侧会自动连接或在需要时拉起。

手动启动（可选）：

```bash
conda run -n agent python -m scripts.run_embedding_server --config configs/default.yaml
```

可选环境变量：`KNOTLIEDGE_EMBED_IDLE_UNLOAD_S`（默认 600）、`KNOTLIEDGE_EMBED_IDLE_CHECK_S`（默认 60）。

## MCP 排障摘要

1. **配置文件**：`--config` 指向的 YAML 必须存在；项目会从配置路径向上查找 `pyproject.toml` 推断 `project_root`，从而解析 `data/...` 相对路径。
2. **Cursor**：`cwd` 建议设为仓库根；启动参数示例：`conda`、`run`、`-n`、`agent`、`python`、`-m`、`scripts.run_mcp_server`、`--config`、`<绝对路径>/configs/default.yaml`。
3. **自检顺序**：改代码或配置后**重启 MCP**。先 **`ping`**（轻量），再 **`stats`**（含 `collection_count` / `chroma_db_dir`）；需要时对 **`stats`** 传入 **`smoke_query`** 做检索冒烟。
4. **未就绪**：若模型路径无效或未入库，`stats` / 检索可能报错；确认 `embedding.model_name_or_path`、依赖与至少一次 **`index_markdown`**。
5. **MinerU API 日志**：`.knotliedge/mineru-api-<port>.log`。

## MCP Tools（两段式）

- `search_knowledge_base(query, top_k=5)`：`chunk_id` / `source_md`
- `search_knowledge_base_fused(...)`：FTS + 向量 + RRF
- `search_hybrid_knowledge_and_radar(...)`：本地融合 + 前沿雷达（OpenAlex）
- `get_knowledge_chunk(chunk_id, window=1)`：更长上下文
- `ping()` / `stats(...)`：健康检查与集合统计
- `universal_academic_search(...)`：统一检索入口
- `plan_research_request` / `summarize_with_citations` / `compare_papers_by_fields`：工作流与引用型执行器（产物在 `output/`，默认被 Git 忽略）

双轨检索默认 **`ai_parse_query=true`** 时会读 `.env` 调用兼容 Chat Completions；可用 MCP 参数关闭或 CLI `--no-ai-parse-query`。

## Cursor 联调（提示）

- 先完成入库后再检索。
- MCP **`cwd`** 为项目根；配置中使用 **`configs/default.yaml` 的绝对路径**更稳。
- 回答前先 `search_knowledge_base`，需严谨引用时用 `get_knowledge_chunk`，并标注 `source_md` + `chunk_id`。

## 推送到 GitHub（简要）

本仓库默认 **忽略** 文献、向量库、`docs/`、`PROJECT_STATUS.md`、`.cursorrules`、`.env` 等。推送需你在本机完成 **Git 身份验证**（见下节）。

```bash
cd /path/to/KnotLiEdge
git init
git add -A
git status   # 确认无 data 正文、无 output 报告、无 .env
git commit -m "Initial public snapshot"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

## GitHub 登录与身份验证说明

- **我无法代替你登录 GitHub**，也**不会自动继承** Cursor/VS Code 里已登录的账号到终端里的 `git push`；是否在编辑器里「绑定 GitHub」只影响编辑器自带功能，不等价于命令行已带好令牌。
- 任选其一即可推送：
  - **HTTPS**：安装 [Git Credential Manager](https://github.com/git-ecosystem/git-credential-manager)，首次 `git push` 时浏览器登录。
  - **GitHub CLI**：`gh auth login`，按提示完成设备授权或浏览器登录。
  - **SSH**：本机生成 SSH key，把公钥加到 GitHub **Settings → SSH keys**，远程改为 `git@github.com:<you>/<repo>.git`。

在 GitHub 网页新建空仓库后，把 `<you>/<repo>` 换成你的即可。

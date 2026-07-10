# RAG 知识库演示

通过 **zvec REST Bridge** 连接向量数据库，使用本地 **Ollama** 的 `qwen3-embedding:4b` 模型进行文本向量化，端到端验证向量库的 RAG 流程：

```
知识入库（文本→向量→存储） → 语义检索 → 生成回答
```

提供两种使用方式：**Web 界面**（可交互）和 **CLI 验证**（自动化测试）。

## 架构

```
┌────────────────┐     HTTP/JSON     ┌──────────────────┐    OpenAI API    ┌────────┐
│  浏览器 / CLI   │ ──────────────►  │  zvec REST Bridge │ ──────────────► │ Ollama │
│                │   注册/插入/检索   │  (FastAPI:8666)   │  qwen3-embedding │ :11434 │
└────────────────┘                   └──────────────────┘                  └────────┘
       │                                      │
       │ Web UI (:8080)                        ▼
       └─────────────► web_app.py (FastAPI)  zvec 向量引擎
```

## 文件结构

```
demo/
├── web_app.py          # Web 后端（FastAPI，端口 8080）
├── rag_demo.py         # CLI 自动化验证脚本
├── kb_data.py          # 共享模块：配置、语料、核心操作
├── static/
│   └── index.html      # Web 前端（单页面应用）
├── requirements.txt    # 依赖
└── README.md
```

## 前置条件

### 1. 启动 zvec REST Bridge

```bash
cd server
pip install -r requirements.txt
pip install openai          # Ollama 兼容 OpenAI 接口，服务端需要此依赖
python main.py              # 默认监听 0.0.0.0:8666
```

验证：`curl http://localhost:8666/health` → `{"status":"UP",...}`

### 2. 启动 Ollama 并拉取模型

```bash
ollama pull qwen3-embedding:4b    # 嵌入模型（必需）
ollama pull qwen3:4b              # 生成模型（RAG 回答用，可选）
```

## 方式一：Web 界面（推荐）

```bash
cd demo
pip install -r requirements.txt
python web_app.py              # 默认端口 8080
```

浏览器访问 **http://localhost:8080**，操作流程：

1. 点击 **「初始化知识库」** — 自动发现维度、注册嵌入函数、创建集合、入库 8 篇文档
2. 在输入框输入问题，或点击示例问题
3. 点击 **「🔍 检索」** 查看语义检索结果（含相关度评分条）
4. 点击 **「🤖 RAG 问答」** 获取基于检索上下文的 AI 生成回答
5. 点击 **「🗑 清理知识库」** 释放资源

## 方式二：CLI 验证

```bash
cd demo
pip install -r requirements.txt
python rag_demo.py
```

运行 8 个步骤的自动化验证（23 项断言），输出每步的 ✅/❌ 结果。

## 配置

通过环境变量覆盖默认地址（对所有方式生效）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ZVEC_URL` | `http://localhost:8666` | zvec REST Bridge 地址 |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 地址 |
| `EMBED_MODEL` | `qwen3-embedding:4b` | 嵌入模型名 |
| `LLM_MODEL` | `qwen3:4b` | 生成模型名（RAG 回答用） |
| `WEB_HOST` | `0.0.0.0` | Web 服务监听地址 |
| `WEB_PORT` | `8080` | Web 服务端口 |

示例：

```bash
LLM_MODEL=gemma4:e4b python web_app.py
```

## 知识库内容

内置 8 篇 RAG / 向量数据库相关中文文档：向量数据库、嵌入模型、RAG、HNSW、余弦相似度、分块策略、混合检索、重排序。

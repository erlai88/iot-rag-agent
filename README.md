# IoT RAG Agent

一个面向 IoT 售后排障场景的 Streamlit RAG Demo。

它支持：

- 上传 PDF 手册并自动入库
- 用向量检索召回相关文档片段
- 用 LLM 基于文档回答问题
- 按需调用模拟设备状态工具
- 记录问答日志、收集 good/bad case、生成评估报告

## 项目介绍

这个项目的目标，是把一个“本地可跑的 RAG 练习”整理成一个更接近真实作品集的演示系统：

- 前端：Streamlit 聊天界面
- 知识库：本地 PDF + ChromaDB
- 检索：OpenAI Embeddings + Top-K 相似度检索
- 工具：模拟设备状态查询
- 观测：JSONL 日志、Bad Case 导出、离线评估报告

你可以把它作为：

- IoT 售后智能助手 Demo
- RAG / Agent 项目模板
- 简历中的在线可访问作品雏形

## 架构说明

```text
用户问题
   |
   v
Streamlit app.py
   |
   v
chain.py -> retriever.py -> ChromaDB
   |             ^
   |             |
   +-> tools.py  |
   |
   v
OpenAI Chat Completions
   |
   v
logger.py -> logs/interactions.jsonl
             evaluate.py -> evaluation_report.md
```

模块说明：

- `ingest.py`
  - 使用 `PyMuPDF` 解析 `data/` 下的 PDF
  - 使用 `RecursiveCharacterTextSplitter` 切分文本
  - 使用 `text-embedding-3-small` 生成向量
  - 写入 `chroma_db/` 的 `iot_manuals` collection
- `retriever.py`
  - 连接已有 Chroma collection
  - 提供向量检索和 BM25 简单重排
- `tools.py`
  - 模拟设备状态工具
  - 提供 OpenAI function calling 所需 schema
- `chain.py`
  - 组合检索、提示词、工具调用和最终回答
- `app.py`
  - 上传 PDF、聊天问答、反馈标注、Bad Case 导出
- `logger.py`
  - 记录问答日志到 JSONL
- `evaluate.py`
  - 读取日志并生成 `evaluation_report.md`

## 目录结构

```text
iot-rag-agent/
├── app.py
├── chain.py
├── ingest.py
├── retriever.py
├── tools.py
├── logger.py
├── evaluate.py
├── requirements.txt
├── .env.example
├── Dockerfile
├── README.md
├── data/
├── logs/
│   └── interactions.jsonl
└── chroma_db/   # 运行后自动生成
```

## 本地运行步骤

### 1. 创建虚拟环境

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
copy .env.example .env
```

然后把 `.env` 中的 `OPENAI_API_KEY` 改成你的真实 Key。

### 4. 启动应用

```bash
streamlit run app.py
```

打开浏览器访问：

- `http://localhost:8501`

### 5. 使用方式

1. 在侧边栏上传 PDF 手册并触发入库
2. 等待知识库建立完成
3. 在主界面提问
4. 查看回答、来源文档、工具调用标签
5. 用 👍 / 👎 标记样例
6. 导出 bad case 报告或运行评估脚本

### 6. 生成评估报告

```bash
python evaluate.py
```

生成文件：

- `evaluation_report.md`

## Docker 运行步骤

### 1. 构建镜像

```bash
docker build -t iot-rag-agent .
```

### 2. 运行容器

```bash
docker run --rm -p 8501:8501 --env-file .env iot-rag-agent
```

然后访问：

- `http://localhost:8501`

## Streamlit Community Cloud 部署步骤

官方文档：

- [Deploy your app on Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)

### 1. 把项目推到 GitHub

建议把当前 `iot-rag-agent/` 目录作为一个独立仓库上传到 GitHub。

务必确认这些内容不要提交：

- `.env`
- `.streamlit/`
- `chroma_db/`
- `logs/`
- `data/`

本仓库自带 `.gitignore`，已经把这些文件排除了。

### 2. 注册并连接 Streamlit Community Cloud

1. 打开 [share.streamlit.io](https://share.streamlit.io/)
2. 注册或登录 Streamlit Community Cloud
3. 连接你的 GitHub 账号

### 3. 创建应用

1. 点击 `Create app`
2. 选择你的 GitHub 仓库
3. 选择分支，通常是 `main`
4. Entrypoint file 填 `app.py`
5. 可选：自定义应用 URL

### 4. 配置 Secrets

根据官方文档，Community Cloud 的 `Advanced settings` 中有 `Secrets` 字段，推荐直接粘贴 `secrets.toml` 风格内容。

最简可用配置：

```toml
OPENAI_API_KEY = "your_openai_api_key"
OPENAI_MODEL = "gpt-4.1-mini"
```

当前代码会优先兼容：

- `st.secrets`
- 环境变量
- 页面手动输入的 API Key

### 5. 点击 Deploy

大多数应用会在几分钟内部署完成。部署成功后，你会拿到一个：

- `https://your-app-name.streamlit.app`

形式的在线链接。

## 部署注意事项

这套项目适合做在线 Demo，但当前仍有一个真实限制：

- `data/`、`logs/`、`chroma_db/` 都是本地文件系统目录
- Streamlit Community Cloud 更适合演示型应用，不适合作为持久化文件/数据库存储

这意味着：

- 你上传的 PDF 和向量库数据不一定能长期保留
- 应用重建、重启或重新部署后，知识库可能需要重新上传

如果你想把它做成更稳定的“真实产品”，下一步建议把：

- PDF 存到对象存储
- 向量库迁移到外部向量数据库
- 日志迁移到数据库或可持久化存储

## 关于 DeepSeek vs OpenAI

当前项目默认使用 OpenAI：

- Embeddings：`text-embedding-3-small`
- Chat model：`OPENAI_MODEL`

如果你未来想切到其他 OpenAI-compatible 服务，不能只改 `chain.py`。

原因是：

- `chain.py` 负责聊天模型
- `ingest.py` 和 `retriever.py` 也在调用 embeddings 接口

所以如果改成 DeepSeek 或其他兼容服务，通常至少需要同时检查：

- Chat client 初始化
- Embeddings client 初始化
- 对应模型名是否可用
- 目标服务是否支持你当前的 embedding 工作流

## 简历写法建议

当你把它成功部署后，可以在简历中这样写：

> 项目已部署上线，可实时访问演示：https://xxx.streamlit.app

这会让项目从“本地练习”更进一步，变成“有真实可访问入口的作品”。

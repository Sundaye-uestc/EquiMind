# EquiMind — 工业重大关键设备多智能体诊断平台

基于 **RAG + ReAct Agent** 的工业设备智能运维平台，专注于高铁接触网、水电机组、航空发动机等重大关键设备的实时监测、故障诊断、预测性维护与报告生成。

## 架构概览

```
EquiMind/
├── run.py                  # 一键启动入口
├── backend/                # FastAPI 后端 (port 8000)
│   ├── api_server.py       # API 路由：诊断、知识库管理、流式输出
│   ├── agent/              # ReAct 智能体
│   │   ├── react_agent.py  # LangChain Agent 封装
│   │   └── tools/          # 工具定义 & 中间件
│   ├── rag/                # RAG 检索增强生成
│   │   ├── vector_store.py # ChromaDB 向量存储（txt/pdf/csv/json）
│   │   └── rag_service.py  # 检索 + LLM 总结服务
│   ├── model/              # 模型工厂（Chat/Embedding）
│   ├── config/             # YAML 配置文件
│   ├── prompts/            # System Prompt 模板
│   ├── utils/              # 工具函数
│   └── data/               # 知识库数据
│       ├── *.txt           # 工业设备维护知识文档
│       ├── cmapss/         # NASA C-MAPSS 涡扇发动机退化数据
│       ├── ntsb/           # NTSB 航空事故数据库
│       └── railway/        # 铁路轨道多模态监测数据
├── frontend/               # Flask 前端 (port 5000)
│   ├── app.py              # Flask 路由
│   └── templates/
│       └── index.html      # 单页应用（诊断对话 + 知识库管理）
├── scripts/                # 工具脚本
│   ├── fetch_datasets.py   # 外部数据集一键下载
│   └── preprocess_datasets.py # 数据预处理（聚合数值数据为摘要）
└── faq-examples.md         # 常见提问示例
```

## 快速开始

### 环境要求

- Python 3.12+
- Windows / Linux / macOS

### 安装

```bash
git clone https://github.com/Sundaye-uestc/AgentLearning.git
cd AgentLearning

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate # Linux/macOS

# 安装依赖
pip install fastapi uvicorn flask python-dotenv langchain langchain-community langchain-chroma langchain-openai chromadb dashscope pandas requests pyyaml
```

### 配置

编辑 `backend/config/rag.yaml`，填写模型和 Embedding 的 API Key：

```yaml
chat_model_name: deepseek-v4-pro
chat_api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
chat_api_key_env: sk-your-api-key
embedding_model_name: text-embedding-v4
embedding_provider: dashscope
embedding_api_key_env: sk-your-api-key
```

### 启动

```bash
python run.py
```

- 后端 API：http://127.0.0.1:8000
- 前端界面：http://127.0.0.1:5000
- API 文档：http://127.0.0.1:8000/docs

## 功能特性

### 智能诊断

- **ReAct 推理**：思考 → 行动 → 观察 → 再思考，自主调用工具检索信息
- **流式输出**：思考过程和诊断报告实时流式渲染
- **打字机效果**：思考过程与报告均支持逐字输出（10ms 间隔）
- **随机开场**：每次对话的开场白和过渡句随机选取，体验更自然
- **报告下载**：诊断完成后支持一键下载 Markdown 格式报告

### 知识库管理

| 功能 | 说明 |
|------|------|
| 文件格式 | PDF / TXT / CSV / JSON |
| 向量存储 | ChromaDB（默认 200 字符分片，20 字符重叠） |
| 上传方式 | Web 界面拖拽上传 / 文件选择 |
| 去重机制 | MD5 哈希自动跳重 |
| 文档管理 | 支持查看、搜索、删除已摄入文档 |
| 目录扫描 | 递归扫描子目录，自动发现新增文件 |

### 内置知识库

| 数据集 | 内容 | 来源 |
|--------|------|------|
| 工业设备状态评估标准 | 设备状态评估方法与判定标准 | 行业规范 |
| 水电机组故障排查与维护 | 水电机组常见故障及处理方案 | 运维手册 |
| 航空发动机预测性维护 | 航空发动机 PHM 技术与标准 | 技术文献 |
| IEC 63270 预测性维护规范 | 预测性维护国际标准要点 | 国际标准 |
| 高铁接触网监测运维 | 接触网监测参数与运维流程 | 行业规程 |
| **NASA C-MAPSS** | 涡扇发动机 FD001-FD004 退化仿真数据 | NASA PCoE |
| **NTSB 航空事故** | 美国航空事故历史记录（2008-至今） | NTSB |
| **铁路轨道监测** | 40Hz 多模态传感器数据（加速度/陀螺仪/GPS） | Zenodo |

### 外部数据集获取

```bash
# 一键下载三大外部数据集
python scripts/fetch_datasets.py --all

# 单独下载
python scripts/fetch_datasets.py --cmapss    # NASA C-MAPSS
python scripts/fetch_datasets.py --ntsb      # NTSB 航空事故
python scripts/fetch_datasets.py --railway   # 铁路轨道监测

# 预处理（聚合数值数据为摘要，优化摄入）
python scripts/preprocess_datasets.py
```

## 技术栈

| 层级 | 技术 |
|------|------|
| LLM | DeepSeek-V4-Pro (DashScope) |
| Embedding | text-embedding-v4 (DashScope) |
| Agent | LangChain ReAct Agent |
| 向量库 | ChromaDB |
| 后端 | FastAPI + SSE 流式输出 |
| 前端 | Flask + 原生 HTML/CSS/JS |
| 文档解析 | LangChain Document Loaders (PyPDF/Text/CSV/JSON) |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/diagnosis/start` | 发起诊断任务 |
| `GET` | `/api/diagnosis/tasks/{id}/status` | 查询任务状态 |
| `GET` | `/api/diagnosis/tasks/{id}/thinking/stream` | 思考过程 SSE 流 |
| `GET` | `/api/diagnosis/tasks/{id}/report/stream` | 诊断报告 SSE 流 |
| `GET` | `/api/knowledge/stats` | 知识库统计信息 |
| `GET` | `/api/knowledge/documents` | 文档列表 |
| `POST` | `/api/knowledge/upload` | 上传文档 |
| `DELETE` | `/api/knowledge/documents/{id}` | 删除文档 |
| `POST` | `/api/knowledge/search` | 知识库搜索 |

## 提问示例

参见 [faq-examples.md](faq-examples.md)，涵盖航空发动机、高铁接触网、水电机组、铁路轨道、预测性维护等 7 大类 45+ 个典型问题。

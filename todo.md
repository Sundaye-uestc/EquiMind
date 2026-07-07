# vLLM 本地模型迁移 + 微调准备 — 可执行任务清单

> 目标：将 EquiMind 的模型层从 DashScope 云端 API 迁移到本地 vLLM 推理服务，并建立微调基础设施，实现"微调 + RAG"组合方案。
>
> **部署环境**：远程服务器 `439-4090-2`（`lab@192.168.31.12:22`，**须经跳板机 `1.14.166.203:22222` 中转**），RTX 4090 24GB ×1，CUDA 13.0，Ubuntu 24.04，RAM 32GB，数据盘 `/data` 12.7TB。
> **工作目录**：`/data/sqx/Equimind`（全部放在数据盘，系统盘仅 16GB 剩余，不可用）。

---

## Phase 0：远程环境准备 ✅ 已完成

### 0.1 工作目录
- [x] 已在远程创建 `/data/sqx/Equimind/`，目录结构同本地项目
- [x] 确认 `lab` 用户对 `/data` 有完整读写权限（属主 lab:lab，0755）

### 0.2 Conda 环境（⚠️ 注意：建在 `/data` 而非 `/home`，系统盘太小）
- [x] Anaconda 位于 `/home/lab/anaconda3`，已加入 PATH
- [x] **conda 环境建在数据盘**（系统盘仅 24GB 剩余，装不下 PyTorch + vLLM）：
  ```bash
  source /home/lab/anaconda3/etc/profile.d/conda.sh
  conda create --prefix /data/sqx/conda/equimind python=3.12 -y
  conda activate /data/sqx/conda/equimind
  ```

### 0.3 模型与数据存储路径
- [x] 环境变量已写入 `~/.bashrc`：
  ```bash
  export HF_HOME=/data/sqx/models/huggingface
  export MODELSCOPE_CACHE=/data/sqx/models/modelscope
  ```
- [x] pip 缓存也重定向到数据盘（避免下载大包时填满系统盘）：
  ```bash
  export PIP_CACHE_DIR=/data/pip-cache
  export TMPDIR=/data/tmp
  ```
- [x] 目录已创建：`/data/sqx/models/{huggingface,modelscope}`、`/data/{pip-cache,tmp}`

### 0.4 依赖安装（实际版本）
- [x] PyTorch：vLLM 自动带入 **2.11.0+cu130**（CUDA 13 原生，兼容性无问题）
- [x] vLLM：**0.24.0**
- [x] LangChain：**1.3.2** | ChromaDB：**1.5.9** | FastAPI：**0.136.3**
- [x] GPU 验证通过：`RTX 4090, 24GB VRAM`
- [x] vLLM 导入验证通过

### 0.5 代码同步
- [x] 通过 tar+ssh 管道同步到 `/data/sqx/Equimind/`
- [x] `rag.yaml`（含 API Key，被 .gitignore 排除）已手动 scp 上传
- [x] 路径已适配 `/data/sqx/Equimind`

### 0.6 快速连接备忘

> ⚠️ 服务器不响应直连 Ping，须走跳板机。当前跳板机可用 key：`~/.ssh/id_ed25519_jumpbox`

```bash
# 方式 A：跳板机一行式
ssh -o ProxyCommand="ssh -i ~/.ssh/id_ed25519_jumpbox -W %h:%p -p 22222 root@1.14.166.203" \
    -i ~/.ssh/id_ed25519_jiuzhou -p 6000 lab@localhost

# 方式 B：SSH 配置（~/.ssh/config）
#   Host 439-4090-2
#     HostName localhost
#     Port 6000
#     User lab
#     IdentityFile ~/.ssh/id_ed25519_jiuzhou
#     ProxyJump jumpbox
#     ServerAliveInterval 60
#   Host jumpbox
#     HostName 1.14.166.203
#     Port 22222
#     User root
#     IdentityFile ~/.ssh/id_ed25519_jumpbox

# 连上后
source /home/lab/anaconda3/etc/profile.d/conda.sh
conda activate /data/sqx/conda/equimind
cd /data/sqx/Equimind
```

---

## Phase 1：模型选型与下载（适配单卡 24GB）

### 1.1 两层"思考"：前端展示的技术基础

EquiMind 的前端展示了两种"思考过程"，它们的技术来源不同：

| 思考层 | 技术来源 | 前端展示方式 | 对模型的依赖 |
|--------|----------|--------------|--------------|
| Agent 推理步骤 (ReAct 循环) | LangChain ReAct Agent 框架 | SSE 流式推送，typewriter 效果 | 任何模型均可，由 LangChain 中间件驱动 |
| 模型内部推理 (thinking tokens) | 模型自身输出的 chain-of-thought | 嵌入 Agent 每个 ReAct 步骤的思考中 | 需模型原生支持 thinking 模式 |

> 当前 DashScope 配置中 `enable_thinking: true` 即启用第二层（DeepSeek 的 `reasoning_content`）。
> 切换到本地模型后，第一层（ReAct Agent 思考）不受影响；第二层需要选用支持 thinking 模式的模型。

### 1.2 Chat 模型选型（修订版 — RTX 4090 24GB 单卡，优先 thinking 支持）

| 模型 | 方案 | 显存 | Thinking | 中文能力 | 工具调用 | 推荐度 |
|------|------|------|---------|---------|---------|--------|
| Qwen3-14B-AWQ | AWQ INT4 | ~9.4 GB | 原生 | ★★★★ | ★★★ | ✅ 首选（已下载） |
| DeepSeek-R1-Distill-Qwen-14B | AWQ | ~8 GB | R1 风格 | ★★★★ | ★★★ | 推理特化备选 |
| Qwen3-8B | FP16 | ~16 GB | 原生 | ★★★ | ★★★ | 备选 |
| Qwen3-32B-AWQ | AWQ INT4 | ~18 GB | 原生 | ★★★★★ | ★★★★ | 显存紧张时 |

> **已排除**：Qwen3-32B FP16（~64GB）、Qwen3-14B FP16（~28GB）——单卡 24GB 装不下。
> **Thinking 说明**：Qwen3 全系支持 thinking 模式（通过 chat_template 输出 `<｜end▁of▁thinking｜>` 标签）；DeepSeek-R1-Distill 系列原生输出 R1 风格的 `<｜end▁of▁thinking｜>` 格式。两者前端均可展示。

### 1.3 vLLM 中启用 Thinking 模式

Qwen3 在 vLLM 中启用 thinking 的两种方式：

**方式 A**：通过 `extra_body` 传参（推荐，与现有 `factory.py` 兼容）
```python
# factory.py 中 ChatOpenAI 的 extra_body 改为:
#   extra_body = {"enable_thinking": True}
# vLLM 0.24+ 对 Qwen3 会通过 chat_template 自动启用 thinking tokens
```

**方式 B**：启动 vLLM 时指定 reasoning parser
```bash
vllm serve Qwen/Qwen3-14B-AWQ \
  --reasoning-parser deepseek_r1 \   # 或 qwen3
  ...
```

> ⚠️ `enable_thinking` 在 vLLM 中的行为与 DashScope 不完全相同：
> - DashScope DeepSeek：模型返回 `reasoning_content` + `content` 两个独立字段
> - vLLM Qwen3：模型在 `content` 中内嵌 `<｜end▁of▁thinking｜>` 标签
> - **需要在 Phase 5 中验证前端 SSE 解析逻辑是否需要适配**

### 1.2 Chat 模型下载 ✅ 已完成

> ⚠️ **重要踩坑**：
> 1. HuggingFace 上模型 ID 是 `Qwen/Qwen3-14B-AWQ`（**无 "Instruct"**），`Qwen/Qwen3-14B-Instruct-AWQ` 不存在
> 2. ModelScope 无 Qwen3 系列（返回 404），只能从 HuggingFace 下载
> 3. 服务器 IPv6 不可用，`huggingface_hub` 默认走 IPv6 → SYN-SENT 卡死，**必须强制 IPv4**
> 4. 跳板机访问链：`jumpbox(1.14.166.203:22222) → localhost:6000`

- [x] **下载命令**（需强制 IPv4）：
  ```python
  # force IPv4 + use correct model ID
  import socket, urllib3.util.connection
  urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET
  snapshot_download("Qwen/Qwen3-14B-AWQ", local_dir="...")
  ```
- [x] **实际大小**：9.4 GB（2×4.7GB safetensors），显存约 9.4 GB
- [x] 验证通过：`model_type: qwen3`, 40 layers, hidden_size=5120, thinking 原生支持

### 1.3 Embedding 模型下载 ✅ 已完成

- [x] `BAAI/bge-m3` 通过 **ModelScope** 下载（ModelScope 有此模型，国内快）：
  ```bash
  modelscope download --model BAAI/bge-m3
  ```
- [x] 文件路径：`/data/sqx/models/modelscope/models/BAAI--bge-m3/snapshots/master/`
- [x] 验证通过：XLM-RoBERTa, hidden=1024, 24 layers, 1024维向量
- [x] 权重文件：`pytorch_model.bin` 2.2 GB

### 1.4 显存预算（实测修正）
```
Chat (Qwen3-14B-AWQ):    ~9.4 GB  （实测，非估算8GB）
Embedding (BGE-M3):      ~2 GB
KV Cache + 开销:         ~6 GB
─────────────────────────────────
合计:                    ~17.4 GB / 24 GB  ← 余量 6.6GB，仍充裕
```

---

## Phase 2：配置文件重构（多后端切换）✅ 已完成

### 2.1 `rag.yaml` 配置扩展 ✅
- [x] 在 `backend/config/rag.yaml` 中增加 `backend` 字段（`local | dashscope | openai_compatible`）
- [x] `local` 配置块：chat/embedding 均走 `http://localhost:8002/v1` + `http://localhost:8003/v1`
- [x] `dashscope` 配置块：保留原有云端 API 所有参数，一键切回
- [x] key 命名统一为 `chat_api_key_env` / `embedding_api_key_env`（与 `factory.py` 兼容）
- [x] 旧格式向后兼容：无 `backend` 字段时原样返回（`_merge_backend_config` passthrough）
- [x] 修改 `config_handler.py`：新增 `_merge_backend_config()` 后端分发逻辑
  - 读取 `backend` 字段 → 合并对应命名块到顶层扁平 dict
  - `factory.py` 无需任何改动即可适配多后端

### 2.2 新增 `vllm.yaml`（vLLM 服务参数）✅
- [x] 创建 `backend/config/vllm.yaml`：
  - `chat`：port 8002, tensor_parallel=1, max_model_len=32768, gpu_mem=0.85, dtype=auto
  - `embedding`：port 8003, tensor_parallel=1, max_model_len=8192, gpu_mem=0.10, task=embed
  - `model_path` 指向实际下载路径（`/data/sqx/models/...`）
- [x] `config_handler.py` 新增 `load_vllm_config()` 函数

> **验证通过**：dashscope 和 local 两种后端的 merge 结果均正确；factory.py 导入无变化。

---

## Phase 3：模型工厂重构 ✅ 已完成

### 3.1 `factory.py` 改造 ✅
- [x] `ChatModelFactory`：显式后端分发 `_create_local()` / `_create_dashscope()` / `_create_openai_compat()`
  - `local`：`ChatOpenAI(base_url=vllm_url, api_key="not-needed")`，**不传 `enable_thinking`**（Qwen3 由 chat_template 处理 thinking）
  - `dashscope`：保持原有逻辑 + `enable_thinking` extra_body
  - `openai_compatible`：通用 OpenAI 兼容模式
- [x] `EmbeddingsFactory`：同样三路分发
  - `local`：`OpenAIEmbeddings(base_url=vllm_embed_url)`
  - `dashscope`：`DashScopeEmbeddings`（不变）
  - `openai_compatible`：`OpenAIEmbeddings`（不变）
- [x] 启动自检：`_check_vllm_health()` ping `/v1/models`，失败时 logger.warning（非阻塞）

### 3.2 Embedding 维度兼容 ✅
- [x] `chroma.yaml` 新增 `embedding_dimension: 1024`（text-embedding-v4 / BGE-M3 均为 1024 维）
- [x] `config_handler.py` 新增：
  - `write_chroma_metadata()`：构建向量库后写入元数据文件（`.equimind_embedding_meta.json`）
  - `check_chroma_embedding_compatibility()`：启动时检查维度一致性（元数据文件 → SQLite 推断 → 跳过）
- [x] `vector_store.py`：`load_document()` 完成后自动写入元数据
- [x] 模块加载时自动执行兼容性检查，不匹配时输出 stderr 警告

> **验证通过**：dashscope/local 两种后端 merge + factory 创建均正确；同维度→兼容，不同维度→报警。

---

## Phase 4：vLLM 服务启动脚本（Linux .sh）✅ 已完成

### 4.1 Chat 服务启动脚本 ✅
- [x] `scripts/start_vllm_chat.sh`：
  - conda 激活 `/data/sqx/conda/equimind`，`set -euo pipefail`
  - 模型路径 `/data/sqx/models/huggingface/Qwen3-14B-AWQ`，端口 8002
  - 日志落 `/data/sqx/Equimind/logs/vllm_chat_<timestamp>.log`
  - 参数：`--max-model-len 32768 --gpu-memory-utilization 0.85 --dtype auto`

### 4.2 Embedding 服务启动脚本 ✅
- [x] `scripts/start_vllm_embedding.sh`：
  - 模型路径 `/data/sqx/models/modelscope/models/BAAI--bge-m3/snapshots/master`
  - 端口 8003，`--task embed`，`--gpu-memory-utilization 0.10`
  - 同样带时间戳日志

### 4.3 一站式启动脚本 ✅
- [x] `scripts/start_all.sh`：
  - 支持 `--frontend-only` / `--backend-only` / `--skip-vllm` 参数
  - vLLM Chat + Embedding 并行后台启动 → `wait_for_health()` 轮询 `/health`（Chat 最长 5min，Embedding 最长 2min）
  - 两个 vLLM 就绪后 → FastAPI (8000) → Flask (5000)
  - `trap cleanup EXIT INT TERM` 统一清理所有子进程
  - 主循环监控子进程存活，任意退出则全部关闭

### 4.4 修改 `run.py` ✅
- [x] 新增 `preflight_check()`：读取 `rag.yaml` 的 `backend` 字段
  - `dashscope`：跳过检查，直接启动
  - `local`：检测 vLLM Chat + Embedding 是否就绪，未就绪时显示醒目提示框并询问是否继续
- [x] 新增 CLI 参数：`--skip-vllm-check`（调试用）、`--backend-only`、`--frontend-only`
- [x] 使用 `argparse` 替代原来的零参数启动

> **部署注意**：将 `.sh` 文件上传到远程后需 `chmod +x scripts/*.sh`

---

## Phase 5：Prompt 适配与调优（核心风险点）✅ 已完成

### 5.1 System Prompt 适配 ✅
- [x] **发现关键问题**：原 `main_prompt.txt` 描述了 7+ 工具，但实际注册仅 7 个，其中 `get_equipment_status`、`get_maintenance_record`、`get_equipment_id`、`get_time_range` 在代码中不存在。Qwen3 工具调用纪律弱于 DeepSeek，此不匹配会显著增加幻觉风险。
- [x] **创建 Qwen3 专用提示词** `backend/prompts/main_prompt_qwen3.txt`：
  - 仅描述实际注册的 7 个工具（`rag_summarize`、`get_weather`、`get_user_city`、`get_user_id`、`get_month`、`get_gender`、`fill_context_for_report`）
  - 精简指令，减少冗余描述，降低 Qwen3 指令遵循负担
  - 保留工业领域专业框架
- [x] `prompt_loader.py` 改造：`_get_main_prompt_filename()` 按后端自动选择提示词
  - `backend: local` → `main_prompt_qwen3.txt`
  - `backend: dashscope` → `main_prompt.txt`（不变）
- [x] vLLM OpenAI 兼容层自动处理 chat_template，无需手动干预 ✅
- [x] `enable_thinking` 仅 dashscope 后端传递，local 模式不传 ✅
- [x] `factory.py` 新增 `chat_model_kwargs` 支持：通过 rag.yaml 传入 temperature/top_p 等参数

### 5.2 工具调用验证 ✅
- [x] 创建 `scripts/test_tool_calling.py` — 自动化测试框架：
  - 覆盖 5 大类别：航空发动机、高铁接触网、水电机组、铁路轨道、报告生成
  - 共 21 条测试用例（来自 `faq-examples.md`）
  - 支持 `--category` 分类测试、`--interactive` 交互模式、`--export` 导出 JSONL 训练数据
  - 测试汇总显示期望工具 vs 实际调用，标记异常 case 供 Phase 8 微调使用
- [x] 需人工审查维度已在脚本输出中明确标注（工具调用正确性、凭空回答、死循环、报告格式、术语准确度）

### 5.3 备选方案 ✅
- [x] `rag.yaml` 新增完整模型切换指南 + 备选模型注释：
  - Qwen3-8B FP16（~16GB）：工具调用可能更稳定
  - Qwen3-32B-AWQ（~18GB）：最强推理但需调低 gpu_mem
  - Qwen2.5-14B-AWQ（备胎）：社区反馈多，工具调用成熟
  - 切换步骤文档（改 backend → 改 vllm.yaml → 重建 ChromaDB）
- [x] 一键切回 DashScope：`backend: dashscope` 即可

> **验证通过**：dashscope + local 双后端提示词自动切换正确；test_tool_calling.py 参数解析正常；chat_model_kwargs 传递正常。

---

## Phase 6：端到端测试

### 6.1 功能回归测试
- [ ] 设置 `backend: local`，启动全服务
- [ ] 重建向量库 + 重新摄入知识库文档（C-MAPSS / NTSB / railway 等）
- [ ] 逐一测试 5 大设备类型（参考 `faq-examples.md`）：
  - [ ] 航空发动机诊断
  - [ ] 高铁接触网运维
  - [ ] 水电机组故障
  - [ ] 铁路轨道监测
  - [ ] 报告生成
- [ ] 对比云端 vs 本地输出质量，记录退化程度

### 6.2 性能测试
- [ ] 首 token 延迟（TTFT）
- [ ] 端到端诊断总时间
- [ ] 并发诊断（ThreadPoolExecutor max_workers=4）
- [ ] 显存峰值（`nvidia-smi` 持续监控）
- [ ] 吞吐量（tokens/s）

### 6.3 云端 vs 本地预期对比

| 指标 | 云端 DeepSeek-V4-Pro | 本地 Qwen3-14B-AWQ |
|------|----------------------|----------------------|
| 首 token 延迟 | ~200-500ms (网络) | ~50-200ms (本地) |
| 复杂推理质量 | ★★★★★ | ★★★★ (预期) |
| 工具调用准确率 | ★★★★★ | ★★★ (待验证) |
| 报告格式遵循 | ★★★★★ | ★★★ (待验证) |
| 成本 | API 按量付费 | 电费 + 硬件折旧 |

---

## Phase 7：微调基础设施（为 Phase 8 做准备）

### 7.1 训练数据采集管线
- [ ] 在 `backend/agent/react_agent.py` 中增加日志记录：
  - 用户原始问题
  - Agent 每一步的 thought
  - 每次工具调用的名称 + 参数 + 返回值
  - 最终输出
- [ ] 落盘为结构化 JSONL（OpenAI messages 格式）：
  ```json
  {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": null, "tool_calls": [{"name": "...", "arguments": {"key": "value"}}]},
      {"role": "tool", "content": "..."},
      {"role": "assistant", "content": "最终回复"}
    ]
  }
  ```
- [ ] 创建 `scripts/export_training_data.py`：日志 → 微调数据集

### 7.2 微调数据标注工具
- [ ] 创建 Streamlit 标注界面（`scripts/annotation_app.py`）：
  - 展示原始对话
  - 标注：工具调用是否正确、最终诊断是否合理
  - 支持手动修正工具调用序列和回复
  - 导出训练格式（ShareGPT / OpenAI messages）

### 7.3 微调框架选型
- [ ] 选择 **LLaMA-Factory**（对 Qwen 支持好，GUI 友好）：
  ```bash
  git clone https://github.com/hiyouga/LLaMA-Factory.git
  pip install -e ".[torch,metrics]"
  ```
- [ ] 微调方法：**LoRA**（单卡 24GB 唯一可行方案）
  - Qwen3-8B FP16 LoRA → ~18GB 显存，可行
  - Qwen3-14B AWQ + LoRA → ~10GB，但配置更复杂
- [ ] 验证集：标注数据中 20% hold-out

### 7.4 评估指标设计
- [ ] 定义评估维度：
  - **工具调用准确率**：正确时机调用正确工具
  - **工具参数正确率**：参数是否合理
  - **报告格式遵循率**：是否遵循预设模板
  - **术语使用正确率**：工业术语准确度
  - **幻觉率**：事实性错误比例
- [ ] 创建 `scripts/eval_finetune.py`：自动化评估

---

## Phase 8：LoRA 微调执行（后续，本期按需启动）

### 8.1 第一阶段：工具调用微调
- [ ] 用标注的工具调用数据，LoRA 微调 Chat 模型
- [ ] 目标：提升"何时调用 RAG""何时触发报告生成"的决策准确率
- [ ] 验证：微调前后 50 个测试 case 工具调用准确率对比

### 8.2 第二阶段：报告格式 + 领域推理
- [ ] 用标注的完整对话数据（含最终报告），继续 LoRA 微调
- [ ] 目标：内化报告模板格式，减少对长 System Prompt 的依赖
- [ ] 验证：报告格式遵循率对比

### 8.3 微调模型部署
- [ ] 合并 LoRA 权重，或 vLLM `--lora-modules` 直接加载
- [ ] 更新 `rag.yaml` 的 `chat_model_name` 指向微调后模型
- [ ] 端到端回归测试

---

## 文件变更总览

| 阶段 | 文件 | 操作 |
|------|------|------|
| Phase 0 | `/data/sqx/Equimind/` | 新建：远程工作目录 |
| Phase 0 | `/data/sqx/conda/equimind/` | 新建：conda 环境（数据盘） |
| Phase 0 | `/data/sqx/models/{huggingface,modelscope}/` | 新建：模型缓存目录 |
| Phase 1 | `/data/sqx/models/huggingface/Qwen3-14B-AWQ/` | 新建：Chat 模型（9.4GB） |
| Phase 1 | `/data/sqx/models/modelscope/models/BAAI--bge-m3/` | 新建：Embedding 模型（4.3GB） |
| Phase 1 | `/data/tmp/dl_qwen3.py` | 新建：HF 下载脚本（强制 IPv4） |
| Phase 2 | `backend/config/rag.yaml` | 修改：增加 backend + local/dashscope 配置块 |
| Phase 2 | `backend/config/vllm.yaml` | 新建：vLLM 服务参数 |
| Phase 2 | `backend/utils/config_handler.py` | 修改：增加 backend 分发逻辑 |
| Phase 3 | `backend/model/factory.py` | 修改：多后端 Chat/Embedding 工厂 |
| Phase 4 | `scripts/start_vllm_chat.sh` | 新建：Chat 启动脚本 |
| Phase 4 | `scripts/start_vllm_embedding.sh` | 新建：Embedding 启动脚本 |
| Phase 4 | `scripts/start_all.sh` | 新建：一站式启动 |
| Phase 4 | `run.py` | 修改：增加 vLLM 状态检测 |
| Phase 5 | `backend/prompts/` | 可能修改：Prompt 模板适配 |
| Phase 6 | `backend/config/chroma.yaml` | 修改：persist_directory 适配新路径 |
| Phase 7 | `backend/agent/react_agent.py` | 修改：增加训练数据日志 |
| Phase 7 | `scripts/export_training_data.py` | 新建：日志→训练数据 |
| Phase 7 | `scripts/annotation_app.py` | 新建：Streamlit 标注工具 |
| Phase 8 | `scripts/eval_finetune.py` | 新建：微调评估 |

---

## 注意事项

1. **端口分配**：vLLM Chat `8002` + Embedding `8003` + FastAPI `8000` + Flask `5000`，互不冲突
2. **ChromaDB 必须重建**：切换 Embedding 模型后向量空间不兼容，用新模型重新摄入所有 `data/` 文档
3. **API Key 占位**：vLLM 本地不需要真 key，`ChatOpenAI` 要求非空，传 `"not-needed"`
4. **冷启动等待**：vLLM 加载 AWQ 模型约 1-3 分钟，启动脚本需轮询 `/health` 等待就绪
5. **CUDA 13 兼容性**：Driver 580 向下兼容 CUDA 12.x 二进制，但未经广泛测试
6. **所有存储落 `/data`**：模型、向量库、数据集、日志全部在数据盘，系统盘不存大文件
7. **环境变量持久化**：`HF_HOME`、`MODELSCOPE_CACHE` 写入 `~/.bashrc`，避免每次手动 export
8. **AWQ 回退**：若 Qwen3-14B-AWQ 不可用 → 尝试 Qwen3-8B FP16（~16GB）或 Qwen2.5-14B-AWQ
9. **跳板机连接**：直连 192.168.31.12 延迟极高/Ping 不通，必须经 `jumpbox(1.14.166.203:22222) → localhost:6000`；推荐写入 `~/.ssh/config` 用 `ssh 439-4090-2` 一键连接
10. **HF 下载强制 IPv4**：服务器 IPv6 不可用，`huggingface_hub` 下载模型时必须在 Python 代码中 `urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET`，否则卡 SYN-SENT
11. **HuggingFace 模型 ID 注意**：`Qwen/Qwen3-14B-AWQ`（**无 "Instruct"**），用错 ID 返回 401
12. **ModelScope 无 Qwen3**：Qwen3 系列不在 ModelScope 上，只能用 HuggingFace 下载；但 bge-m3 等模型在 ModelScope 上有且下载快

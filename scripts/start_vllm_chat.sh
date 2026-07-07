#!/bin/bash
# ============================================================
# EquiMind — 启动 vLLM Chat 服务 (Qwen3-14B-AWQ)
# ============================================================
# 端口: 8002  |  显存: ~9.4 GB  |  GPU: RTX 4090 ×1
# 参数与 backend/config/vllm.yaml 保持同步
#
# 环境修复（PyTorch 2.11+cu130 on CUDA 12.0 系统）：
#   1. CUDA 13 运行时库 → LD_LIBRARY_PATH
#   2. CUDA 13 nvcc → PATH（FlashInfer JIT 编译需要）
#   3. enforce-eager → 禁用 CUDA Graph（避免 FlashInfer 采样内核 JIT 失败）
# ============================================================
set -euo pipefail

# ---- conda 环境 ----
source /home/lab/anaconda3/etc/profile.d/conda.sh
conda activate /data/sqx/conda/equimind

# ---- CUDA 13 路径修复（pip 安装的 nvidia-cuda-* 包） ----
NVIDIA_DIR="/data/sqx/conda/equimind/lib/python3.12/site-packages/nvidia"
export PATH="${NVIDIA_DIR}/cu13/bin:${PATH}"              # nvcc 13（FlashInfer JIT）
export LD_LIBRARY_PATH="${NVIDIA_DIR}/cu13/lib:${NVIDIA_DIR}/nccl/lib:${LD_LIBRARY_PATH:-}"

# ---- FlashInfer CCCL 兼容性修复（用 CUDA 13 CCCL 替换 FlashInfer 绑定的 CUDA 12 CCCL） ----
FLASHINFER_CCCL="/data/sqx/conda/equimind/lib/python3.12/site-packages/flashinfer/data/cccl"
CUDA13_CCCL="/data/sqx/conda/equimind/lib/python3.12/site-packages/nvidia/cu13/include/cccl"
if [ ! -L "$FLASHINFER_CCCL" ]; then
    mv "$FLASHINFER_CCCL" "$FLASHINFER_CCCL.bak" 2>/dev/null || true
    ln -s "$CUDA13_CCCL" "$FLASHINFER_CCCL"
fi

# ---- 清除 FlashInfer 缓存（强制用 CUDA 13 nvcc 重新编译） ----
rm -rf /home/lab/.cache/flashinfer

echo "[start_vllm_chat] 使用 nvcc: $(which nvcc)"
echo "[start_vllm_chat] 使用 nvcc 版本: $(nvcc --version 2>&1 | head -3 || echo 'nvcc not found')"

# ---- 模型与服务参数 ----
MODEL_PATH="/data/sqx/models/huggingface/Qwen3-14B-AWQ"
HOST="0.0.0.0"
PORT=8002
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.85
TP=1

# ---- 日志 ----
LOG_DIR="/data/sqx/Equimind/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_chat_$(date +%Y%m%d_%H%M%S).log"

echo "[start_vllm_chat] 模型: $MODEL_PATH"
echo "[start_vllm_chat] 端口: $PORT  |  显存利用率: $GPU_MEM_UTIL  |  max_model_len: $MAX_MODEL_LEN"
echo "[start_vllm_chat] 日志: $LOG_FILE"
echo "[start_vllm_chat] 加载 AWQ 模型约需 1-2 分钟..."

vllm serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --dtype auto \
  --trust-remote-code \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  2>&1 | tee "$LOG_FILE"

#!/bin/bash
# ============================================================
# EquiMind — 启动 vLLM Embedding 服务 (BAAI/bge-m3)
# ============================================================
# 端口: 8003  |  显存: ~2 GB  |  GPU: RTX 4090 ×1
# 参数与 backend/config/vllm.yaml 保持同步
# ============================================================
set -euo pipefail

# ---- conda 环境 ----
source /home/lab/anaconda3/etc/profile.d/conda.sh
conda activate /data/sqx/conda/equimind

# ---- 模型与服务参数 ----
MODEL_PATH="/data/sqx/models/modelscope/models/BAAI--bge-m3/snapshots/master"
HOST="0.0.0.0"
PORT=8003
MAX_MODEL_LEN=8192
GPU_MEM_UTIL=0.10
TP=1

# ---- 日志 ----
LOG_DIR="/data/sqx/Equimind/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_embedding_$(date +%Y%m%d_%H%M%S).log"

echo "[start_vllm_embedding] 模型: $MODEL_PATH"
echo "[start_vllm_embedding] 端口: $PORT  |  显存利用率: $GPU_MEM_UTIL  |  task: embed"
echo "[start_vllm_embedding] 日志: $LOG_FILE"
echo "[start_vllm_embedding] bge-m3 加载约需 30-60 秒..."

vllm serve "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --task embed \
  --trust-remote-code \
  2>&1 | tee "$LOG_FILE"

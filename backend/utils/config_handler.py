"""
EquiMind 配置加载模块

支持多后端配置：
  - dashscope（云端 API）
  - local（本地 vLLM 推理服务）
  - openai_compatible（其他 OpenAI 兼容 API）

rag.yaml 使用 backend + 命名块结构，此模块在加载时将活跃后端的
配置合并为扁平 dict，使上游调用方（factory.py 等）无需感知后端切换。
"""
import json
import os
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from utils.path_tool import get_abs_path


project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 已知的后端配置块名称（用于合并时过滤）
_KNOWN_BACKENDS = {"local", "dashscope", "openai_compatible"}

# ChromaDB 元数据文件名
_CHROMA_META_FILENAME = ".equimind_embedding_meta.json"


def _merge_backend_config(raw: dict) -> dict:
    """将 raw[backend] 块的内容合并到顶层，返回扁平化后的配置。

    若 raw 中没有 backend 字段（旧格式），直接返回 raw 本身。
    合并后的 dict 会保留 backend 字段以标记当前活跃后端。
    """
    backend = raw.get("backend")
    if backend is None:
        # 旧格式：没有 backend 字段，视为 dashscope 兼容模式
        return raw

    # 收集顶层非后端块的键（如未来可能存在的全局配置）
    merged = {
        k: v for k, v in raw.items()
        if k != "backend" and k not in _KNOWN_BACKENDS
    }

    # 合并活跃后端块
    if backend in raw:
        merged.update(raw[backend])

    # 保留 backend 标记，方便上游判断当前模式
    merged["backend"] = backend
    return merged


def load_rag_config(
    config_path: str = None,
    encoding: str = "utf-8",
) -> dict:
    """加载 RAG 配置，自动将 backend 块合并为扁平格式。"""
    if config_path is None:
        config_path = get_abs_path(project_root_path + "/config/rag.yaml")
    with open(config_path, "r", encoding=encoding) as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)
    return _merge_backend_config(raw)


def load_chroma_config(
    config_path: str = None,
    encoding: str = "utf-8",
) -> dict:
    """加载 ChromaDB 向量库配置。"""
    if config_path is None:
        config_path = get_abs_path(project_root_path + "/config/chroma.yaml")
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(
    config_path: str = None,
    encoding: str = "utf-8",
) -> dict:
    """加载 Agent 配置。"""
    if config_path is None:
        config_path = get_abs_path(project_root_path + "/config/agent.yaml")
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(
    config_path: str = None,
    encoding: str = "utf-8",
) -> dict:
    """加载 Prompt 模板路径配置。"""
    if config_path is None:
        config_path = get_abs_path(project_root_path + "/config/prompts.yaml")
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_vllm_config(
    config_path: str = None,
    encoding: str = "utf-8",
) -> dict:
    """加载 vLLM 服务运行参数配置。

    仅在 backend: local 时使用。返回包含 chat / embedding 两个子 dict 的字典。
    """
    if config_path is None:
        config_path = get_abs_path(project_root_path + "/config/vllm.yaml")
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# ---------------------------------------------------------------------------
# Embedding 维度一致性检查
# ---------------------------------------------------------------------------

def _get_chroma_meta_path(persist_dir: str) -> str:
    """返回 ChromaDB 元数据文件路径。"""
    return os.path.join(persist_dir, _CHROMA_META_FILENAME)


def write_chroma_metadata(persist_dir: str, embedding_model: str, dimension: int) -> None:
    """写入 ChromaDB Embedding 元数据（在构建/重建向量库后调用）。"""
    meta = {
        "embedding_model": embedding_model,
        "embedding_dimension": dimension,
    }
    meta_path = _get_chroma_meta_path(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    from utils.logger_handler import logger
    logger.info(f"[Chroma 元数据] 已写入: model={embedding_model}, dim={dimension}")


def check_chroma_embedding_compatibility(
    persist_dir: str,
    expected_dim: int,
    expected_model: str = "",
) -> bool:
    """检查已有 ChromaDB 的 embedding 维度是否与当前配置一致。

    检查策略（按优先级）：
      1. 读取 .equimind_embedding_meta.json（首选，可靠）
      2. 回退：读取 chroma.sqlite3 的 embeddings 表第一行，通过 blob 长度推断维度
      3. 若均无法判断（空库），跳过检查

    Returns:
        True 如果兼容（或无法判断），False 如果维度不匹配需要重建。
    """
    # 策略 1：元数据文件
    meta_path = _get_chroma_meta_path(persist_dir)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            actual_dim = meta.get("embedding_dimension")
            actual_model = meta.get("embedding_model", "")
            if actual_dim is not None and actual_dim != expected_dim:
                from utils.logger_handler import logger
                logger.warning(
                    f"[Chroma 兼容性] ⚠️ 嵌入维度不匹配！"
                    f" 已有: {actual_dim} (模型: {actual_model}), "
                    f" 当前配置: {expected_dim} (模型: {expected_model})"
                )
                return False
            # 维度相同但模型不同 → 同维度模型可兼容，仅提示
            if actual_model and expected_model and actual_model != expected_model:
                from utils.logger_handler import logger
                logger.info(
                    f"[Chroma 兼容性] 嵌入模型已切换 ({actual_model} → {expected_model})，"
                    f"但维度相同 ({expected_dim})，向量库可复用"
                )
            return True
        except Exception:
            pass  # 元数据损坏，回退到策略 2

    # 策略 2：从 SQLite 推断
    sqlite_path = os.path.join(persist_dir, "chroma.sqlite3")
    if os.path.exists(sqlite_path):
        try:
            conn = sqlite3.connect(sqlite_path)
            cursor = conn.execute(
                "SELECT embedding FROM embeddings LIMIT 1"
            )
            row = cursor.fetchone()
            conn.close()
            if row and row[0]:
                # embedding 是 float32 数组，每个元素 4 字节
                blob_len = len(row[0])
                if blob_len % 4 == 0:
                    actual_dim = blob_len // 4
                    if actual_dim != expected_dim:
                        from utils.logger_handler import logger
                        logger.warning(
                            f"[Chroma 兼容性] ⚠️ 嵌入维度不匹配！"
                            f" 已有向量库: {actual_dim} 维, "
                            f" 当前配置: {expected_dim} 维。"
                            f" 请重建向量库：删除 {persist_dir} 后重新摄入文档。"
                        )
                        return False
        except Exception:
            pass  # 无法判断，跳过

    return True


# 模块级配置实例（保持向后兼容）
rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
agent_conf = load_agent_config()
prompts_conf = load_prompts_config()

# 启动时检查 ChromaDB 嵌入维度兼容性
_check_ok = check_chroma_embedding_compatibility(
    persist_dir=os.path.join(project_root_path, chroma_conf.get("persist_directory", "rag/chroma_db")),
    expected_dim=chroma_conf.get("embedding_dimension", 1024),
    expected_model=rag_conf.get("embedding_model_name", ""),
)
if not _check_ok:
    import sys as _sys
    _sys.stderr.write(
        "[Chroma 兼容性] ⚠️ 向量库维度与当前 Embedding 模型不匹配！\n"
        "  请删除旧向量库后重新摄入：rm -rf backend/rag/chroma_db\n"
        "  或切回原来的 Embedding 模型。\n"
    )

if __name__ == "__main__":
    print(f"Backend: {rag_conf.get('backend', 'dashscope (default)')}")
    print(f"Chat model: {rag_conf['chat_model_name']}")
    print(f"Chat base URL: {rag_conf['chat_api_base']}")
    print(f"Embedding model: {rag_conf['embedding_model_name']}")
    print(f"Embedding provider: {rag_conf.get('embedding_provider', 'N/A')}")

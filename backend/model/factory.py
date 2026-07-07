"""
模型工厂 — 多后端 Chat / Embedding 模型构建

支持的后端（由 rag.yaml 的 backend 字段控制）：
  - dashscope : 阿里云 DashScope API
  - local     : 本地 vLLM 推理服务
  - openai_compatible : 其他 OpenAI 兼容 API
"""
import os
import urllib.request
import urllib.error
import json
from abc import ABC, abstractmethod
from typing import Optional

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils.config_handler import rag_conf
from utils.logger_handler import logger


def _resolve_api_key(value: str) -> str | None:
    """先当环境变量名解析，解析不到则把 value 本身当 API Key 使用。"""
    env_val = os.getenv(value)
    if env_val:
        return env_val
    return value


def _check_vllm_health(base_url: str, timeout: float = 3.0) -> bool:
    """检测 vLLM 服务是否就绪。

    Args:
        base_url: vLLM OpenAI 兼容 API 地址（如 http://localhost:8002/v1）
        timeout: 请求超时秒数

    Returns:
        True 如果 /v1/models 返回 200
    """
    url = base_url.rstrip("/") + "/models"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                models = [m["id"] for m in data.get("data", [])]
                logger.info(f"[vLLM 健康检查] {base_url} 就绪，可用模型: {models}")
                return True
    except urllib.error.URLError as e:
        logger.warning(f"[vLLM 健康检查] {base_url} 不可达: {e.reason}")
    except Exception as e:
        logger.warning(f"[vLLM 健康检查] {base_url} 异常: {e}")
    return False


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | ChatOpenAI]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | ChatOpenAI]:
        backend = rag_conf.get("backend", "dashscope")
        logger.info(f"[ChatModelFactory] 当前后端: {backend}")

        if backend == "local":
            return self._create_local()
        elif backend == "openai_compatible":
            return self._create_openai_compat()
        else:
            return self._create_dashscope()

    def _create_local(self) -> ChatOpenAI:
        """本地 vLLM 推理服务。

        vLLM 通过 OpenAI 兼容 API 暴露，直接用 ChatOpenAI 连接。
        Qwen3 的 thinking 模式由 chat_template 自动处理，不需要 extra_body。

        支持通过 rag.yaml 的 chat_model_kwargs 传入额外参数（temperature 等）。
        """
        base_url = rag_conf.get("chat_api_base", "http://localhost:8002/v1")
        api_key = _resolve_api_key(rag_conf.get("chat_api_key_env", "not-needed"))

        # 启动自检（非阻塞：失败只 warn，不阻断创建）
        _check_vllm_health(base_url)

        # 允许通过配置传入额外的 ChatOpenAI 参数（temperature, top_p 等）
        extra_kwargs = dict(rag_conf.get("chat_model_kwargs", {}))

        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            base_url=base_url,
            api_key=api_key,
            **extra_kwargs,
        )

    def _create_dashscope(self) -> ChatOpenAI:
        """阿里云 DashScope 云端 API。

        enable_thinking 是 DeepSeek 专有参数，启用后返回 reasoning_content。
        """
        api_key = _resolve_api_key(rag_conf["chat_api_key_env"])
        kwargs = {
            "model": rag_conf["chat_model_name"],
            "base_url": rag_conf["chat_api_base"],
        }
        if api_key:
            kwargs["api_key"] = api_key
        if rag_conf.get("enable_thinking"):
            kwargs["extra_body"] = {"enable_thinking": True}
        return ChatOpenAI(**kwargs)

    def _create_openai_compat(self) -> ChatOpenAI:
        """通用 OpenAI 兼容 API（如 OneAPI、OpenRouter 等）。"""
        api_key = _resolve_api_key(rag_conf.get("chat_api_key_env", ""))
        kwargs = {
            "model": rag_conf["chat_model_name"],
            "base_url": rag_conf["chat_api_base"],
        }
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | ChatOpenAI]:
        backend = rag_conf.get("backend", "dashscope")
        logger.info(f"[EmbeddingsFactory] 当前后端: {backend}")

        if backend == "local":
            return self._create_local()
        elif backend == "openai_compatible":
            return self._create_openai_compat()
        else:
            return self._create_dashscope()

    def _create_local(self) -> OpenAIEmbeddings:
        """本地 vLLM Embedding 服务。"""
        base_url = rag_conf.get("embedding_api_base", "http://localhost:8003/v1")
        api_key = _resolve_api_key(rag_conf.get("embedding_api_key_env", "not-needed"))

        _check_vllm_health(base_url)

        return OpenAIEmbeddings(
            model=rag_conf["embedding_model_name"],
            base_url=base_url,
            api_key=api_key,
        )

    def _create_dashscope(self) -> DashScopeEmbeddings:
        """阿里云 DashScope Embedding 服务。"""
        api_key = _resolve_api_key(rag_conf.get("embedding_api_key_env", ""))
        kwargs_ds = {"model": rag_conf["embedding_model_name"]}
        if api_key:
            kwargs_ds["dashscope_api_key"] = api_key
        return DashScopeEmbeddings(**kwargs_ds)

    def _create_openai_compat(self) -> OpenAIEmbeddings:
        """通用 OpenAI 兼容 Embedding 服务。"""
        api_key = _resolve_api_key(rag_conf.get("embedding_api_key_env", ""))
        kwargs = {
            "model": rag_conf["embedding_model_name"],
            "base_url": rag_conf["embedding_api_base"],
        }
        if api_key:
            kwargs["api_key"] = api_key
        return OpenAIEmbeddings(**kwargs)


# 模块级模型实例（保持向后兼容）
chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingsFactory().generator()

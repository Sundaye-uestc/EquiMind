import os
from abc import ABC, abstractmethod
from typing import Optional

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils.config_handler import rag_conf


def _resolve_api_key(value: str) -> str | None:
    """先当环境变量名解析，解析不到则把 value 本身当 API Key 使用。"""
    env_val = os.getenv(value)
    if env_val:
        return env_val
    return value


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | ChatOpenAI]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | ChatOpenAI]:
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


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | ChatOpenAI]:
        provider = rag_conf.get("embedding_provider", "dashscope")
        if provider == "dashscope":
            api_key = _resolve_api_key(rag_conf.get("embedding_api_key_env", ""))
            kwargs_ds = {"model": rag_conf["embedding_model_name"]}
            if api_key:
                kwargs_ds["dashscope_api_key"] = api_key
            return DashScopeEmbeddings(**kwargs_ds)
        # openai_compatible 分支（智谱等）
        api_key = _resolve_api_key(rag_conf.get("embedding_api_key_env", ""))
        kwargs = {
            "model": rag_conf["embedding_model_name"],
            "base_url": rag_conf["embedding_api_base"],
        }
        if api_key:
            kwargs["api_key"] = api_key
        return OpenAIEmbeddings(**kwargs)


chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingsFactory().generator()
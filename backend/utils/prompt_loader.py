"""
Prompt 加载模块 — 支持按后端自动切换提示词模板

本地 vLLM (Qwen3) 使用精简版提示词（仅描述实际注册的工具），
以避免模型试图调用不存在的工具导致幻觉。
"""
import os

from .config_handler import prompts_conf, rag_conf
from .logger_handler import logger


project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Qwen3 后端使用的主提示词文件名
_QWEN3_MAIN_PROMPT = "main_prompt_qwen3.txt"


def _get_main_prompt_filename() -> str:
    """根据当前后端返回主提示词文件名。

    Qwen3 本地后端使用精简版提示词（仅列出实际工具），
    DashScope 使用完整版（含领域能力描述）。
    """
    backend = rag_conf.get("backend", "dashscope")
    if backend == "local":
        return _QWEN3_MAIN_PROMPT
    return "main_prompt.txt"


def load_system_prompts():
    """加载主系统提示词（自动按后端选择）。"""
    filename = _get_main_prompt_filename()
    try:
        prompt_path = os.path.join(project_root_path, "prompts", filename)
    except KeyError as e:
        logger.error(f"[load_system_prompts] prompts.yaml 缺少 main_prompt_path 配置")
        raise e

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"[load_system_prompts] 已加载: {filename} (backend={rag_conf.get('backend', 'dashscope')})")
        return content
    except Exception as e:
        logger.error(f"[load_system_prompts] 解析系统提示词出错：{str(e)}")
        raise e


def load_rag_prompts():
    """加载 RAG 总结提示词。"""
    try:
        rag_prompt_path = project_root_path + "/" + prompts_conf["rag_summarize_prompt_path"]
    except KeyError as e:
        logger.error(f"[load_rag_prompts] 在yaml配置项中没有rag_summarize_prompt_path配置项")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts] 解析RAG总结提示词出错：{str(e)}")
        raise e


def load_report_prompts():
    """加载报告生成提示词。"""
    try:
        report_prompt_path = project_root_path + "/" + prompts_conf["report_prompt_path"]
    except KeyError as e:
        logger.error(f"[load_report_prompts] 在yaml配置项中没有report_prompt_path配置项")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts] 解析报告生成提示词出错：{str(e)}")
        raise e

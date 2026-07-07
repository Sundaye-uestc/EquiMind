"""
训练数据采集模块 — 捕获 Agent 完整对话链路用于微调数据构建

记录内容：
  - 用户原始问题
  - 系统提示词
  - Agent 每一步的 thought / action / observation
  - 每次工具调用的名称 + 参数 + 返回值
  - 最终输出

输出格式：JSONL（OpenAI messages 兼容）
"""
import os
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from .logger_handler import logger

# 默认落盘路径（相对于项目根目录）
_DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent / "logs" / "training_data"


class TrainingDataRecorder:
    """单次对话的训练数据记录器。

    在 Agent 流式执行过程中逐步累积数据，最后 flush 到 JSONL 文件。
    线程安全：每个对话创建独立实例，不跨线程共享。
    """

    def __init__(self, query: str, system_prompt: str, backend: str, model: str):
        self.query = query
        self.system_prompt = system_prompt
        self.backend = backend
        self.model = model

        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        self.tool_calls: list[dict] = []
        self.final_output = ""
        self.start_time = time.perf_counter()
        self.issues: list[str] = []

    def record_tool_call(self, name: str, args: dict, result: str):
        """记录一次工具调用（由 middleware 回调）。"""
        call_record = {
            "name": name,
            "arguments": args,
            "result": str(result)[:2000],  # 截断过长结果
            "timestamp": datetime.now().isoformat(),
        }
        self.tool_calls.append(call_record)

        # 追加到 messages（OpenAI 训练格式）
        tool_call_id = f"call_{len(self.tool_calls):04d}"
        self.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }],
        })
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(result)[:2000],
        })

    def record_assistant_message(self, content: str):
        """记录模型的纯文本回复（非工具调用）。"""
        self.messages.append({
            "role": "assistant",
            "content": content,
        })

    def set_final_output(self, output: str):
        """设置最终完整输出。"""
        self.final_output = output

    def add_issue(self, issue: str):
        """记录异常/问题。"""
        self.issues.append(issue)

    def to_record(self) -> dict:
        """导出为完整的训练数据记录。"""
        elapsed = time.perf_counter() - self.start_time
        return {
            "messages": self.messages,
            "metadata": {
                "backend": self.backend,
                "model": self.model,
                "tool_calls_count": len(self.tool_calls),
                "tool_calls": self.tool_calls,
                "total_time_s": round(elapsed, 2),
                "issues": self.issues,
                "timestamp": datetime.now().isoformat(),
            },
        }


class TrainingDataWriter:
    """训练数据写入器（线程安全）。"""

    _lock = threading.Lock()
    _output_dir: Optional[Path] = None

    @classmethod
    def set_output_dir(cls, path: str):
        cls._output_dir = Path(path)
        cls._output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_output_dir(cls) -> Path:
        if cls._output_dir is None:
            cls._output_dir = _DEFAULT_DIR
            cls._output_dir.mkdir(parents=True, exist_ok=True)
        return cls._output_dir

    @classmethod
    def flush(cls, record: dict):
        """将一条训练记录写入 JSONL 文件。"""
        output_dir = cls.get_output_dir()
        filename = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H')}.jsonl"

        with cls._lock:
            with open(filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.debug(f"[TrainingDataWriter] 已写入 {filename}")


# 全局开关
_recording_enabled = True


def set_recording(enabled: bool):
    """控制是否启用训练数据录制。"""
    global _recording_enabled
    _recording_enabled = enabled


def is_recording() -> bool:
    return _recording_enabled


def flush_training_record(recorder: TrainingDataRecorder):
    """完成一次对话录制并写入磁盘。"""
    if not _recording_enabled:
        return
    if not recorder.final_output:
        recorder.add_issue("no_final_output")

    # 如果最后一条消息不是 assistant 且没有工具调用，添加 final_output
    last_msg = recorder.messages[-1] if recorder.messages else None
    if last_msg and last_msg.get("role") == "tool":
        recorder.record_assistant_message(recorder.final_output)

    record = recorder.to_record()
    TrainingDataWriter.flush(record)
    logger.info(
        f"[training_logger] 已录制对话: {len(recorder.messages)} 条消息, "
        f"{len(recorder.tool_calls)} 次工具调用, "
        f"耗时 {record['metadata']['total_time_s']}s"
    )

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent

from agent.tools.agent_tools import *
from agent.tools.middleware import *
from model.factory import chat_model
from utils.config_handler import rag_conf
from utils.prompt_loader import load_system_prompts as _load_sys_prompt
from utils.training_logger import (
    TrainingDataRecorder,
    flush_training_record,
    is_recording,
)
from utils.logger_handler import logger


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_city, get_user_id,
                   get_month, get_gender, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str):
        """流式执行 Agent，可选录制训练数据。

        训练数据录制由 training_logger.is_recording() 全局开关控制。
        """
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # ---- 训练数据录制初始化 ----
        recorder = None
        if is_recording():
            recorder = TrainingDataRecorder(
                query=query,
                system_prompt=_load_sys_prompt(),
                backend=rag_conf.get("backend", "dashscope"),
                model=rag_conf.get("chat_model_name", "unknown"),
            )

        full_output = []

        # 第三个参数 context 即 runtime 上下文，用于提示词切换标记
        for chunk in self.agent.stream(
            input_dict, stream_mode="values", context={"report": False},
        ):
            latest_message = chunk["messages"][-1]

            # ---- 捕获工具调用 ----
            if recorder and hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                for tc in latest_message.tool_calls:
                    recorder.record_tool_call(
                        name=tc.get("name", "unknown"),
                        args=tc.get("args", {}),
                        result="",  # 结果在下一条 ToolMessage 中
                    )

            # ---- 捕获工具返回结果 ----
            if recorder and latest_message.type == "tool":
                # 补充上一条工具调用的结果
                tool_content = latest_message.content or ""
                if recorder.tool_calls:
                    recorder.tool_calls[-1]["result"] = str(tool_content)[:2000]
                    # 更新 messages 中的 tool 消息
                    for i in range(len(recorder.messages) - 1, -1, -1):
                        if recorder.messages[i]["role"] == "tool":
                            recorder.messages[i]["content"] = str(tool_content)[:2000]
                            break

            # ---- 流式输出 ----
            if latest_message.content:
                content = latest_message.content.strip()
                full_output.append(content)
                yield content + "\n"

        # ---- 完成录制 ----
        final_text = "\n".join(full_output)
        if recorder:
            recorder.set_final_output(final_text)
            # 如果最后没有 assistant 消息，补充一个
            has_assistant = any(
                m["role"] == "assistant" and m.get("content")
                for m in recorder.messages
            )
            if not has_assistant and final_text:
                recorder.record_assistant_message(final_text)
            flush_training_record(recorder)

    def execute_stream_with_recording(self, query: str):
        """显式开启训练数据录制的流式执行（不受全局开关控制）。"""
        from utils.training_logger import set_recording
        prev = is_recording()
        set_recording(True)
        try:
            yield from self.execute_stream(query)
        finally:
            set_recording(prev)


if __name__ == "__main__":
    agent = ReactAgent()
    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)

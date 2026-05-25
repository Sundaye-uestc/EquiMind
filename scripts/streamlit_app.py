import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import streamlit as st
import streamlit.components.v1 as components

from agent.react_agent import ReactAgent

st.set_page_config(
    page_title="航空、交通领域边缘侧大模型智能系统",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
/* ===== 全局变量 ===== */
:root {
    --primary: #667eea;
    --primary-dark: #764ba2;
    --bg: #f5f7fb;
    --panel: #ffffff;
    --text: #162238;
    --muted: #4b5a73;
    --line: #dce3f1;
}

/* ===== 页面背景 ===== */
.stApp {
    background: radial-gradient(circle at 20% 15%, #f0f3ff 0%, transparent 30%),
                radial-gradient(circle at 80% 25%, #f8f0ff 0%, transparent 35%),
                var(--bg);
}

/* ===== 主容器 ===== */
.stMainBlockContainer {
    max-width: 900px !important;
    padding-top: 0 !important;
}

/* ===== 顶部标题栏 ===== */
[data-testid="stAppViewContainer"] > .stMainBlockContainer > div:first-child {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 24px 32px !important;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
}
[data-testid="stAppViewContainer"] > .stMainBlockContainer > div:first-child h1 {
    color: #fff !important;
    font-size: 26px !important;
    padding: 0 !important;
    margin: 0 0 4px 0 !important;
}
[data-testid="stAppViewContainer"] > .stMainBlockContainer > div:first-child p {
    color: rgba(255,255,255,0.85) !important;
    font-size: 14px !important;
}

/* ===== 标题下方的分隔线隐藏 ===== */
hr {
    display: none;
}

/* ===== 聊天消息容器 ===== */
[data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}

/* ===== 用户消息气泡 ===== */
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
    background: #667eea !important;
    color: #fff !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 12px 18px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}

/* ===== 助手消息气泡 ===== */
[data-testid="stChatMessage"][data-testid="stChatMessageIconAssistant"] ~ div [data-testid="stChatMessageContent"],
div:has(> [data-testid="stChatMessageIconAssistant"]) + div [data-testid="stChatMessageContent"] {
    background: #f0f0f0 !important;
    color: #333 !important;
    border-radius: 12px !important;
    padding: 12px 18px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}
/* 通用：非用户消息都用灰色背景 */
[data-testid="stChatMessage"] {
    background: transparent !important;
}

/* ===== 展开器（思考过程）样式 ===== */
[data-testid="stExpander"] {
    background: #fafafa !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    color: #667eea !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 14px !important;
}
[data-testid="stExpander"] summary:hover {
    background: #f0f3ff !important;
}
.thinking-text {
    font-style: italic;
    color: #888888;
    font-size: 14px;
    line-height: 1.6;
    padding: 8px 14px;
}

/* ===== 输入框 ===== */
[data-testid="stChatInput"] textarea {
    border: 1px solid #ddd !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
    background: #fbfcfe !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.12) !important;
    outline: none !important;
}
[data-testid="stChatInput"] {
    position: sticky !important;
    bottom: 0 !important;
    background: #fff !important;
    padding: 16px 0 !important;
    border-top: 1px solid #e0e0e0 !important;
}

/* ===== 按钮样式 ===== */
button[kind="secondary"] {
    background: #fff !important;
    color: #667eea !important;
    border: 1px solid #667eea !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.3s !important;
}
button[kind="secondary"]:hover {
    background: #f0f3ff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2) !important;
}
button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.25) !important;
}
button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.35) !important;
}

/* ===== Spinner ===== */
[data-testid="stSpinner"] {
    color: #667eea !important;
}

/* ===== 滚动条 ===== */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: #d0d5e0;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #a0a8c0;
}
</style>
""", unsafe_allow_html=True)

# 渐变标题
st.markdown("""
<div style="text-align:center;">
    <h1 style="color:#fff;font-size:26px;margin:0 0 4px;">航空、交通领域边缘侧大模型智能系统</h1>
    <p style="color:rgba(255,255,255,0.85);font-size:14px;margin:0;">基于 LangGraph + Agentic RAG 架构</p>
</div>
""", unsafe_allow_html=True)

def scroll_to_bottom():
    # 方式1：st.markdown 注入脚本（脚本在 Streamlit app 的 DOM 内执行）
    st.markdown(
        """
        <script>
            (function() {
                var el = document.querySelector('.main') ||
                         document.querySelector('[data-testid="stAppViewContainer"]') ||
                         document.querySelector('.stApp');
                if (el) { el.scrollTop = el.scrollHeight; }
                var m = document.getElementById('bottom-marker');
                if (m) { m.scrollIntoView({block: 'end'}); }
            })();
        </script>
        """,
        unsafe_allow_html=True,
    )
    # 方式2：components.html 注入脚本（从子 iframe 访问父级 Streamlit app DOM）
    components.html(
        """
        <script>
            (function() {
                var doc = window.parent.document;
                var el = doc.querySelector('.main') ||
                         doc.querySelector('[data-testid="stAppViewContainer"]') ||
                         doc.querySelector('.stApp');
                if (el) { el.scrollTop = el.scrollHeight; }
                var m = doc.getElementById('bottom-marker');
                if (m) { m.scrollIntoView({block: 'end'}); }
            })();
        </script>
        """,
        height=0,
        width=0,
    )

def stop_generation():
    st.session_state["_stop_requested"] = True

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

# 处理中止生成：将已流式输出的部分内容作为截断回答保存
if st.session_state.get("_stop_requested") and st.session_state.get("_stream_chunks") is not None:
    chunks = st.session_state["_stream_chunks"]
    if chunks:
        if len(chunks) > 1:
            thinking = "".join(chunks[:-1])
            answer = chunks[-1]
        else:
            thinking = ""
            answer = chunks[0]
        answer += "\n\n*[已中止]*"
    else:
        thinking = ""
        answer = "*[已中止]*"

    st.session_state["message"].append({
        "role": "assistant",
        "content": answer,
        "thinking": thinking,
    })
    # 清理临时状态
    st.session_state["_stop_requested"] = False
    st.session_state["_generating"] = False
    st.session_state.pop("_stream_chunks", None)
    scroll_to_bottom()

# 渲染历史消息
for msg in st.session_state["message"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        thinking = msg.get("thinking", "")
        with st.chat_message("assistant"):
            if thinking:
                with st.expander("💭 查看思考过程", expanded=False):
                    st.markdown(
                        f'<div class="thinking-text">{thinking}</div>',
                        unsafe_allow_html=True,
                    )
            st.write(msg["content"])

# 当有历史消息时，页面自动滚动到底部
if st.session_state["message"]:
    scroll_to_bottom()

st.markdown('<div id="bottom-marker"></div>', unsafe_allow_html=True)

# 中止按钮：仅在模型生成过程中显示
if st.session_state.get("_generating", False):
    c1, c2, c3 = st.columns([3, 1, 3])
    with c2:
        st.button("⏹ 中止生成", on_click=stop_generation, key="stop_btn", use_container_width=True)

prompt = st.chat_input()

if prompt:
    st.session_state["_stop_requested"] = False
    st.session_state["_generating"] = True
    st.session_state["_stream_chunks"] = []

    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})
    scroll_to_bottom()

    all_chunks = []

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            stream_placeholder = st.empty()
            res_stream = st.session_state["agent"].execute_stream(prompt)

            for chunk in res_stream:
                all_chunks.append(chunk)
                st.session_state["_stream_chunks"] = all_chunks
                cumulative = "".join(all_chunks)
                stream_placeholder.markdown(
                    f'<div class="thinking-text">{cumulative}</div>',
                    unsafe_allow_html=True,
                )

        stream_placeholder.empty()

        # 分离思考过程与最终答案
        if len(all_chunks) > 1:
            thinking = "".join(all_chunks[:-1])
            answer = all_chunks[-1]
        elif len(all_chunks) == 1:
            thinking = ""
            answer = all_chunks[0]
        else:
            thinking = ""
            answer = ""

        # 思考过程放入折叠框
        if thinking:
            with st.expander("💭 查看思考过程", expanded=False):
                st.markdown(
                    f'<div class="thinking-text">{thinking}</div>',
                    unsafe_allow_html=True,
                )

        # 最终答案逐字符流式输出
        answer_placeholder = st.empty()
        displayed = ""
        for char in answer:
            displayed += char
            answer_placeholder.markdown(displayed)
            time.sleep(0.01)

    st.session_state["_generating"] = False
    st.session_state.pop("_stream_chunks", None)

    st.session_state["message"].append({
        "role": "assistant",
        "content": answer,
        "thinking": thinking,
    })
    st.rerun()

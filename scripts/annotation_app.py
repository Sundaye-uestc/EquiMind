#!/usr/bin/env python
"""
EquiMind 微调数据标注工具 (Streamlit)

功能：
  - 浏览训练日志中的所有对话记录
  - 标注：工具调用是否正确、最终诊断是否合理
  - 手动修正工具调用序列和回复
  - 导出已标注数据为训练格式（OpenAI messages）

Usage:
  streamlit run scripts/annotation_app.py -- --input logs/training_data/
"""
import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

try:
    import streamlit as st
except ImportError:
    print("请先安装 streamlit: pip install streamlit")
    sys.exit(1)


def load_records(input_dir: str) -> list[dict]:
    records = []
    input_path = Path(input_dir)
    if input_path.is_file():
        jsonl_files = [input_path]
    else:
        jsonl_files = sorted(input_path.glob("training_*.jsonl"))
    for fp in jsonl_files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def render_message(msg: dict):
    """渲染单条消息。"""
    role = msg.get("role", "unknown")
    content = msg.get("content", "")

    role_colors = {
        "system": "gray",
        "user": "blue",
        "assistant": "green",
        "tool": "orange",
    }
    color = role_colors.get(role, "black")

    with st.chat_message(role):
        if content:
            st.markdown(f":{color}[**{role.upper()}**]\n\n{content}")
        if msg.get("tool_calls"):
            st.markdown(f":{color}[**TOOL CALLS**]")
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                st.code(
                    f"{func.get('name', '?')}(\n  {func.get('arguments', '{}')}\n)",
                    language="python",
                )


def render_annotation_form(record_index: int, record: dict):
    """渲染标注表单。"""
    meta = record.get("metadata", {})

    st.subheader(f"记录 #{record_index}")
    st.caption(
        f"后端: {meta.get('backend', '?')} | "
        f"模型: {meta.get('model', '?')} | "
        f"工具调用: {meta.get('tool_calls_count', 0)} 次 | "
        f"耗时: {meta.get('total_time_s', '?')}s"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        tool_correct = st.radio(
            "工具调用是否正确？",
            ["未评估", "正确", "部分正确", "错误", "不适用"],
            key=f"tool_{record_index}",
        )
    with col2:
        diagnosis_ok = st.radio(
            "最终诊断是否合理？",
            ["未评估", "合理", "部分合理", "不合理", "不适用"],
            key=f"diag_{record_index}",
        )
    with col3:
        hallucination = st.radio(
            "是否存在幻觉？",
            ["未评估", "无幻觉", "轻微幻觉", "严重幻觉"],
            key=f"hallu_{record_index}",
        )

    notes = st.text_area("标注备注", key=f"notes_{record_index}")
    manual_fix = st.text_area(
        "手动修正回复（留空则保留原始回复）",
        key=f"fix_{record_index}",
        height=100,
    )

    return {
        "tool_correct": tool_correct,
        "diagnosis_ok": diagnosis_ok,
        "hallucination": hallucination,
        "notes": notes,
        "manual_fix": manual_fix,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="训练日志目录")
    args = parser.parse_args()

    st.set_page_config(page_title="EquiMind 训练数据标注", layout="wide")
    st.title("EquiMind 微调数据标注工具")

    records = load_records(args.input)
    if not records:
        st.error(f"未找到训练数据: {args.input}")
        return

    st.sidebar.metric("总记录数", len(records))

    # 筛选器
    backend_filter = st.sidebar.selectbox(
        "后端筛选",
        ["全部"] + sorted(set(
            r.get("metadata", {}).get("backend", "?") for r in records
        )),
    )
    tool_filter = st.sidebar.slider(
        "最少工具调用次数", 0, 10, 0,
    )

    # 过滤
    filtered = records
    if backend_filter != "全部":
        filtered = [r for r in filtered
                    if r.get("metadata", {}).get("backend") == backend_filter]
    if tool_filter > 0:
        filtered = [r for r in filtered
                    if r.get("metadata", {}).get("tool_calls_count", 0) >= tool_filter]

    st.sidebar.metric("筛选后", len(filtered))

    # 分页
    page_size = st.sidebar.number_input("每页条数", 1, 50, 5)
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page = st.sidebar.number_input("页码", 1, total_pages, 1) - 1

    # 初始化 session state
    if "annotations" not in st.session_state:
        st.session_state.annotations = {}

    # 显示当前页记录
    start = page * page_size
    end = min(start + page_size, len(filtered))
    current_records = filtered[start:end]

    for i, rec in enumerate(current_records):
        real_idx = start + i
        with st.expander(
            f"#{real_idx}: {rec.get('metadata', {}).get('tool_calls_count', 0)} 工具调用 | "
            f"{rec.get('messages', [{}])[1].get('content', '?')[:60] if len(rec.get('messages', [])) > 1 else '?'}...",
            expanded=(len(current_records) <= 2),
        ):
            # 显示对话
            for msg in rec.get("messages", []):
                render_message(msg)

            # 标注表单
            annotation = render_annotation_form(real_idx, rec)
            if st.button(f"保存标注 #{real_idx}", key=f"save_{real_idx}"):
                st.session_state.annotations[real_idx] = annotation
                st.success(f"已保存标注 #{real_idx}")

    # 导出
    st.sidebar.divider()
    if st.sidebar.button("导出已标注数据", type="primary"):
        if not st.session_state.annotations:
            st.sidebar.warning("尚未标注任何记录")
        else:
            output_path = Path(args.input) / f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            exported = 0
            with open(output_path, "w", encoding="utf-8") as f:
                for idx, ann in st.session_state.annotations.items():
                    rec = records[idx]
                    # 如果手动修正了回复，更新 messages
                    if ann.get("manual_fix"):
                        for msg in reversed(rec.get("messages", [])):
                            if msg["role"] == "assistant" and msg.get("content"):
                                msg["content"] = ann["manual_fix"]
                                break

                    export_rec = {
                        "messages": rec.get("messages", []),
                        "metadata": {
                            **rec.get("metadata", {}),
                            "annotation": ann,
                            "annotated_at": datetime.now().isoformat(),
                        },
                    }
                    f.write(json.dumps(export_rec, ensure_ascii=False) + "\n")
                    exported += 1
            st.sidebar.success(f"已导出 {exported} 条标注数据 -> {output_path}")


if __name__ == "__main__":
    main()

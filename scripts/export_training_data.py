#!/usr/bin/env python
"""
EquiMind 训练数据导出脚本

从训练日志 JSONL 中读取原始对话记录，导出为多种微调格式：
  - OpenAI messages 格式（完整对话链）
  - 工具调用专项格式（仅含工具调用的对话）
  - 报告生成格式（仅报告类对话）

Usage:
  # 导出全部数据为 OpenAI 格式
  python scripts/export_training_data.py --input logs/training_data/ --output training_openai.jsonl

  # 仅导出工具调用数据
  python scripts/export_training_data.py --input logs/training_data/ --output training_tools.jsonl --mode tools_only

  # 仅导出报告生成数据
  python scripts/export_training_data.py --input logs/training_data/ --output training_reports.jsonl --mode reports_only

  # 统计摘要
  python scripts/export_training_data.py --input logs/training_data/ --stats
"""
import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_records(input_dir: str) -> list[dict]:
    """加载所有 JSONL 训练记录。"""
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
                    except json.JSONDecodeError as e:
                        print(f"  [WARN] 跳过损坏记录: {fp} — {e}")

    return records


def export_openai(records: list[dict], output_path: str):
    """导出为 OpenAI messages 格式。"""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            msgs = rec.get("messages", [])
            if not msgs:
                continue
            # 确保有 user + assistant
            has_user = any(m["role"] == "user" for m in msgs)
            has_assistant = any(m["role"] == "assistant" for m in msgs)
            if not (has_user and has_assistant):
                continue
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            count += 1
    print(f"  导出 OpenAI 格式: {count} 条 -> {output_path}")


def export_tools_only(records: list[dict], output_path: str):
    """仅导出包含工具调用的对话。"""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            meta = rec.get("metadata", {})
            if meta.get("tool_calls_count", 0) == 0:
                continue
            msgs = rec.get("messages", [])
            # 截断到第一个 tool 消息后的 assistant 回复
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            count += 1
    print(f"  导出工具调用格式: {count} 条 -> {output_path}")


def export_reports_only(records: list[dict], output_path: str):
    """仅导出报告生成类对话（对话中包含 fill_context_for_report 工具调用）。"""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            msgs = rec.get("messages", [])
            has_report_tool = any(
                "fill_context_for_report" in json.dumps(m, ensure_ascii=False)
                for m in msgs
            )
            if not has_report_tool:
                continue
            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            count += 1
    print(f"  导出报告生成格式: {count} 条 -> {output_path}")


def print_stats(records: list[dict]):
    """打印训练数据统计摘要。"""
    print(f"\n{'='*50}")
    print(f"  训练数据统计")
    print(f"{'='*50}")
    print(f"  总记录数: {len(records)}")

    # 按后端/模型分组
    by_model = defaultdict(int)
    by_backend = defaultdict(int)
    tool_call_counts = []
    total_time = 0
    issues_count = 0

    for rec in records:
        meta = rec.get("metadata", {})
        by_model[meta.get("model", "unknown")] += 1
        by_backend[meta.get("backend", "unknown")] += 1
        tc = meta.get("tool_calls_count", 0)
        tool_call_counts.append(tc)
        total_time += meta.get("total_time_s", 0)
        if meta.get("issues"):
            issues_count += 1

    print(f"\n  按后端: {dict(by_backend)}")
    print(f"  按模型: {dict(by_model)}")
    print(f"  工具调用: 总计 {sum(tool_call_counts)}, "
          f"平均 {sum(tool_call_counts)/len(tool_call_counts):.1f}/条"
          if tool_call_counts else "  工具调用: 无")
    print(f"  总耗时: {total_time:.0f}s, 平均 {total_time/len(records):.1f}s/条"
          if records else "")
    print(f"  含异常记录: {issues_count}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="EquiMind 训练数据导出工具"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="训练日志目录或 JSONL 文件路径")
    parser.add_argument("--output", "-o",
                        help="导出文件路径")
    parser.add_argument("--mode", choices=["all", "tools_only", "reports_only"],
                        default="all", help="导出模式（默认 all）")
    parser.add_argument("--stats", action="store_true",
                        help="仅打印统计摘要，不导出")
    args = parser.parse_args()

    print(f"[export] 加载 {args.input} ...")
    records = load_records(args.input)
    print(f"[export] 已加载 {len(records)} 条记录")

    if args.stats:
        print_stats(records)
        return

    if not args.output:
        print("[export] 请指定 --output 导出路径，或使用 --stats 查看统计")
        return

    if args.mode == "tools_only":
        export_tools_only(records, args.output)
    elif args.mode == "reports_only":
        export_reports_only(records, args.output)
    else:
        export_openai(records, args.output)

    print_stats(records)


if __name__ == "__main__":
    main()

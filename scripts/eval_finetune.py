#!/usr/bin/env python
"""
EquiMind 微调评估脚本

基于已标注的测试数据，评估微调前后模型在以下维度的表现：
  - 工具调用准确率（正确时机调用正确工具）
  - 工具参数正确率（参数是否合理）
  - 报告格式遵循率（是否遵循预设模板）
  - 术语使用正确率（工业术语准确度）
  - 幻觉率（事实性错误比例）

Usage:
  # 使用标注数据评估
  python scripts/eval_finetune.py --input logs/training_data/annotated.jsonl

  # 对比两个模型
  python scripts/eval_finetune.py --baseline baseline_results.jsonl --candidate finetuned_results.jsonl
"""
import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_annotated(path: str) -> list[dict]:
    """加载已标注数据。"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class EvalMetrics:
    """评估指标计算器。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0
        self.tool_correct = 0
        self.tool_partial = 0
        self.tool_wrong = 0
        self.diag_ok = 0
        self.diag_partial = 0
        self.diag_wrong = 0
        self.hallu_none = 0
        self.hallu_mild = 0
        self.hallu_severe = 0
        self.not_evaluated = 0

    def add(self, annotation: dict):
        """摄入一条标注。"""
        self.total += 1

        tc = annotation.get("tool_correct", "未评估")
        if tc == "正确":
            self.tool_correct += 1
        elif tc == "部分正确":
            self.tool_partial += 1
        elif tc == "错误":
            self.tool_wrong += 1
        else:
            self.not_evaluated += 1

        diag = annotation.get("diagnosis_ok", "未评估")
        if diag == "合理":
            self.diag_ok += 1
        elif diag == "部分合理":
            self.diag_partial += 1
        elif diag == "不合理":
            self.diag_wrong += 1

        hallu = annotation.get("hallucination", "未评估")
        if hallu == "无幻觉":
            self.hallu_none += 1
        elif hallu == "轻微幻觉":
            self.hallu_mild += 1
        elif hallu == "严重幻觉":
            self.hallu_severe += 1

    def summary(self) -> dict:
        """生成汇总指标。"""
        evaluated = self.total - self.not_evaluated
        if evaluated == 0:
            return {"error": "无已评估样本"}

        return {
            "total": self.total,
            "evaluated": evaluated,
            "tool_accuracy": round(self.tool_correct / evaluated * 100, 1),
            "tool_partial_rate": round(self.tool_partial / evaluated * 100, 1),
            "tool_error_rate": round(self.tool_wrong / evaluated * 100, 1),
            "diagnosis_accuracy": round(self.diag_ok / evaluated * 100, 1),
            "diagnosis_partial_rate": round(self.diag_partial / evaluated * 100, 1),
            "diagnosis_error_rate": round(self.diag_wrong / evaluated * 100, 1),
            "hallucination_free_rate": round(self.hallu_none / evaluated * 100, 1),
            "hallucination_mild_rate": round(self.hallu_mild / evaluated * 100, 1),
            "hallucination_severe_rate": round(self.hallu_severe / evaluated * 100, 1),
        }


def compare_models(baseline_metrics: dict, candidate_metrics: dict) -> str:
    """生成对比报告。"""
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  微调前后对比")
    lines.append("=" * 60)

    metrics_to_compare = [
        ("tool_accuracy", "工具调用准确率", "%", "higher"),
        ("diagnosis_accuracy", "诊断合理率", "%", "higher"),
        ("hallucination_free_rate", "无幻觉率", "%", "higher"),
        ("hallucination_severe_rate", "严重幻觉率", "%", "lower"),
    ]

    for key, label, unit, direction in metrics_to_compare:
        base_val = baseline_metrics.get(key, 0)
        cand_val = candidate_metrics.get(key, 0)
        delta = round(cand_val - base_val, 1)
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        direction_ok = (direction == "higher" and delta > 0) or (direction == "lower" and delta < 0)
        mark = "✅" if direction_ok else "⚠️" if delta == 0 else "❌"
        lines.append(
            f"  {mark} {label}: {base_val}{unit} → {cand_val}{unit} "
            f"({arrow}{abs(delta)}{unit})"
        )

    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="EquiMind 微调评估工具")
    parser.add_argument("--input", "-i",
                        help="已标注的 JSONL 文件路径")
    parser.add_argument("--baseline",
                        help="基线模型结果（用于对比）")
    parser.add_argument("--candidate",
                        help="候选模型结果（用于对比）")
    args = parser.parse_args()

    # 对比模式
    if args.baseline and args.candidate:
        baseline = load_annotated(args.baseline)
        candidate = load_annotated(args.candidate)

        bm = EvalMetrics()
        for rec in baseline:
            ann = rec.get("metadata", {}).get("annotation", {})
            bm.add(ann)

        cm = EvalMetrics()
        for rec in candidate:
            ann = rec.get("metadata", {}).get("annotation", {})
            cm.add(ann)

        print("\n--- 基线模型 ---")
        for k, v in bm.summary().items():
            print(f"  {k}: {v}")
        print("\n--- 候选模型 ---")
        for k, v in cm.summary().items():
            print(f"  {k}: {v}")
        print(compare_models(bm.summary(), cm.summary()))
        return

    # 单模型评估
    if args.input:
        records = load_annotated(args.input)
        metrics = EvalMetrics()
        for rec in records:
            ann = rec.get("metadata", {}).get("annotation", {})
            metrics.add(ann)

        print(f"\n{'='*50}")
        print(f"  评估结果: {Path(args.input).name}")
        print(f"{'='*50}")
        for k, v in metrics.summary().items():
            print(f"  {k}: {v}")
        print(f"{'='*50}\n")
        return

    print("请指定 --input 或 --baseline + --candidate")
    print("示例: python scripts/eval_finetune.py --input annotated.jsonl")


if __name__ == "__main__":
    main()

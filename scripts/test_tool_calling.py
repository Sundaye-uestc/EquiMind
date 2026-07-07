#!/usr/bin/env python
"""
EquiMind 工具调用验证脚本

针对 5 大设备类型逐一测试 ReAct Agent 的工具调用能力。
用于 Phase 5 Prompt 适配验证和 Phase 8 微调数据采集。

Usage:
  # 全部测试
  python scripts/test_tool_calling.py

  # 仅测试指定类别
  python scripts/test_tool_calling.py --category aviation
  python scripts/test_tool_calling.py --category railway

  # 采集训练数据（输出 JSONL）
  python scripts/test_tool_calling.py --export training_data.jsonl

  # 交互模式（逐题确认）
  python scripts/test_tool_calling.py --interactive
"""
import sys
import os
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

# 确保 backend 在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from utils.config_handler import rag_conf
from utils.logger_handler import logger

# ============================================================
# 测试用例定义
# ============================================================
TEST_CASES = {
    "aviation": {
        "label": "航空发动机诊断",
        "expected_tools": ["rag_summarize"],
        "cases": [
            "涡扇发动机高压压气机退化有哪些典型传感器表现？",
            "如何根据传感器数据判断涡扇发动机的剩余使用寿命（RUL）？",
            "航空发动机压气机叶片疲劳裂纹的修复方案是什么？",
            "发动机排气温度（EGT）异常升高可能的原因有哪些？",
            "C-MAPSS FD001 数据集中发动机的典型退化曲线是怎样的？",
        ],
    },
    "railway_contacts": {
        "label": "高铁接触网运维",
        "expected_tools": ["rag_summarize"],
        "cases": [
            "高铁接触网实时监测的关键参数包括哪些？",
            "接触网几何参数（拉出值、导高）超限的标准是什么？",
            "高铁接触网常见故障类型及处理方法有哪些？",
            "接触网覆冰对运行安全的影响及处置措施？",
        ],
    },
    "hydropower": {
        "label": "水电机组故障",
        "expected_tools": ["rag_summarize"],
        "cases": [
            "水电机组运行监测的关键指标有哪些？",
            "水电机组转速波动异常的排查思路是什么？",
            "水轮机振动过大的常见原因有哪些？",
            "发电机定子绕组绝缘下降的处理方案？",
        ],
    },
    "railway_track": {
        "label": "铁路轨道监测",
        "expected_tools": ["rag_summarize"],
        "cases": [
            "铁路轨道振动监测中加速度计的典型采样频率是多少？",
            "如何通过加速度数据识别轨道波磨和轨面剥落？",
            "铁路轨道多模态传感器融合分析的方法有哪些？",
            "GPS 轨迹数据在轨道病害定位中如何应用？",
        ],
    },
    "report": {
        "label": "报告生成",
        "expected_tools": ["fill_context_for_report", "get_user_id"],
        "cases": [
            "给我生成我的使用报告",
            "生成设备的近期监测运维报告",
            "帮我生成这个月的设备故障分析报告",
        ],
    },
}


def run_single_test(agent, query: str, expected_tools: list) -> dict:
    """运行单条测试，返回结果记录。

    Returns:
        {
            "query": str,
            "expected_tools": [...],
            "tools_called": [...],
            "response": str,
            "passed": bool,
            "issues": [...],
            "timestamp": str,
        }
    """
    result = {
        "query": query,
        "expected_tools": expected_tools,
        "tools_called": [],
        "response": "",
        "passed": False,
        "issues": [],
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n{'─'*60}")
    print(f"  测试: {query}")
    print(f"  期望工具: {expected_tools}")
    print(f"{'─'*60}")

    try:
        full_response = []
        for chunk in agent.execute_stream(query):
            full_response.append(chunk)
            print(chunk, end="", flush=True)

        result["response"] = "".join(full_response)

        # 检查是否调用了期望的工具
        # 注：LangChain ReAct Agent 的工具调用信息在流式输出中不直接暴露，
        # 需通过 logger 或中间件捕获。本脚本依赖 middleware 的日志输出。
        # 在 Phase 5 验证阶段，主要通过人工审查输出质量来判断。
        result["passed"] = True  # 默认通过，需人工审查确认

    except Exception as e:
        result["issues"].append(str(e))
        print(f"\n  ✗ 异常: {e}")

    return result


def run_category(agent, category_key: str, interactive: bool = False) -> list:
    """运行一个类别的所有测试。"""
    info = TEST_CASES[category_key]
    print(f"\n{'='*60}")
    print(f"  类别: {info['label']} ({category_key})")
    print(f"  用例数: {len(info['cases'])}")
    print(f"{'='*60}")

    results = []
    for i, query in enumerate(info["cases"], 1):
        print(f"\n[{i}/{len(info['cases'])}]")

        if interactive:
            input("按 Enter 执行测试...")

        result = run_single_test(agent, query, info["expected_tools"])
        results.append(result)

        if interactive and i < len(info["cases"]):
            cont = input("\n继续下一题？[Y/n]: ")
            if cont.lower() in ("n", "no"):
                break

    return results


def export_jsonl(results: list, path: str):
    """导出结果为 JSONL 格式（OpenAI messages 兼容，用于微调数据采集）。"""
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            record = {
                "messages": [
                    {"role": "user", "content": r["query"]},
                    {"role": "assistant", "content": r["response"]},
                ],
                "metadata": {
                    "expected_tools": r["expected_tools"],
                    "tools_called": r["tools_called"],
                    "issues": r["issues"],
                    "timestamp": r["timestamp"],
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n已导出 {len(results)} 条记录到: {path}")


def print_summary(all_results: dict):
    """打印测试汇总。"""
    print(f"\n{'='*60}")
    print(f"  测试汇总")
    print(f"{'='*60}")

    total = 0
    for category, results in all_results.items():
        label = TEST_CASES[category]["label"]
        count = len(results)
        total += count
        print(f"  {label}: {count} 条测试")

    print(f"  ─────────────────")
    print(f"  合计: {total} 条测试")
    print(f"\n  ⚠️ 注意：本脚本仅执行测试并记录输出。")
    print(f"  需人工审查以下维度：")
    print(f"    1. 工具调用是否正确触发（期望工具：rag_summarize / fill_context_for_report）")
    print(f"    2. 是否有凭空回答（未调用工具直接编造）")
    print(f"    3. 是否有重复调用、死循环")
    print(f"    4. fill_context_for_report 调用时机是否合理")
    print(f"    5. 报告格式是否遵循模板")
    print(f"    6. 术语使用是否准确")
    print(f"  ─────────────────")
    print(f"  后端: {rag_conf.get('backend', 'unknown')}")
    print(f"  模型: {rag_conf.get('chat_model_name', 'unknown')}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="EquiMind 工具调用验证")
    parser.add_argument(
        "--category", "-c",
        choices=list(TEST_CASES.keys()),
        help="仅测试指定类别",
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="PATH",
        help="导出结果为 JSONL（用于微调数据采集）",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="交互模式：逐题确认",
    )
    args = parser.parse_args()

    # 延迟导入 agent（避免启动时加载模型）
    print(f"[test_tool_calling] 后端: {rag_conf.get('backend', 'dashscope')}")
    print(f"[test_tool_calling] 模型: {rag_conf.get('chat_model_name', 'unknown')}")
    print(f"[test_tool_calling] 正在初始化 Agent...")

    from agent.react_agent import ReactAgent
    agent = ReactAgent()

    # 确定要测试的类别
    if args.category:
        categories = {args.category: TEST_CASES[args.category]}
    else:
        categories = TEST_CASES

    all_results = {}
    for key in categories:
        all_results[key] = run_category(agent, key, interactive=args.interactive)

    print_summary(all_results)

    # 导出
    if args.export:
        flat_results = []
        for results in all_results.values():
            flat_results.extend(results)
        export_jsonl(flat_results, args.export)


if __name__ == "__main__":
    main()

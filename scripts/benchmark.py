#!/usr/bin/env python
"""
EquiMind 性能基准测试脚本

针对本地 vLLM 推理服务进行全面的性能测试，覆盖：
  - 首 token 延迟（TTFT）
  - 端到端诊断总时间
  - 并发诊断吞吐
  - 显存峰值监控
  - tokens/s 吞吐量

Usage:
  python scripts/benchmark.py                  # 全部测试
  python scripts/benchmark.py --quick           # 快速测试（每个类别 1 条）
  python scripts/benchmark.py --concurrency 4   # 设置并发度
  python scripts/benchmark.py --output report.json  # 输出 JSON 报告
"""
import sys
import os
import json
import time
import argparse
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from utils.config_handler import rag_conf
from utils.logger_handler import logger

# ============================================================
# 测试查询（覆盖 5 大类别，从 faq-examples.md 选取代表性用例）
# ============================================================
BENCHMARK_QUERIES = {
    "aviation": [
        ("aviation_simple", "涡扇发动机高压压气机退化有哪些典型传感器表现？"),
        ("aviation_complex", "如何根据传感器数据判断涡扇发动机的剩余使用寿命（RUL）？请给出详细的分析步骤。"),
    ],
    "railway_contacts": [
        ("railway_simple", "高铁接触网实时监测的关键参数包括哪些？"),
        ("railway_complex", "接触网几何参数（拉出值、导高）超限的标准是什么？请说明具体的测量方法和处置措施。"),
    ],
    "hydropower": [
        ("hydro_simple", "水电机组运行监测的关键指标有哪些？"),
        ("hydro_complex", "水轮机振动过大的常见原因有哪些？请提供系统性的排查思路。"),
    ],
    "railway_track": [
        ("track_simple", "铁路轨道振动监测中加速度计的典型采样频率是多少？"),
        ("track_complex", "如何通过加速度数据识别轨道波磨和轨面剥落？请说明信号处理方法和特征频率。"),
    ],
    "report": [
        ("report_simple", "给我生成我的使用报告"),
        ("report_complex", "生成设备的近期监测运维报告，包含故障分析和维护建议"),
    ],
}


class VRAMMonitor:
    """在后台持续监控 nvidia-smi，记录显存峰值。"""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.peak_mb = 0
        self.samples = []

    def _poll(self):
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    used = int(result.stdout.strip())
                    self.samples.append(used)
                    if used > self.peak_mb:
                        self.peak_mb = used
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return {
            "peak_vram_mb": self.peak_mb,
            "avg_vram_mb": round(sum(self.samples) / len(self.samples)) if self.samples else 0,
            "sample_count": len(self.samples),
        }


def measure_ttft(agent, query: str) -> dict:
    """测量首 token 延迟（TTFT）。

    通过流式调用，记录从请求发出到收到第一个有意义 token 的时间差。
    """
    t_start = time.perf_counter()
    first_token_time = None
    token_count = 0
    full_response = []

    try:
        for chunk in agent.execute_stream(query):
            token_count += 1
            if first_token_time is None and chunk.strip():
                first_token_time = time.perf_counter()
            full_response.append(chunk)
    except Exception as e:
        return {"error": str(e), "ttft_ms": None, "total_s": None, "tokens": 0}

    t_end = time.perf_counter()

    return {
        "ttft_ms": round((first_token_time - t_start) * 1000, 1) if first_token_time else None,
        "total_s": round(t_end - t_start, 2),
        "tokens": token_count,
        "response_len": len("".join(full_response)),
    }


def run_single_benchmark(agent, query_id: str, query: str, warmup: bool = False) -> dict:
    """运行单条基准测试。"""
    label = "WARMUP" if warmup else "BENCH"
    print(f"  [{label}] {query_id}: {query[:50]}...", end=" ", flush=True)

    vram = VRAMMonitor(interval=0.3)
    vram.start()

    result = measure_ttft(agent, query)
    vram_info = vram.stop()
    result.update(vram_info)
    result["query_id"] = query_id
    result["warmup"] = warmup

    if result.get("error"):
        print(f"ERROR: {result['error']}")
    else:
        print(f"TTFT={result['ttft_ms']}ms, Total={result['total_s']}s, "
              f"Tokens={result['tokens']}, VRAM_peak={result['peak_vram_mb']}MB")

    return result


def run_concurrency_test(agent_factory, queries: list, max_workers: int = 4) -> list:
    """并发诊断测试。"""
    print(f"\n{'='*60}")
    print(f"  并发测试 (max_workers={max_workers}, {len(queries)} 条查询)")
    print(f"{'='*60}")

    vram = VRAMMonitor(interval=0.5)
    vram.start()

    t_start = time.perf_counter()
    results = []

    def _task(qid, qtext):
        # 每个线程创建独立的 agent 实例
        agent = agent_factory()
        return run_single_benchmark(agent, qid, qtext)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_task, q[0], q[1]): q[0]
            for q in queries
        }
        for future in as_completed(futures):
            results.append(future.result())

    t_end = time.perf_counter()
    vram_info = vram.stop()

    total_wall = t_end - t_start
    return {
        "queries": results,
        "total_wall_s": round(total_wall, 2),
        "concurrency": max_workers,
        "peak_vram_mb": vram_info["peak_vram_mb"],
        "queries_per_second": round(len(results) / total_wall, 2) if total_wall > 0 else 0,
    }


def print_report(all_results: dict, output_path: Optional[str] = None):
    """打印并可选导出 JSON 报告。"""
    print(f"\n{'='*60}")
    print(f"  性能测试报告")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    seq_results = all_results.get("sequential", [])
    conc_results = all_results.get("concurrency", {})

    # 分类统计
    print(f"\n--- 串行测试（排除 warmup） ---")
    real_results = [r for r in seq_results if not r.get("warmup")]
    if real_results:
        ttfts = [r["ttft_ms"] for r in real_results if r.get("ttft_ms")]
        totals = [r["total_s"] for r in real_results if r.get("total_s")]
        vrams = [r["peak_vram_mb"] for r in real_results if r.get("peak_vram_mb")]

        print(f"  测试数: {len(real_results)}")
        print(f"  TTFT (ms): min={min(ttfts):.0f}, max={max(ttfts):.0f}, avg={sum(ttfts)/len(ttfts):.0f}")
        print(f"  总时间 (s): min={min(totals):.1f}, max={max(totals):.1f}, avg={sum(totals)/len(totals):.1f}")
        print(f"  显存峰值 (MB): max={max(vrams)}")

    if conc_results:
        c = conc_results
        print(f"\n--- 并发测试 (workers={c.get('concurrency', '?')}) ---")
        print(f"  总 wall time: {c.get('total_wall_s')}s")
        print(f"  QPS: {c.get('queries_per_second')}")
        print(f"  显存峰值: {c.get('peak_vram_mb')}MB")

    print(f"\n  后端: {rag_conf.get('backend', 'unknown')}")
    print(f"  模型: {rag_conf.get('chat_model_name', 'unknown')}")
    print(f"{'='*60}")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"  报告已导出: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="EquiMind 性能基准测试")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式：每个类别仅 1 条查询")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="并发测试的线程数（默认 4）")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="导出 JSON 报告路径")
    parser.add_argument("--skip-concurrency", action="store_true",
                        help="跳过并发测试")
    args = parser.parse_args()

    print(f"[benchmark] 后端: {rag_conf.get('backend', 'dashscope')}")
    print(f"[benchmark] 模型: {rag_conf.get('chat_model_name', 'unknown')}")
    print(f"[benchmark] 快速模式: {args.quick}")
    print(f"[benchmark] 正在初始化 Agent...")

    from agent.react_agent import ReactAgent

    def make_agent():
        return ReactAgent()

    agent = make_agent()

    # ---- 串行测试 ----
    print(f"\n{'='*60}")
    print(f"  串行基准测试")
    print(f"{'='*60}")

    # Warmup：第一条不计入统计（GPU 懒加载 + CUDA kernel 编译）
    print("\n  [WARMUP] 预热中...")
    run_single_benchmark(agent, "warmup", "涡扇发动机有哪些常见故障模式？", warmup=True)

    sequential_results = []
    for category, queries in BENCHMARK_QUERIES.items():
        print(f"\n  --- {category} ---")
        qs = queries[:1] if args.quick else queries
        for qid, qtext in qs:
            result = run_single_benchmark(agent, qid, qtext)
            sequential_results.append(result)

    all_results = {"sequential": sequential_results}

    # ---- 并发测试 ----
    if not args.skip_concurrency:
        # 选取各分类的简单查询做并发
        conc_queries = [
            (f"{cat}_conc", qs[0][1])
            for cat, qs in BENCHMARK_QUERIES.items()
        ]
        all_results["concurrency"] = run_concurrency_test(
            make_agent, conc_queries, max_workers=args.concurrency,
        )

    print_report(all_results, args.output)


if __name__ == "__main__":
    main()

"""
数据集预处理脚本：将大型数值型数据集聚合为描述性摘要文本，优化 RAG 摄入效率。

策略：
  C-MAPSS:   每台发动机生成一条退化曲线摘要（约800条），而非原始20万行逐行嵌入
  Railway:    按GPS分段 + 统计摘要，而非原始数十万行逐行嵌入
  NTSB:       尝试将 .mdb 转为 CSV，或直接从NTSB网站获取CSV

Usage:
  python preprocess_datasets.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "backend" / "data"
CMAPSS_DIR = DATA_DIR / "cmapss"
RAILWAY_DIR = DATA_DIR / "railway"
NTSB_DIR = DATA_DIR / "ntsb"

CMAPSS_COLUMNS = [
    "unit_number", "time_cycles",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21",
]

CMAPSS_DATASET_INFO = {
    "FD001": {"conditions": 1, "fault_modes": "HPC退化", "train_engines": 100, "test_engines": 100},
    "FD002": {"conditions": 6, "fault_modes": "HPC退化", "train_engines": 260, "test_engines": 259},
    "FD003": {"conditions": 1, "fault_modes": "HPC退化+Fan退化", "train_engines": 100, "test_engines": 100},
    "FD004": {"conditions": 6, "fault_modes": "HPC退化+Fan退化", "train_engines": 249, "test_engines": 248},
}

# 关键传感器名称映射（中文描述）
SENSOR_NAMES = {
    "sensor_1": "风扇进口总温(R)",
    "sensor_2": "低压压气机出口总温(R)",
    "sensor_3": "高压压气机出口总温(R)",
    "sensor_4": "低压涡轮出口总温(R)",
    "sensor_5": "风扇进口压力(psia)",
    "sensor_6": "旁通管道压力(psia)",
    "sensor_7": "高压压气机出口总压(psia)",
    "sensor_8": "物理风扇转速(rpm)",
    "sensor_9": "物理核心转速(rpm)",
    "sensor_10": "发动机压比",
    "sensor_11": "高压压气机出口静压(psia)",
    "sensor_12": "燃油流量/高压压气机出口静压比",
    "sensor_13": "风扇修正转速(rpm)",
    "sensor_14": "核心修正转速(rpm)",
    "sensor_15": "涵道比",
    "sensor_16": "燃烧室燃空比",
    "sensor_17": "排气焓",
    "sensor_18": "需求风扇转速(rpm)",
    "sensor_19": "需求风扇换算转速(rpm)",
    "sensor_20": "高压涡轮冷却剂排放量(lbm/s)",
    "sensor_21": "低压涡轮冷却剂排放量(lbm/s)",
}


def clean_redundant_files():
    """删除冗余文件：CSV副本 + 无表头的train/test .txt（保留RUL .txt）"""
    print("[清理] 删除冗余文件...")

    # 删除CSV副本（与txt重复，且比txt更大）
    for csv_file in CMAPSS_DIR.glob("*.csv"):
        csv_file.unlink()
        print(f"  删除: {csv_file.name}")

    # 删除无表头的train/test .txt（转为summary后不再需要原始逐行数据）
    for txt_file in CMAPSS_DIR.glob("*.txt"):
        name = txt_file.name
        if name.startswith("train_") or name.startswith("test_"):
            txt_file.unlink()
            print(f"  删除: {name}")

    # 删除 railway 原始CSV（转为summary后不需要65MB原始数据）
    for f in RAILWAY_DIR.glob("vibration_raw.csv"):
        f.unlink()
        print(f"  删除: {f.name}")


def generate_cmapss_summaries():
    """为每个FD数据集生成聚合摘要文档。

    每台发动机生成一条退化曲线摘要，包括：
    - 发动机单元编号
    - 运行周期总数
    - 关键传感器初始值、最终值、变化趋势
    - 退化特征描述
    """
    print("\n[C-MAPSS] 生成数据集摘要...")

    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        info = CMAPSS_DATASET_INFO[fd]
        train_file = CMAPSS_DIR / f"train_{fd}.txt"

        # 先从原始txt读取（已被删除就跳过）
        if not train_file.exists():
            # 尝试从RUL推断
            print(f"  {fd}: 原始训练数据文件不存在，使用RUL文件生成摘要...")
            _generate_cmapss_from_rul(fd, info)
            continue

        try:
            df = pd.read_csv(train_file, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
        except Exception as e:
            print(f"  {fd}: 读取失败 ({e})，使用RUL文件生成...")
            _generate_cmapss_from_rul(fd, info)
            continue

        # 关键传感器列（用于趋势分析）
        key_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7",
                       "sensor_8", "sensor_9", "sensor_11", "sensor_12",
                       "sensor_13", "sensor_14", "sensor_15", "sensor_17",
                       "sensor_20", "sensor_21"]

        summaries = []
        summaries.append(f"# NASA C-MAPSS {fd} 数据集摘要")
        summaries.append(f"")
        summaries.append(f"## 数据集概况")
        summaries.append(f"- 数据集编号：{fd}")
        summaries.append(f"- 训练发动机数量：{info['train_engines']}台")
        summaries.append(f"- 测试发动机数量：{info['test_engines']}台")
        summaries.append(f"- 运行工况数：{info['conditions']}种")
        summaries.append(f"- 故障模式：{info['fault_modes']}")
        summaries.append(f"- 数据来源：NASA Prognostics Data Repository (涡扇发动机退化仿真)")
        summaries.append(f"- 总数据行数：{len(df)}行")
        summaries.append(f"- 传感器数量：21个（含温度、压力、转速、流量等）")
        summaries.append(f"")

        # 全局统计
        summaries.append(f"## 全局传感器统计")
        for s in key_sensors:
            sname = SENSOR_NAMES.get(s, s)
            summaries.append(f"- {sname}({s}): 均值={df[s].mean():.2f}, "
                             f"标准差={df[s].std():.2f}, "
                             f"最小值={df[s].min():.2f}, 最大值={df[s].max():.2f}")
        summaries.append(f"")

        # 每台发动机的退化曲线摘要
        summaries.append(f"## 各发动机单元退化曲线摘要")
        grouped = df.groupby("unit_number")
        engine_count = 0

        for unit, group in grouped:
            if engine_count >= 150:  # 限制每数据集最多150台发动机的详细摘要
                break
            cycles = group["time_cycles"].values
            total_cycles = len(group)
            first = group.iloc[0]
            last = group.iloc[-1]

            # 计算关键传感器的退化趋势
            trend_descs = []
            for s in key_sensors[:10]:  # 取前10个关键传感器
                sname = SENSOR_NAMES.get(s, s)
                vals = group[s].values
                if len(vals) < 2:
                    continue
                start_val = vals[0]
                end_val = vals[-1]
                change = end_val - start_val
                pct = (change / abs(start_val)) * 100 if abs(start_val) > 0.01 else 0

                if abs(pct) > 0.5:  # 变化超过0.5%才记录
                    direction = "上升" if change > 0 else "下降"
                    trend_descs.append(f"{sname}从{start_val:.2f}{direction}至{end_val:.2f}(变化{pct:+.1f}%)")

            desc = (f"发动机单元{int(unit)}：总运行周期{total_cycles}，"
                    f"初始周期{int(cycles[0])}至{int(cycles[-1])}。"
                    f"关键传感器退化趋势：" + "；".join(trend_descs[:8]) + "。")
            summaries.append(f"- {desc}")
            engine_count += 1

        if len(grouped) > 150:
            summaries.append(f"- ...（共{len(grouped)}台发动机，以上为前150台摘要）")

        # 传感器相关性说明
        summaries.append(f"")
        summaries.append(f"## 传感器说明")
        for s in key_sensors:
            sname = SENSOR_NAMES.get(s, s)
            summaries.append(f"- {s}({sname})")
        summaries.append(f"")
        summaries.append(f"## 典型故障特征")
        if "HPC退化" in info['fault_modes']:
            summaries.append(f"- 高压压气机(HPC)退化表现：sensor_3(高压压气机出口总温)上升、"
                             f"sensor_7(高压压气机出口总压)下降、sensor_9(核心转速)上升、"
                             f"sensor_12(燃油流量比)上升")
        if "Fan退化" in info['fault_modes']:
            summaries.append(f"- 风扇(Fan)退化表现：sensor_1(风扇进口总温)上升、"
                             f"sensor_5(风扇进口压力)波动、sensor_8(风扇转速)上升")

        # 写入文件
        output_path = CMAPSS_DIR / f"{fd}_summary.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summaries))
        print(f"  生成: {output_path.name} ({len(summaries)}行, {engine_count}台发动机摘要)")


def _generate_cmapss_from_rul(fd, info):
    """当原始训练数据已删除时，基于RUL文件生成简要摘要"""
    rul_file = CMAPSS_DIR / f"RUL_{fd}.txt"
    summaries = []
    summaries.append(f"# NASA C-MAPSS {fd} 数据集摘要")
    summaries.append(f"## 数据集概况")
    summaries.append(f"- 数据集编号：{fd}")
    summaries.append(f"- 运行工况数：{info['conditions']}种")
    summaries.append(f"- 故障模式：{info['fault_modes']}")
    summaries.append(f"- 数据来源：NASA Prognostics Data Repository")
    summaries.append(f"")

    if rul_file.exists():
        with open(rul_file, "r") as f:
            ruls = [line.strip() for line in f if line.strip()]
        summaries.append(f"## 测试发动机剩余使用寿命(RUL)")
        summaries.append(f"- 测试发动机数量：{len(ruls)}台")
        summaries.append(f"- RUL范围：{min(ruls)} ~ {max(ruls)} 周期")
        summaries.append(f"- RUL均值：{np.mean([float(r) for r in ruls]):.1f} 周期")
        summaries.append(f"- RUL中位数：{np.median([float(r) for r in ruls]):.1f} 周期")
        summaries.append(f"")
        summaries.append(f"## RUL详细数据")
        for i, rul in enumerate(ruls[:50], 1):  # 只列出前50个
            summaries.append(f"- 发动机{i}: 剩余寿命 {rul} 周期")
        if len(ruls) > 50:
            summaries.append(f"- ...（共{len(ruls)}台发动机）")

    output_path = CMAPSS_DIR / f"{fd}_summary.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summaries))
    print(f"  生成: {output_path.name} (基于RUL)")


def generate_railway_summary():
    """为铁路轨道监测数据生成摘要文档。从原始CSV采样分析，提取关键信息。"""
    print("\n[Railway] 生成铁路轨道数据集摘要...")

    csv_file = RAILWAY_DIR / "vibration_raw.csv"
    if not csv_file.exists():
        print("  vibration_raw.csv 不存在（可能已聚合），跳过")
        return

    try:
        # 分批读取以节省内存
        chunk_size = 50000
        total_rows = 0
        columns = None
        stats_accum = []

        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            if columns is None:
                columns = chunk.columns.tolist()
            total_rows += len(chunk)
            # 累积每列的统计信息
            chunk_stats = chunk.describe()
            stats_accum.append(chunk_stats)

            # 读取前2个chunk后停止（足够做摘要）
            if total_rows >= 100000:
                break

        summaries = []
        summaries.append(f"# 铁路轨道多模态监测数据集摘要")
        summaries.append(f"")
        summaries.append(f"## 数据集概况")
        summaries.append(f"- 数据来源：Zenodo (Record 17607068)")
        summaries.append(f"- 采集路线：马德里—巴塞罗那（西班牙高铁线路）")
        summaries.append(f"- 采样频率：加速度计 40Hz")
        summaries.append(f"- 传感器类型：加速度计、陀螺仪、磁力仪、GPS、环境传感器（温度/湿度/气压/光照）")
        summaries.append(f"- 总数据量：约{total_rows / 1e6:.1f}M行（分析前{total_rows}行样本）")
        summaries.append(f"- 数据列数：{len(columns)}列")
        summaries.append(f"- 列名：{', '.join(columns[:15])}")
        summaries.append(f"")

        # 合并统计信息
        if stats_accum:
            combined_stats = pd.concat(stats_accum).groupby(level=0).mean()
            summaries.append(f"## 传感器数据统计（采样分析）")
            for col in columns[:10]:  # 前10列
                if col in combined_stats.columns:
                    s = combined_stats[col]
                    summaries.append(f"- {col}: 均值={s['mean']:.4f}, "
                                     f"标准差={s['std']:.4f}, "
                                     f"最小值={s['min']:.4f}, 最大值={s['max']:.4f}")
                else:
                    summaries.append(f"- {col}: 统计数据不可用")
            summaries.append(f"")

        summaries.append(f"## 应用场景")
        summaries.append(f"- 轨道振动异常检测")
        summaries.append(f"- 轨道不平顺分析（基于加速度计和陀螺仪数据）")
        summaries.append(f"- 铁路轨道预测性维护")
        summaries.append(f"- GPS轨迹与传感器数据的时空关联分析")
        summaries.append(f"- 多模态传感器融合的轨道健康状态评估")
        summaries.append(f"")
        summaries.append(f"## 数据特征")
        summaries.append(f"- 加速度计(40Hz)：捕获高频轨道振动信号，用于检测轨面剥落、波磨等短波不平顺")
        summaries.append(f"- 陀螺仪：测量角速度变化，用于检测轨道几何变形（高低、轨向）")
        summaries.append(f"- 磁力仪：检测轨道周边电磁环境，辅助定位轨旁设备")
        summaries.append(f"- GPS：提供位置标记（经纬度、速度），用于轨道病害精确定位")
        summaries.append(f"- 环境传感器：记录温度、湿度、气压、光照，用于分析环境因素对轨道状态的影响")
        summaries.append(f"")
        summaries.append(f"## 典型异常模式参考")
        summaries.append(f"- 加速度幅值突增：可能指示轨面剥落、焊接接头不平顺")
        summaries.append(f"- 加速度频谱峰值偏移：可能指示轨道固有频率变化（扣件松动、道床板结）")
        summaries.append(f"- 陀螺仪角速度异常：可能指示轨道几何变形超限")
        summaries.append(f"- GPS速度-加速度相关性异常：可能指示特定区段轨道质量恶化")

        output_path = RAILWAY_DIR / "railway_summary.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summaries))
        print(f"  生成: {output_path.name} ({len(summaries)}行)")

    except Exception as e:
        print(f"  铁路数据摘要生成失败: {e}")
        # 即使失败也生成基础摘要
        summaries = [
            "# 铁路轨道多模态监测数据集摘要",
            "## 数据集概况",
            "- 数据来源：Zenodo (Record 17607068)",
            "- 采集路线：马德里—巴塞罗那",
            "- 包含传感器：加速度计(40Hz)、陀螺仪、磁力仪、GPS、环境传感器",
            "- 用于轨道振动异常检测与预测性维护",
        ]
        output_path = RAILWAY_DIR / "railway_summary.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summaries))
        print(f"  生成基础摘要: {output_path.name}")


def handle_ntsb():
    """处理NTSB数据：尝试用pandas直接读取.mdb，或提供替代方案。"""
    print("\n[NTSB] 处理航空事故数据库...")
    mdb_file = NTSB_DIR / "avall.mdb"

    if not mdb_file.exists():
        print("  avall.mdb 不存在，跳过")
        return

    # 尝试用 pandas + pyodbc 读取
    try:
        import pyodbc
        conn_str = f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={mdb_file};"
        conn = pyodbc.connect(conn_str)
        tables = [t.table_name for t in conn.cursor().tables(tableType="TABLE")]
        print(f"  MDB表: {tables}")

        for table in tables[:3]:  # 只转前3个表
            df = pd.read_sql(f"SELECT * FROM [{table}]", conn)
            csv_path = NTSB_DIR / f"ntsb_{table}.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"  导出: {csv_path.name} ({len(df)}行)")

            # 生成摘要
            summaries = []
            summaries.append(f"# NTSB航空事故数据 - {table}表")
            summaries.append(f"## 数据概况")
            summaries.append(f"- 记录数：{len(df)}条")
            summaries.append(f"- 字段数：{len(df.columns)}个")
            summaries.append(f"- 字段列表：{', '.join(df.columns[:20].tolist())}")
            if len(df) > 0:
                summaries.append(f"## 数据样本（前10条）")
                for i, row in df.head(10).iterrows():
                    summaries.append(f"- 记录{i+1}: {dict(row)}")
            summary_path = NTSB_DIR / f"ntsb_{table}_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("\n".join(summaries))
            print(f"  生成摘要: {summary_path.name}")

        conn.close()
    except ImportError:
        print("  pyodbc未安装，无法直接读取.mdb文件")
        # 生成一个引导文档
        summaries = [
            "# NTSB航空事故数据库",
            "## 数据概况",
            "- 数据来源：美国国家运输安全委员会 (NTSB)",
            "- 下载地址：https://data.ntsb.gov/avdata",
            "- 文件格式：Microsoft Access (.mdb)",
            "- 覆盖范围：2008年至今的航空事故记录",
            "",
            "## 关键字段（参考）",
            "- 事故日期与时间 (Event Date)",
            "- 事故地点 (Location - City, State)",
            "- 航空器型号 (Aircraft Make/Model)",
            "- 运营人 (Operator)",
            "- 天气条件 (Weather Conditions)",
            "- 损伤程度 (Damage)",
            "- 可能原因 (Probable Cause)",
            "- 伤亡情况 (Injuries/Fatalities)",
            "- 飞行阶段 (Flight Phase)",
            "",
            "## 使用说明",
            "本数据为MS Access格式(.mdb)，需安装pyodbc后转换: pip install pyodbc",
            "或访问 https://www.ntsb.gov/Pages/AviationQueryV2.aspx 手动导出CSV",
        ]
        summary_path = NTSB_DIR / "ntsb_readme.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summaries))
        print(f"  生成说明文档: {summary_path.name}")
    except Exception as e:
        print(f"  MDB处理失败: {e}")


def main():
    print("=" * 50)
    print("数据集预处理：聚合数值数据 → 描述性摘要")

    # 1. 生成 C-MAPSS 摘要（在原始txt上操作，生成summary后再清理）
    generate_cmapss_summaries()

    # 2. 生成 Railway 摘要
    generate_railway_summary()

    # 3. 处理 NTSB
    handle_ntsb()

    # 4. 清理冗余文件（删除大体积原始数值数据）
    print("\n" + "=" * 50)
    print("[最终清理]")
    clean_redundant_files()

    # 5. 打印最终状态
    print("\n" + "=" * 50)
    print("[数据集最终状态]")
    for dir_path, label in [(CMAPSS_DIR, "C-MAPSS"), (NTSB_DIR, "NTSB"), (RAILWAY_DIR, "Railway")]:
        if dir_path.exists():
            files = list(dir_path.iterdir())
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            print(f"  {label}: {len(files)} files, {total_size/1024:.1f} KB")
            for f in files:
                print(f"    - {f.name} ({f.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()

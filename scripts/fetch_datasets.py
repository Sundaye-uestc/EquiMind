"""
外部数据集获取脚本
支持三种数据集的一键下载：
  1. NASA C-MAPSS 涡扇发动机退化仿真数据 (FD001-FD004)
  2. NTSB 航空事故数据库
  3. Zenodo 铁路轨道监测数据集

Usage:
  python fetch_datasets.py --all            # 下载全部
  python fetch_datasets.py --cmapss         # 仅 C-MAPSS
  python fetch_datasets.py --ntsb           # 仅 NTSB
  python fetch_datasets.py --railway        # 仅 铁路轨道
  python fetch_datasets.py --ntsb --months 6  # NTSB 近6个月
"""

import argparse
import os
import sys
import time
import zipfile
import io
from pathlib import Path

# 将 backend 目录加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import requests
from utils.logger_handler import logger

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "backend" / "data"
CMAPSS_DIR = DATA_DIR / "cmapss"
NTSB_DIR = DATA_DIR / "ntsb"
RAILWAY_DIR = DATA_DIR / "railway"

# C-MAPSS 数据文件列名（26列空格分隔）
CMAPSS_COLUMNS = [
    "unit_number", "time_cycles",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21",
]

CMAPSS_FILES = [
    "train_FD001.txt", "train_FD002.txt", "train_FD003.txt", "train_FD004.txt",
    "test_FD001.txt",  "test_FD002.txt",  "test_FD003.txt",  "test_FD004.txt",
    "RUL_FD001.txt",   "RUL_FD002.txt",   "RUL_FD003.txt",   "RUL_FD004.txt",
]
CMAPSS_BASE_URL = "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master"

# ============================================================
# 工具函数
# ============================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """下载单个文件到目标路径，支持断点提示"""
    try:
        logger.info(f"  下载: {url}")
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
        size_mb = downloaded / (1024 * 1024)
        logger.info(f"  完成: {dest.name} ({size_mb:.1f} MB)")
        return True
    except requests.RequestException as e:
        logger.error(f"  下载失败 {url}: {e}")
        return False


# ============================================================
# 1. NASA C-MAPSS 数据集
# ============================================================
def fetch_cmapss(add_headers: bool = True):
    """
    从 GitHub 镜像下载 C-MAPSS 数据集。
    原始为空格分隔的 .txt 文件（无表头）。
    add_headers=True 时会生成带表头的 CSV 副本，更适合 RAG 摄入。
    """
    logger.info("=" * 50)
    logger.info("[C-MAPSS] 开始下载 NASA 涡扇发动机退化仿真数据...")
    ensure_dir(CMAPSS_DIR)

    success_count = 0
    for filename in CMAPSS_FILES:
        url = f"{CMAPSS_BASE_URL}/{filename}"
        dest = CMAPSS_DIR / filename
        if dest.exists():
            logger.info(f"  {filename} 已存在，跳过")
            success_count += 1
            continue

        if download_file(url, dest):
            success_count += 1

        time.sleep(0.3)  # 避免 GitHub 限速

    logger.info(f"[C-MAPSS] 下载完成: {success_count}/{len(CMAPSS_FILES)} 个文件")

    # 为训练/测试文件生成带表头的 CSV 版本（更适合 RAG 语义检索）
    if add_headers:
        logger.info("[C-MAPSS] 生成带表头的 CSV 文件...")
        for name in CMAPSS_FILES:
            if not (name.startswith("train_") or name.startswith("test_")):
                continue
            txt_path = CMAPSS_DIR / name
            csv_path = CMAPSS_DIR / name.replace(".txt", ".csv")
            if csv_path.exists():
                continue
            try:
                import pandas as pd
                df = pd.read_csv(txt_path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
                df.to_csv(csv_path, index=False, encoding="utf-8")
                logger.info(f"  生成: {csv_path.name}")
            except Exception as e:
                logger.warning(f"  生成CSV失败 {name}: {e}")

    return success_count


# ============================================================
# 2. NTSB 航空事故数据库
# ============================================================
def fetch_ntsb(months: int = 12):
    """
    从 NTSB 官方下载航空事故数据。

    方法: 直接下载 NTSB 的 avall.zip (2008至今全部航空事故),
    解压后转为 CSV 存入 ntsb/ 目录。

    如果安装了 ntsb-api (pip install ntsb-api), 也可以用 API 按月拉取。
    """
    logger.info("=" * 50)
    logger.info("[NTSB] 开始获取航空事故数据库...")
    ensure_dir(NTSB_DIR)

    # 方案A: 尝试使用 ntsb-api Python 包 (按月拉取)
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list"],
            capture_output=True, text=True
        )
        if "ntsb-api" in result.stdout or "ntsb_api" in result.stdout:
            logger.info("[NTSB] 检测到 ntsb-api，使用 API 按月拉取...")
            _fetch_ntsb_via_api(months)
            return
    except Exception:
        pass

    # 方案B: 直接下载 avall.zip 并转换
    logger.info("[NTSB] 使用直接下载方式 (avall.zip)...")
    _fetch_ntsb_direct()


def _fetch_ntsb_via_api(months: int):
    """通过 ntsb-api 包按月拉取数据"""
    try:
        from ntsb_api import NTSBClient
        client = NTSBClient()

        import datetime
        today = datetime.date.today()
        all_records = []

        for m in range(months):
            month = today.month - m
            year = today.year
            while month < 1:
                month += 12
                year -= 1
            logger.info(f"  拉取 {year}-{month:02d}...")
            try:
                data = client.download_month(year, month, mode="Aviation")
                all_records.append(data)
            except Exception as e:
                logger.warning(f"  {year}-{month:02d} 拉取失败: {e}")
            time.sleep(0.5)

        # 合并写入 CSV
        import pandas as pd
        if all_records:
            df = pd.concat(all_records, ignore_index=True)
            csv_path = NTSB_DIR / "ntsb_aviation_accidents.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info(f"[NTSB] 保存 {len(df)} 条记录到 {csv_path}")
    except ImportError:
        logger.warning("[NTSB] ntsb-api 不可用，回退到直接下载方式")
        _fetch_ntsb_direct()


def _fetch_ntsb_direct():
    """直接下载 NTSB avall.zip 并解压"""
    # NTSB 航空数据下载地址 (2008至今，MDB格式)
    ntsb_url = "https://data.ntsb.gov/avdata/FileDirectory/DownloadFile?fileID=C%3A%5Cavdata%5Cavall.zip"
    zip_path = NTSB_DIR / "avall.zip"

    if not zip_path.exists():
        if not download_file(ntsb_url, zip_path):
            logger.error("[NTSB] avall.zip 下载失败")
            logger.info("[NTSB] 替代方案: 手动访问 https://www.ntsb.gov/Pages/AviationQueryV2.aspx 导出CSV")
            return

    # 解压
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(NTSB_DIR)
        logger.info(f"[NTSB] 解压完成: {NTSB_DIR}")

        # 如果解压出 .mdb 文件，尝试转为 CSV
        for f in NTSB_DIR.glob("*.mdb"):
            _convert_mdb_to_csv(f)
        for f in NTSB_DIR.glob("*.accdb"):
            _convert_mdb_to_csv(f)
    except Exception as e:
        logger.error(f"[NTSB] 解压/转换失败: {e}")


def _convert_mdb_to_csv(mdb_path: Path):
    """将 MS Access .mdb 文件转为 CSV"""
    try:
        import pandas as pd
        import pyodbc
        conn_str = f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={mdb_path};"
        conn = pyodbc.connect(conn_str)
        tables = [t.table_name for t in conn.cursor().tables(tableType="TABLE")]
        for table in tables:
            df = pd.read_sql(f"SELECT * FROM [{table}]", conn)
            csv_path = mdb_path.with_suffix(f".{table}.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info(f"  导出: {csv_path.name} ({len(df)} 行)")
        conn.close()
    except ImportError:
        logger.warning("[NTSB] pyodbc 未安装，无法转换 .mdb。请手动安装: pip install pyodbc")
    except Exception as e:
        logger.warning(f"[NTSB] MDB转换失败: {e}")


# ============================================================
# 3. Zenodo 铁路轨道监测数据集
# ============================================================
def fetch_railway():
    """
    从 Zenodo 下载铁路轨道监测数据集。
    包含加速度计(40Hz)、陀螺仪、磁力仪、GPS等多模态传感器数据。
    """
    logger.info("=" * 50)
    logger.info("[Railway] 开始下载铁路轨道监测数据集...")
    ensure_dir(RAILWAY_DIR)

    # Zenodo 记录: Train track environmental data for predictive maintenance
    zenodo_record_id = "17607068"
    zenodo_api = f"https://zenodo.org/api/records/{zenodo_record_id}"

    try:
        resp = requests.get(zenodo_api, timeout=30)
        resp.raise_for_status()
        record = resp.json()

        files = record.get("files", [])
        logger.info(f"[Railway] 找到 {len(files)} 个文件")

        for f in files:
            filename = f.get("key", "")
            file_url = f.get("links", {}).get("self", "")
            file_size = f.get("size", 0)

            if not file_url:
                continue

            dest = RAILWAY_DIR / filename
            if dest.exists():
                logger.info(f"  {filename} 已存在，跳过")
                continue

            logger.info(f"  下载 {filename} ({file_size / 1e6:.1f} MB)...")
            download_file(file_url, dest)

        logger.info("[Railway] 下载完成")

    except requests.RequestException as e:
        logger.error(f"[Railway] Zenodo API 请求失败: {e}")
        # 回退: 直接使用已知的文件URL
        fallback_url = f"https://zenodo.org/records/{zenodo_record_id}/files/vibration_raw.csv?download=1"
        dest = RAILWAY_DIR / "vibration_raw.csv"
        if not dest.exists():
            logger.info("[Railway] 使用回退URL直接下载...")
            download_file(fallback_url, dest)


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="外部数据集获取工具")
    parser.add_argument("--all", action="store_true", help="下载全部三个数据集")
    parser.add_argument("--cmapss", action="store_true", help="下载 NASA C-MAPSS 数据集")
    parser.add_argument("--ntsb", action="store_true", help="下载 NTSB 航空事故数据库")
    parser.add_argument("--railway", action="store_true", help="下载铁路轨道监测数据集")
    parser.add_argument("--months", type=int, default=12, help="NTSB 拉取月数 (默认12)")
    parser.add_argument("--no-headers", action="store_true", help="C-MAPSS 不生成带表头的CSV")
    args = parser.parse_args()

    # 默认下载全部
    run_all = args.all or not (args.cmapss or args.ntsb or args.railway)

    if run_all or args.cmapss:
        try:
            fetch_cmapss(add_headers=not args.no_headers)
        except Exception as e:
            logger.error(f"[C-MAPSS] 下载过程出错: {e}")

    if run_all or args.ntsb:
        try:
            fetch_ntsb(months=args.months)
        except Exception as e:
            logger.error(f"[NTSB] 下载过程出错: {e}")

    if run_all or args.railway:
        try:
            fetch_railway()
        except Exception as e:
            logger.error(f"[Railway] 下载过程出错: {e}")

    logger.info("=" * 50)
    logger.info("全部下载任务结束。")
    logger.info(f"数据目录: {DATA_DIR}")
    _print_summary()


def _print_summary():
    """打印已下载文件摘要"""
    print("\n[数据集下载状态]")
    for dir_path, label in [(CMAPSS_DIR, "C-MAPSS"), (NTSB_DIR, "NTSB"), (RAILWAY_DIR, "Railway")]:
        if dir_path.exists():
            files = list(dir_path.iterdir())
            if files:
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                print(f"  [OK] {label}: {len(files)} files, {total_size/1e6:.1f} MB")
            else:
                print(f"  [--] {label}: empty")
        else:
            print(f"  [XX] {label}: not downloaded")


if __name__ == "__main__":
    main()

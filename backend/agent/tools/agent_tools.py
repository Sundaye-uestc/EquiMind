import sys
import os
import random
from pathlib import Path

# 将项目根目录加入 sys.path，确保后续所有项目内模块可正常导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path

rag = RagSummarizeService()
external_data = {}


def _get_source_names(query: str) -> str:
    """从检索结果中提取知识库来源文件名，拼接为来源标注。"""
    try:
        docs = rag.retriever_docs(query)
        sources = []
        seen = set()
        for doc in docs:
            source = doc.metadata.get("source", "")
            name = os.path.basename(source)
            if name and name not in seen:
                seen.add(name)
                sources.append(name)
        if sources:
            return "\n\n📚 参考来源：" + "、".join(sources)
    except Exception:
        pass
    return ""


@tool(description="从向量存储库中检索参考资料")
def rag_summarize(query: str) -> str:
    summary = rag.rag_summarize(query)
    return summary + _get_source_names(query)

@tool(description="获取城市天气，以消息字符串形式返回")
def get_weather(city: str) -> str:
    return f"城市{city}天气为晴天，气温26℃，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"

@tool(description="获取用户所在城市名称，以str返回")
def get_user_city() -> str:
    return random.choice(["深圳", "成都", "上海"])

@tool(description="获取用户ID，以str返回")
def get_user_id() -> str:
    return random.choice(["1001","1002","1003","1004","1005","1006","1007","1008","1009","1010"])

@tool(description="获取当前月份，以str形式返回")
def get_month() -> str:
    return random.choice(["2025-01","2025-02","2025-03","2025-04","2025-05","2025-06","2025-07","2025-08","2025-09","2025-10","2025-11","2025-12"])

@tool(description="获取用户性别，以str形式返回")
def get_gender() -> str:
    return random.choice(["男", "女"])

def generate_external_data():
    """
    {
        "user_id": {
            "month" : {"特征"：XXX，"效率"：XXX，...}
            "month" : {"特征"：XXX，"效率"：XXX，...}
            "month" : {"特征"：XXX，"效率"：XXX，...}
            ...
        }
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr: list[str] = line.strip().split(",")

                user_id: str = arr[0].replace('"',"")
                feature: str = arr[1].replace('"',"")
                efficiency: str = arr[2].replace('"',"")
                consumable: str = arr[3].replace('"',"")
                comparison: str = arr[4].replace('"',"")
                time: str = arr[5].replace('"',"")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumable,
                    "对比": comparison,
                }


@tool(description="从外部系统中获取用户使用记录，以str返回，如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    generate_external_data()

    try:
        return external_data[user_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{month}的使用记录数据")
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"


if __name__ == '__main__':
    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
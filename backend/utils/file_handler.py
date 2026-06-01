import hashlib
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, JSONLoader
from langchain_core.documents import Document

from .logger_handler import logger


def get_file_md5_hex(filepath: str):             # 获取文件的md5的十六进制字符串
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return None

    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return None

    md5_obj = hashlib.md5()
    chunk_size = 4096           # 4KB切片，避免文件过大
    try:
        with open(filepath, "rb") as f:     # 必须二进制读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
            """ :=的等价写法
            chunk = f.read(chunk_size)
            while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    """递归扫描目录，返回所有匹配后缀的文件路径列表。"""
    files = []

    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return ()

    for root, _dirs, filenames in os.walk(path):
        for f in filenames:
            if f.endswith(allowed_types):
                files.append(os.path.join(root, f))

    return tuple(files)


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding='utf-8').load()


def csv_loader(filepath: str) -> list[Document]:
    """
    加载CSV文件，将每行数据转为文本文档。
    CSV第一行作为表头，后续每行生成一个Document，
    page_content格式为 "col1: val1, col2: val2, ..."
    """
    try:
        loader = CSVLoader(
            file_path=filepath,
            encoding="utf-8",
            csv_args={"delimiter": ",", "quotechar": '"'},
        )
        docs = loader.load()
        logger.info(f"[CSV加载]{filepath}：共{len(docs)}行记录")
        return docs
    except Exception as e:
        logger.error(f"[CSV加载]{filepath}失败：{str(e)}")
        return []


def json_loader(filepath: str, jq_schema: str = ".[]") -> list[Document]:
    """
    加载JSON文件，使用jq表达式提取文档。
    默认 jq_schema=".[]" 提取顶层数组的每个元素。
    每个元素的所有字段展平为 "key: value" 格式的文本。
    """
    try:
        loader = JSONLoader(
            file_path=filepath,
            jq_schema=jq_schema,
            text_content=False,
        )
        docs = loader.load()
        logger.info(f"[JSON加载]{filepath}：共{len(docs)}个文档")
        return docs
    except Exception as e:
        logger.error(f"[JSON加载]{filepath}失败：{str(e)}")
        return []
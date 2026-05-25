"""
为整个工程提供统一的绝对路径，并在导入时自动将项目根目录加入 sys.path。
"""
import os
import sys
from pathlib import Path

# path_tool.py 位于项目根目录下的 utils/ 子目录中
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def get_project_root() -> str:
    """获取工程所在的根目录"""
    return _project_root


def get_abs_path(relative_path: str) -> str:
    """
    传递相对路径（相对于项目根目录），返回绝对路径。
    如果传入的已经是绝对路径则直接返回。
    """
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(_project_root, relative_path)


if __name__ == '__main__':
    print(get_abs_path(__file__))
    print(get_project_root())
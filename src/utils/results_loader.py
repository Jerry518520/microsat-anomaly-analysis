"""
结果文件加载工具
统一 data/results/ 目录下的 JSON 文件查找逻辑
"""

import json
import os
from pathlib import Path


def get_results_dir() -> str:
    """获取结果目录路径"""
    project_root = Path(__file__).parent.parent.parent
    return str(project_root / "data" / "results")


def load_json(filename: str, results_dir: str = None) -> dict | None:
    """
    从 results 目录加载 JSON 文件

    查找顺序：
    1. results_dir/filename
    2. results_dir/*/filename（子目录）

    Args:
        filename: JSON 文件名
        results_dir: 结果目录路径，默认为 data/results/

    Returns:
        解析后的 dict，未找到返回 None
    """
    if results_dir is None:
        results_dir = get_results_dir()

    if not os.path.isdir(results_dir):
        return None

    # 直接查找
    path = os.path.join(results_dir, filename)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 子目录查找
    for sub in os.listdir(results_dir):
        sub_path = os.path.join(results_dir, sub, filename)
        if os.path.isfile(sub_path):
            with open(sub_path, "r", encoding="utf-8") as f:
                return json.load(f)

    return None

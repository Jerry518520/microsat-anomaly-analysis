"""
重跑段级IForest Baseline — 用项目模块确保一致性
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.iforest_baseline import run_baseline_pipeline
from src.utils.data_loader import load_config

config = load_config()
results = run_baseline_pipeline(config)

print("\n=== 结果 ===")
print(f"c=0.2时 F1 = {results['0.2']['f1']:.6f}")

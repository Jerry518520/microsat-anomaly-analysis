"""上线前稳定性验收：连续跑 N 次检测，验证结果确定性。

**为什么这是正确性测试而非性能测试**：上线前必须确认「同代码同数据多次运行
结果完全一致」。若同一份数据两次跑出不同 F1，说明存在未被控制的随机性
（字典序、并发、缓存顺序），那论文数字就不可信。

内存增长由外部 PowerShell 测量 —— 本脚本刻意不引入 psutil / ctypes 技巧，
避免为一个验收脚本增加依赖或易错代码。

用法：
    ./.venv/Scripts/python.exe scripts/verify_stability.py [N]   # 默认 5
"""
import gc
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.integration.anomaly_rag_pipeline import AnomalyRAGPipeline  # noqa: E402


def metrics(p):
    """从 last_seg_results 算混淆矩阵与 F1（段级）。"""
    s = p.last_seg_results
    y = s["y_true"].to_numpy().astype(int)
    yp = s["is_anomaly"].to_numpy().astype(int)
    tp = int(((yp == 1) & (y == 1)).sum())
    fp = int(((yp == 1) & (y == 0)).sum())
    fn = int(((yp == 0) & (y == 1)).sum())
    tn = int(((yp == 0) & (y == 0)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn)
    return f1, tp, fp, fn, tn


def main(n=5):
    print(f"=== 稳定性验收：连续运行 {n} 次 ===")
    rows = []
    for i in range(1, n + 1):
        p = AnomalyRAGPipeline()
        p.detect_and_explain(max_explanations=0)
        f1, tp, fp, fn, tn = metrics(p)
        rows.append((f1, tp, fp, fn, tn))
        print(f"  第 {i} 次: F1={f1:.9f}  TP={tp} FP={fp} FN={fn} TN={tn}")
        del p
        gc.collect()

    print()
    first = rows[0]
    diffs = [i + 1 for i, r in enumerate(rows) if r != first]
    if diffs:
        print(f"x 不一致：第 {diffs} 次与首次结果不同")
        for i, r in enumerate(rows, 1):
            flag = "" if r == first else "  <-- 与首次不同"
            print(f"    第{i}次 F1={r[0]:.17g} TP={r[1]} FP={r[2]} FN={r[3]} TN={r[4]}{flag}")
        return 1
    print(f"+ {n} 次结果完全一致（混淆矩阵与 F1 逐位相同）")
    print(f"  F1 = {first[0]:.17g}")
    print(f"  混淆矩阵 TP={first[1]} FP={first[2]} FN={first[3]} TN={first[4]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 5))

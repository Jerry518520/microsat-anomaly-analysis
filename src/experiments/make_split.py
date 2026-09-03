"""
生成 v3 唯一权威数据划分 —— split_indices.json

划分口径（纲领 1.1，全项目唯一权威）：
    官方 train (1594 段) --[stratify, seed=42, test_size=0.2]--> fit (80%) + val (20%)
    官方 test  (529  段) --> 最终评估集，全程只碰一次，禁止用于任何参数选择

产物：data/results/v3/split_indices.json（存 segment id 列表，不存 DataFrame）

用法：
    python src/experiments/make_split.py
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_REL = os.path.join("data", "raw", "dataset-提取的合成特征被计算到每个手动分割和标记的遥测段上.csv")
DATA_PATH = os.path.join(PROJECT_ROOT, DATA_REL)
OUT_REL = os.path.join("data", "results", "v3", "split_indices.json")
OUT_PATH = os.path.join(PROJECT_ROOT, OUT_REL)

RANDOM_STATE = 42
VAL_SIZE = 0.2
META_COLS = ["channel", "segment", "anomaly", "train", "sampling"]
TARGET_ANOMALY_RATE = 0.2044  # 2123 段整体异常率 434/2123
RATE_TOLERANCE = 0.01  # 偏差 > 1 个百分点要告警


def _git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def _git_dirty():
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:
        pass
    return None


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"[make_split] 读取 {DATA_REL}: shape={df.shape}")

    if not df["segment"].is_unique:
        raise RuntimeError("segment 列不唯一，无法作为切分 id，请检查数据文件")

    official_train = df[df["train"] == 1].copy()
    official_test = df[df["train"] == 0].copy()
    print(f"[make_split] 官方 train={len(official_train)}（异常 {int(official_train['anomaly'].sum())}），"
          f"官方 test={len(official_test)}（异常 {int(official_test['anomaly'].sum())}）")

    # 只在官方 train 内部切 fit/val；官方 test 完全不参与
    train_ids = official_train["segment"].to_numpy()
    y_train = official_train["anomaly"].to_numpy()

    fit_ids, val_ids = train_test_split(
        train_ids,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    test_ids = official_test["segment"].to_numpy()

    # 排序保证可复现的文件内容（与切分结果无关，仅影响文件字节序）
    fit_ids = sorted(int(i) for i in fit_ids)
    val_ids = sorted(int(i) for i in val_ids)
    test_ids = sorted(int(i) for i in test_ids)

    id_map = {"fit": set(fit_ids), "val": set(val_ids), "test": set(test_ids)}
    for a, b in [("fit", "val"), ("fit", "test"), ("val", "test")]:
        overlap = id_map[a] & id_map[b]
        if overlap:
            raise RuntimeError(f"{a} 与 {b} 存在重叠 id：{sorted(overlap)[:10]}")
    if len(id_map["fit"]) + len(id_map["val"]) + len(id_map["test"]) != len(df):
        raise RuntimeError("三段 id 总数与全量数据不一致，切分有误")

    payload = {
        "fit_ids": fit_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "counts": {"fit": len(fit_ids), "val": len(val_ids), "test": len(test_ids)},
        "anomaly_counts": {
            "fit": int(df[df["segment"].isin(fit_ids)]["anomaly"].sum()),
            "val": int(df[df["segment"].isin(val_ids)]["anomaly"].sum()),
            "test": int(df[df["segment"].isin(test_ids)]["anomaly"].sum()),
        },
        "spec": {
            "source": "official train column == 1 / 0",
            "method": "sklearn.model_selection.train_test_split",
            "test_size": VAL_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": "anomaly (官方 train 内部)",
            "id_column": "segment",
            "note": "test_ids 为最终评估集，禁止用于任何参数/阈值/模型/特征选择",
        },
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "python": sys.version.split()[0],
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "random_state": RANDOM_STATE,
            "data_file": DATA_REL.replace(os.sep, "/"),
            "data_sha256": _sha256(DATA_PATH),
            "script": sys.argv[0],
            "split_spec": "official_train -> 80% fit / 20% val, official_test = final eval only",
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if os.path.exists(OUT_PATH):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = OUT_PATH.replace(".json", f"_{stamp}.json")
        os.rename(OUT_PATH, backup)
        print(f"[make_split][警告] {OUT_REL} 已存在，未覆盖；旧文件已重命名为 {os.path.basename(backup)}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n[make_split] === 各子集样本数与异常率 ===")
    warned = False
    for name in ["fit", "val", "test"]:
        ids = payload[f"{name}_ids"]
        sub = df[df["segment"].isin(ids)]
        n = len(sub)
        k = int(sub["anomaly"].sum())
        rate = k / n if n else float("nan")
        flag = ""
        if abs(rate - TARGET_ANOMALY_RATE) > RATE_TOLERANCE:
            flag = f"  <== 告警：异常率偏离 {TARGET_ANOMALY_RATE:.4f} 超过 {RATE_TOLERANCE:.0%}"
            warned = True
        print(f"  {name:<5} n={n:<5} 异常={k:<4} 异常率={rate:.4f}{flag}")

    channels = {}
    for name in ["fit", "val", "test"]:
        sub = df[df["segment"].isin(payload[f"{name}_ids"])]
        channels[name] = {ch: int(c) for ch, c in sub["channel"].value_counts().sort_index().items()}
    print("\n[make_split] === 各子集通道分布 ===")
    for ch in sorted(df["channel"].unique()):
        print(f"  {ch:<10} fit={channels['fit'].get(ch, 0):<5} "
              f"val={channels['val'].get(ch, 0):<5} test={channels['test'].get(ch, 0)}")

    size = os.path.getsize(OUT_PATH)
    print(f"\n[make_split] 已写入 {OUT_REL}  ({size:,} bytes, sha256={_sha256(OUT_PATH)[:16]}...)")
    print(f"[make_split] 校验命令: python -c \"import json;d=json.load(open(r'{OUT_PATH}',encoding='utf-8'));"
          f"print(len(d['fit_ids']),len(d['val_ids']),len(d['test_ids']))\"")
    if warned:
        print("[make_split][警告] 存在异常率偏差超标的子集，请人工确认后再继续。")


if __name__ == "__main__":
    main()

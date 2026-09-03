"""
v3 统一实验框架 —— 全项目共用的评估与溯源地基

================================================================================
!!                              警          告                                !!
================================================================================
** test_df 只允许在最终评估时使用一次，禁止用于任何参数选择。**

具体禁止的行为（违反即数字作废，不得进论文）：
  * 禁止用 test_df 选超参（contamination、n_estimators、k、阈值、窗口……）
  * 禁止用 test_df 选模型、选特征、选融合方式、选后处理规则
  * 禁止用 test_df 做 early stopping、做阈值扫描、做"看一眼再调"
  * 禁止因为 test_df 上的分数不好看而回过头改任何东西

正确做法：
  1. 在 fit_df 上 fit 模型 / 拟合规则阈值
  2. 在 val_df 上选超参、选阈值、选模型、选特征
  3. 超参全部冻结后，才在 test_df 上跑**一次**最终评估
  4. 用 save_result() 落盘，val 分数与 test 分数都要写进去

本模块提供的四个函数：
    load_split()                              -> (fit_df, val_df, test_df, feature_cols)
    evaluate(y_true, y_pred, n_boot, seed)    -> 全套指标 dict（含 bootstrap 95% CI）
    save_result(result, name, extra)          -> 写入 data/results/v3/{name}.json（自动注入 meta）
    bootstrap_ci(y_true, y_pred, n_boot, seed)-> F1 的 95% 百分位置信区间

标准调用姿势：
    from src.experiments.framework import load_split, evaluate, save_result
    fit_df, val_df, test_df, feature_cols = load_split()
    metrics = evaluate(y_true, y_pred)
    save_result(metrics, "if_global_c020", extra={"val_f1": 0.53, "best_c": 0.2})
================================================================================
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
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------- 路径与常量

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_REL = "data/raw/dataset-提取的合成特征被计算到每个手动分割和标记的遥测段上.csv"
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "dataset-提取的合成特征被计算到每个手动分割和标记的遥测段上.csv")

SPLIT_REL = os.path.join("data", "results", "v3", "split_indices.json")
SPLIT_PATH = os.path.join(PROJECT_ROOT, SPLIT_REL)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results", "v3")

META_COLS = ["channel", "segment", "anomaly", "train", "sampling"]
FEATURE_COLS = [
    "duration", "len", "mean", "var", "std", "kurtosis", "skew",
    "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks",
    "diff_peaks", "diff2_peaks", "diff_var", "diff2_var",
    "gaps_squared", "len_weighted", "var_div_duration", "var_div_len",
]

RANDOM_STATE = 42
SPLIT_SPEC = "official_train -> 80% fit / 20% val, official_test = final eval only"

_SMALL_SAMPLE_N = 30  # 纲领 1.3：n_test < 30 的通道必须标注"不具统计意义"


# ---------------------------------------------------------------- 溯源 meta

def _run_git(args):
    try:
        out = subprocess.run(
            ["git"] + args, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _git_commit():
    return _run_git(["rev-parse", "HEAD"]) or "unknown"


def _git_dirty():
    out = _run_git(["status", "--porcelain"])
    if out is None:
        return None
    return bool(out)


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_meta():
    """构造铁律 3 要求的溯源块"""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "python": sys.version.split()[0],
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "random_state": RANDOM_STATE,
        "data_file": DATA_REL,
        "data_sha256": _sha256(DATA_PATH) if os.path.exists(DATA_PATH) else None,
        "script": sys.argv[0],
        "split_spec": SPLIT_SPEC,
        "split": SPLIT_SPEC,  # 与纲领 JSON 示例的字段名保持一致
    }


# ---------------------------------------------------------------- load_split

def load_split():
    """加载 v3 权威划分。

    返回 (fit_df, val_df, test_df, feature_cols)
      - fit_df      : 官方 train 的 80%（约 1275 段）—— 模型/规则在这上面 fit
      - val_df      : 官方 train 的 20%（约 319 段）—— 超参、阈值在这上面选
      - test_df     : 官方 test（529 段）—— **只允许在最终评估时使用一次，
                      禁止用于任何参数选择**（选超参/阈值/模型/特征一律只用 val_df）
      - feature_cols: 18 个特征列（已排除 5 个元数据列 channel/segment/anomaly/train/sampling）

    三个 df 均保留原始列（含 channel / anomaly 等元数据），index 已 reset。
    若 split_indices.json 不存在，抛出 FileNotFoundError 并提示先跑 make_split.py。
    """
    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(
            f"找不到划分文件 {SPLIT_REL}\n"
            f"请先运行：  python src/experiments/make_split.py\n"
            f"（划分必须全项目唯一，禁止各 agent 自己切）"
        )

    with open(SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"数据文件缺少预期特征列：{missing}")

    out = []
    for key in ["fit_ids", "val_ids", "test_ids"]:
        ids = split[key]
        sub = df[df["segment"].isin(ids)].copy()
        if len(sub) != len(ids):
            raise RuntimeError(
                f"{key} 中有 {len(ids) - len(sub)} 个 segment id 在数据文件里找不到，划分文件已失效"
            )
        out.append(sub.reset_index(drop=True))

    fit_df, val_df, test_df = out
    return fit_df, val_df, test_df, list(FEATURE_COLS)


# ---------------------------------------------------------------- bootstrap

def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42):
    """F1 的 95% 百分位置信区间。

    返回 [lower, upper]。用 np.random.default_rng(seed) 有放回重采样 n_boot 次。
    """
    yt = np.asarray(y_true).astype(int).ravel()
    yp = np.asarray(y_pred).astype(int).ravel()
    n = len(yt)
    if n == 0:
        return [None, None]

    rng = np.random.default_rng(seed)
    scores = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[i] = f1_score(yt[idx], yp[idx], zero_division=0)

    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


# ---------------------------------------------------------------- evaluate

def evaluate(y_true, y_pred, n_boot=1000, seed=42):
    """统一评估口径（纲领 1.3）。

    返回 dict:
        f1, precision, recall, mcc,
        tn, fp, fn, tp,
        n, n_anomaly_true,
        f1_ci95: [lower, upper],
        false_alarm_ratio: fp/(tp+fp)，tp+fp==0 时为 None

    y_true / y_pred 接受 list / np.array / pd.Series。一律 zero_division=0。
    当 n < 30 时额外给出 'small_sample_warning' 字段（纲领 1.3 要求标注）。
    """
    yt = np.asarray(y_true).astype(int).ravel()
    yp = np.asarray(y_pred).astype(int).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"y_true 与 y_pred 长度不一致：{yt.shape} vs {yp.shape}")

    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

    far = float(fp / (tp + fp)) if (tp + fp) > 0 else None

    res = {
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "mcc": float(matthews_corrcoef(yt, yp)) if len(np.unique(yp)) > 1 or len(np.unique(yt)) > 1 else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(yt)),
        "n_anomaly_true": int(yt.sum()),
        "n_anomaly_pred": int(yp.sum()),
        "f1_ci95": bootstrap_ci(yt, yp, n_boot=n_boot, seed=seed),
        "false_alarm_ratio": far,
        "bootstrap": {"n_boot": int(n_boot), "seed": int(seed), "metric": "f1", "ci": 0.95},
    }

    if res["n"] < _SMALL_SAMPLE_N:
        res["small_sample_warning"] = f"n={res['n']} 过小（< {_SMALL_SAMPLE_N}），不具统计意义，不得单独作为结论"

    return res


# ---------------------------------------------------------------- save_result

def _unique_backup_path(name: str) -> str:
    """生成一个不会与已有文件冲突的备份路径（带秒级时间戳，冲突则追加序号）。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(RESULTS_DIR, f"{name}_{stamp}.json")
    if not os.path.exists(base):
        return base
    i = 1
    while os.path.exists(os.path.join(RESULTS_DIR, f"{name}_{stamp}_{i}.json")):
        i += 1
    return os.path.join(RESULTS_DIR, f"{name}_{stamp}_{i}.json")


def save_result(result: dict, name: str, extra: dict = None):
    """把结果写入 data/results/v3/{name}.json，自动注入 meta 溯源块。

    结构：
        {
          "name": name,
          "meta": { timestamp, git_commit, git_dirty, python, sklearn, numpy, pandas,
                    random_state, data_file, data_sha256, script, split_spec, split },
          "results": { <result 的全部内容> + <extra 合并进来> }
        }

    extra 用于放 val 分数、选出的超参等（铁律：val 与 test 分数都要报）。
    若 {name}.json 已存在，**不覆盖**，改名为 {name}_{YYYYmmdd_HHMMSS}.json 并打印警告。

    返回实际写入的文件路径。
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    target = os.path.join(RESULTS_DIR, f"{name}.json")

    if os.path.exists(target):
        backup = _unique_backup_path(name)
        os.rename(target, backup)
        print(f"[save_result][警告] {name}.json 已存在，未覆盖；"
              f"旧文件已重命名为 {os.path.basename(backup)}")

    results = dict(result)
    if extra:
        results.update(extra)

    payload = {
        "name": name,
        "meta": build_meta(),
        "results": results,
    }

    with open(target, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[save_result] 已写入 data/results/v3/{os.path.basename(target)}")
    return target


# ---------------------------------------------------------------- 打印辅助

def format_metrics(metrics: dict, title: str = "") -> str:
    """把 evaluate() 的结果格式化成可读文本（供各实验脚本统一打印）。"""
    lo, hi = metrics["f1_ci95"]
    lines = []
    if title:
        lines.append(title)
    lines.append(
        f"  F1        = {metrics['f1']:.4f}   95% CI [{lo:.4f}, {hi:.4f}]"
    )
    lines.append(f"  Precision = {metrics['precision']:.4f}")
    lines.append(f"  Recall    = {metrics['recall']:.4f}")
    lines.append(f"  MCC       = {metrics['mcc']:.4f}")
    lines.append(f"  n         = {metrics['n']}   真值异常 = {metrics['n_anomaly_true']}   "
                 f"预测异常 = {metrics['n_anomaly_pred']}")
    far = metrics["false_alarm_ratio"]
    lines.append(f"  误报率 fp/(tp+fp) = {'None' if far is None else f'{far:.4f}'}")
    lines.append(
        f"  混淆矩阵: TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}  TP={metrics['tp']}"
    )
    if metrics.get("small_sample_warning"):
        lines.append(f"  [!] {metrics['small_sample_warning']}")
    return "\n".join(lines)


def evaluate_per_channel(df: pd.DataFrame, y_pred, group_col="channel") -> dict:
    """按通道分组评估，返回 {channel: metrics_dict}。

    方便做逐通道消融；n < 30 的通道会自动带 small_sample_warning 标注（纲领 1.3）。
    """
    y_pred = np.asarray(y_pred).astype(int).ravel()
    out = {}
    tmp = df.reset_index(drop=True).copy()
    tmp["_y_pred"] = y_pred
    for ch, sub in tmp.groupby(group_col, sort=True):
        out[str(ch)] = evaluate(sub["anomaly"].to_numpy(), sub["_y_pred"].to_numpy())
    return out

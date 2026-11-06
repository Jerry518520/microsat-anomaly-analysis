"""Run the predeclared comparisons and train-only nested evaluation."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.detection.frozen import load_bundle, predict_features, TRAINING_CSV, SPLIT_JSON
from src.evaluation.saved_predictions import evaluate_file, metrics
from src.experiments.confirmed import select_recipe, predict_recipe


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_predictions(path, predictions, labelled):
    result = predictions.merge(labelled[["segment", "anomaly"]].rename(columns={"anomaly": "truth"}),
                               on="segment", validate="many_to_one")
    result.to_csv(path, index=False, float_format="%.17g")
    return result


def run(output):
    judges = load_bundle()  # Verify the canonical input and current frozen-source identity first.
    frame = pd.read_csv(ROOT / TRAINING_CSV, float_precision="round_trip").sort_values("segment").reset_index(drop=True)
    split = json.loads((ROOT / SPLIT_JSON).read_text(encoding="utf-8"))
    sets = [set(split[key]) for key in ("fit_ids", "val_ids", "test_ids")]
    if frame.segment.duplicated().any() or sum(map(len, sets)) != len(frame):
        raise ValueError("split membership count mismatch")
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)) or set.union(*sets) != set(frame.segment):
        raise ValueError("split overlap or missing IDs")
    fit, val, test = [frame[frame.segment.isin(ids)].copy() for ids in sets]
    if (len(fit), len(val), len(test)) != (1275, 319, 529):
        raise ValueError("unexpected fixed split size")
    if not fit.train.eq(1).all() or not val.train.eq(1).all() or not test.train.eq(0).all():
        raise ValueError("official train/test flags disagree with split")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    recipe = select_recipe(fit, val)
    write_json(output / "selection.json", recipe)
    save_predictions(output / "validation_predictions.csv", predict_recipe(fit, val.drop(columns="anomaly"), recipe), val)
    predictions = predict_recipe(fit, test.drop(columns="anomaly"), recipe)
    engineering = predict_features(test, judges)
    engineering["method"] = "engineering_gate"
    engineering["score"] = np.nan  # Its display score is not a common ranking score across gates.
    predictions = pd.concat([predictions, engineering], ignore_index=True)
    save_predictions(output / "predictions.csv", predictions, test)
    print("Saved 529-segment comparisons", flush=True)

    official_train = frame[frame.train == 1].reset_index(drop=True)
    nested, memberships, recipes = [], [], []
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (train_indices, held_indices) in enumerate(outer.split(official_train, official_train.anomaly)):
        outer_train, held = official_train.iloc[train_indices], official_train.iloc[held_indices]
        inner_fit, inner_val = train_test_split(outer_train, test_size=.2, random_state=43 + fold,
                                               stratify=outer_train.anomaly)
        inner_fit, inner_val = inner_fit.sort_values("segment"), inner_val.sort_values("segment")
        selected = select_recipe(inner_fit, inner_val)
        predicted = predict_recipe(outer_train.sort_values("segment"), held.drop(columns="anomaly"), selected)
        predicted["fold"] = fold
        nested.append(predicted)
        recipes.append({"fold": fold, **selected})
        memberships.append({"fold": fold, "inner_fit_ids": inner_fit.segment.tolist(),
                            "inner_val_ids": inner_val.segment.tolist(),
                            "outer_train_ids": outer_train.segment.tolist(), "held_ids": held.segment.tolist()})
        print(f"Saved nested fold {fold + 1}/3", flush=True)
    nested = save_predictions(output / "nested_predictions.csv", pd.concat(nested, ignore_index=True), official_train)
    write_json(output / "nested_selections.json", recipes)
    write_json(output / "split_manifest.json", {"fixed": {key: split[key] for key in ("fit_ids", "val_ids", "test_ids")},
                                               "nested": memberships})
    # The scorer reads the CSV back and imports no detector or experiment component.
    results = evaluate_file(output / "predictions.csv")
    nested_results = evaluate_file(output / "nested_predictions.csv")
    nested_results["fold_metrics"] = {str(fold): {method: metrics(rows.truth, rows.final_pred)
        for method, rows in subset.groupby("method")} for fold, subset in nested.groupby("fold")}
    write_json(output / "metrics.json", results)
    write_json(output / "nested_metrics.json", nested_results)
    saved = pd.read_csv(output / "predictions.csv")
    saved[saved.truth != saved.final_pred].to_csv(output / "error_cases.csv", index=False, float_format="%.17g")
    summary = []
    for method, result in results["overall"].items():
        summary.append({"method": method, **{key: result[key] for key in ("precision", "recall", "f1", "accuracy", "mcc", "roc_auc", "average_precision")},
                        **result["confusion"], "f1_ci_low": result["f1_channel_bootstrap_95"][0],
                        "f1_ci_high": result["f1_channel_bootstrap_95"][1]})
    pd.DataFrame(summary).to_csv(output / "comparison_table.csv", index=False, float_format="%.17g")
    lines = ["# 毕业设计检测对照数据", "", "规程为 graduation-confirmation-v1。统一对照与当前工程方案分别登记。", "",
             "| 方法 | Precision | Recall | F1 | Accuracy | MCC | FP | FN |", "|---|---|---|---|---|---|---|---|"]
    for row in summary:
        lines.append(f"| {row['method']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['accuracy']:.4f} | {row['mcc']:.4f} | {row['fp']} | {row['fn']} |")
    lines += ["", "529 段属于已查看基准复核。engineering_gate 使用历史固定通道参数，单独作为工程方案参照。", "",
              "nested_metrics.json 使用官方训练部分的 1594 段完成 3 折嵌套评价。每个外层段只评分一次。所有参数与通道算子均在对应内层验证部分选择。", "",
              "metrics.json 含逐通道指标、混淆矩阵、适用 ROC-AUC 与 AP、通道簇 bootstrap 区间及配对差值。小样本通道和单类通道均保留。通道簇只有 9 个，区间需结合各通道结果阅读。", "",
              "独立复算命令为 .venv/Scripts/python.exe -m src.evaluation.saved_predictions data/results/graduation-confirmation-v1/predictions.csv --output metrics-recomputed.json。", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    sources = [TRAINING_CSV, SPLIT_JSON, "docs/contracts/detection_experiment_protocol.md",
               "src/experiments/confirmed.py", "src/evaluation/saved_predictions.py",
               "scripts/run_confirmed_experiments.py", "src/utils/stat_rules.py", "src/detection/frozen.py"]
    identity = next(iter(judges.values())).bundle_identity.as_fields()
    manifest = {"status": "complete", "protocol": "graduation-confirmation-v1",
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "sklearn": sklearn.__version__, "fit_rows": len(fit), "validation_rows": len(val), "benchmark_rows": len(test),
                "internal_out_of_fold_rows": len(official_train), "benchmark_previously_seen": True,
                "engineering_bundle": identity,
                "inputs_and_sources": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources},
                "artifacts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir())}}
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps({row["method"]: round(row["f1"], 6) for row in summary}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/results/graduation-confirmation-v1")
    run(parser.parse_args().output)

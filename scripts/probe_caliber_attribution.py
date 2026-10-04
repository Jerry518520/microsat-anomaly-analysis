"""2x2 口径归因：把生产 0.6446 与论文 0.6281 的差距完全拆解。

两个变量：
  拟合口径: train 全集 1594（生产现行） vs fit 1275（论文权威划分）
  0884 排除: 有（生产现行 NO_ANOMALY_CHANNELS） vs 无（论文 gate_perchannel 未排除）

已实测两格：
  fit 1275 + 有排除 = 0.6386554621848739
  fit 1275 + 无排除 = 0.628099173553719  ← 与论文 0.6281 逐位相同
本脚本补齐另两格，得出完整归因表。
"""
import os
import sys
import json
import shutil
import tempfile
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.integration.anomaly_rag_pipeline import AnomalyRAGPipeline
from src.experiments.framework import evaluate
import src.integration.anomaly_rag_pipeline as arp

DATA_REL = os.path.join("data", "raw",
                        "dataset-提取的合成特征被计算到每个手动分割和标记的遥测段上.csv")
KEYS = ["tp", "fp", "fn", "tn", "f1", "precision", "recall", "mcc"]
sp = json.load(open(os.path.join(ROOT, "data/results/v3/split_indices.json"),
                    encoding="utf-8"))
FIT, VAL, TEST = sp["fit_ids"], sp["val_ids"], sp["test_ids"]
FULL = set(FIT) | set(VAL)


def run(fit_set, no_anom, tmpdir, tag):
    raw = pd.read_csv(os.path.join(ROOT, DATA_REL), encoding="utf-8-sig")
    raw["train"] = raw["segment"].isin(set(fit_set)).astype(int)
    fn = f"f_{tag}.csv"
    p = os.path.join(tmpdir, fn)
    raw.to_csv(p, index=False, encoding="utf-8")
    orig = arp.NO_ANOMALY_CHANNELS
    arp.NO_ANOMALY_CHANNELS = no_anom
    try:
        pipe = AnomalyRAGPipeline()
        pipe.config["data"]["raw_dir"] = tmpdir
        pipe.config["data"]["features_file"] = fn
        pipe.detect(segments_df=pd.DataFrame())
    finally:
        arp.NO_ANOMALY_CHANNELS = orig
    s = pipe.last_seg_results
    s = s[s["segment"].isin(set(TEST))]
    m = evaluate(s["y_true"].astype(int).values, s["is_anomaly"].astype(int).values)
    return {k: m[k] for k in KEYS}


def main():
    tmpdir = tempfile.mkdtemp(prefix="attrib_")
    try:
        cells = {}
        for cal_name, fs in (("train1594", FULL), ("fit1275", set(FIT))):
            for ex_name, na in (("excl0884", {"CADC0884"}), ("noexcl", set())):
                r = run(fs, na, tmpdir, f"{cal_name}_{ex_name}")
                cells[f"{cal_name}|{ex_name}"] = r

        print("=" * 92)
        print("2x2 归因表（test 529 段，全部走真实 detect()）")
        print("=" * 92)
        print(f"{'口径':34s} {'TP':>3s} {'FP':>3s} {'FN':>3s} {'TN':>3s} "
              f"{'F1':>10s} {'P':>7s} {'R':>7s}")
        order = ["train1594|excl0884", "train1594|noexcl",
                 "fit1275|excl0884", "fit1275|noexcl"]
        label = {"train1594|excl0884": "(1) train1594 + 排除 = 生产现行",
                 "train1594|noexcl": "(2) train1594 + 不排除",
                 "fit1275|excl0884": "(3) fit1275  + 排除",
                 "fit1275|noexcl": "(4) fit1275  + 不排除 = 论文口径"}
        for k in order:
            c = cells[k]
            print(f"{label[k]:34s} {c['tp']:3d} {c['fp']:3d} {c['fn']:3d} "
                  f"{c['tn']:3d} {c['f1']:10.6f} {c['precision']:7.4f} "
                  f"{c['recall']:7.4f}")

        prod = cells["train1594|excl0884"]["f1"]
        paper = cells["fit1275|noexcl"]["f1"]
        print(f"\n生产现行 (1) = {prod!r}")
        print(f"论文口径   (4) = {paper!r}   (fusion.json 记录 0.628099173553719)")
        print(f"总差距 (1)-(4) = {prod - paper:+.6f}")

        d_excl = cells["fit1274|excl0884"]["f1"] if False else \
            cells["fit1275|excl0884"]["f1"] - cells["fit1275|noexcl"]["f1"]
        d_cal = cells["train1594|excl0884"]["f1"] - cells["fit1275|excl0884"]["f1"]
        print(f"\n归因分解:")
        print(f"  0884 排除贡献(同 fit 口径)   : {d_excl:+.6f}")
        print(f"  拟合口径贡献(同排除策略)     : {d_cal:+.6f}")
        print(f"  两项相加                     : {d_excl + d_cal:+.6f}")

        print(f"\n逐位一致性检查:")
        print(f"  (4) == fusion.json gate_perchannel.test.f1 : "
              f"{cells['fit1275|noexcl']['f1'] == 0.628099173553719}")

        with open(os.path.join(ROOT, "data/results/v3/_caliber_attrib.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"cells": cells, "labels": label,
                       "total_gap": prod - paper,
                       "delta_0884_exclusion": d_excl,
                       "delta_fit_caliber": d_cal}, f,
                      ensure_ascii=False, indent=2)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()

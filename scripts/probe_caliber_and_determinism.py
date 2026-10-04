"""生产管线口径差异 + 确定性验证 harness（只读诊断，不改生产代码）。

设计原则：绝不另写一套检测逻辑。三种口径全部走真实
AnomalyRAGPipeline.detect()，唯一变化是「哪些行 features_df["train"]==1」。

实现手法：detect() 内部用 train 列做掩码，故对每种口径生成一份临时 CSV，
把 train 列按该口径重写，再把 config["data"]["raw_dir"] 指向临时目录。
生产代码一行未改，其余逻辑（阈值/分通道IF/门控/段级聚合）完全不变。

test 评估：last_seg_results 会因口径 (b) 多出 val 段，按官方 test_ids 过滤后评估。
过滤安全性已由「口径 (a) 经临时CSV 与原生路径逐位一致」验证（见 asserts）。
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
from src.experiments.framework import evaluate, load_split

DATA_REL = os.path.join("data", "raw",
                        "dataset-提取的合成特征被计算到每个手动分割和标记的遥测段上.csv")
OUT = os.path.join(ROOT, "data", "results", "v3", "_caliber_probe.json")


def _load_raw():
    return pd.read_csv(os.path.join(ROOT, DATA_REL), encoding="utf-8-sig")


def run_once(caliber, tmpdir, fit_ids=None, order=None):
    """跑一次真实 detect()。

    caliber: 'native'  = 原文件，train==1 即 train 全集
             'masked'  = 用 fit_ids 重写 train 列
    order:   None = 保持 CSV 原序（detect 内部会 sort_values("segment")）
             'fit_then_val' = 按 fit_ids 段 + val_ids 段的顺序重排后再交给 detect
    """
    import hashlib
    raw = _load_raw()
    if order == "fit_then_val":
        # 只重排 train 部分（fit 段在前、val 段在后），test 段必须保留，
        # 否则 test_df 为空会让 _segment_baseline_iforest 的 predict 抛
        # "Found array with 0 sample(s)"。
        # ⚠ fv 必须用权威 fit/val 列表本身，不能用调用方传入的 fit_ids
        #   （c' 传入的是 fit∪val 全集，直接拿来拼会让 val 行重复出现）。
        sp = json.load(open(os.path.join(ROOT, "data/results/v3/split_indices.json"),
                           encoding="utf-8"))
        _, val_df, _, _ = load_split()
        fv = list(sp["fit_ids"]) + list(val_df["segment"])
        is_tr = raw["train"] == 1
        tr_part = raw[is_tr].set_index("segment").loc[fv].reset_index()
        te_part = raw[~is_tr]
        raw = pd.concat([tr_part, te_part], ignore_index=True)

    if caliber == "masked":
        raw = raw.copy()
        raw["train"] = raw["segment"].isin(set(fit_ids)).astype(int)

    tmp_csv = os.path.join(tmpdir, "features_probe.csv")
    raw.to_csv(tmp_csv, index=False, encoding="utf-8")
    sha = hashlib.sha256(open(tmp_csv, "rb").read()).hexdigest()[:16]

    pipe = AnomalyRAGPipeline()
    pipe.config["data"]["raw_dir"] = tmpdir          # 绝对路径，os.path.join 直接采用
    pipe.config["data"]["features_file"] = "features_probe.csv"
    pipe.detect(segments_df=pd.DataFrame())
    seg = pipe.last_seg_results
    seg.attrs["_sha"] = sha
    seg.attrs["_n_train"] = int(raw["train"].sum())
    return seg


def score(seg, test_ids):
    """按官方 test 段过滤后评估。"""
    s = seg[seg["segment"].isin(set(test_ids))]
    yt = s["y_true"].astype(int).values
    yp = s["is_anomaly"].astype(int).values
    m = evaluate(yt, yp)
    return {
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
        "mcc": m["mcc"], "n_test": int(len(yt)), "n_pred_pos": int(yp.sum()),
        "f1_ci95": m["f1_ci95"],
    }


def main():
    sp = json.load(open(os.path.join(ROOT, "data/results/v3/split_indices.json"),
                        encoding="utf-8"))
    fit_ids, val_ids, test_ids = sp["fit_ids"], sp["val_ids"], sp["test_ids"]
    fit_df, val_df, _, _ = load_split()

    print("=" * 70)
    print("口径集合核对")
    print("=" * 70)
    raw = _load_raw()
    tr = set(raw.loc[raw["train"] == 1, "segment"])
    print(f"  data train==1        : {len(tr)}")
    print(f"  framework fit_df      : {len(fit_df)}")
    print(f"  framework val_df      : {len(val_df)}")
    print(f"  fit ∪ val             : {len(set(fit_ids) | set(val_ids))}")
    print(f"  fit ∪ val == train   : {(set(fit_ids) | set(val_ids)) == tr}")

    results = {"set_facts": {
        "train_mask_rows": len(tr),
        "fit_rows": len(fit_df), "val_rows": len(val_df),
        "test_rows": len(test_ids),
        "fit_union_val_equals_train": (set(fit_ids) | set(val_ids)) == tr,
    }}

    tmpdir = tempfile.mkdtemp(prefix="caliber_")
    try:
        # ---------- 确定性：原生路径连跑 3 次 ----------
        print("\n" + "=" * 70)
        print("确定性验证：原生生产路径连跑 3 次（caliber a = train 全集）")
        print("=" * 70)
        det_runs = []
        for i in range(3):
            seg = run_once("native", tmpdir)
            sc = score(seg, test_ids)
            det_runs.append(sc)
            print(f"  run{i+1}: TP={sc['tp']} FP={sc['fp']} FN={sc['fn']} "
                  f"TN={sc['tn']} F1={sc['f1']!r} n_test={sc['n_test']}")

        keys = ["tp", "fp", "fn", "tn", "f1", "precision", "recall", "mcc"]
        identical = all(
            all(repr(r[k]) == repr(det_runs[0][k]) for k in keys)
            for r in det_runs
        )
        print(f"\n  逐位 repr 全部一致: {identical}")
        for k in keys:
            vals = [repr(r[k]) for r in det_runs]
            print(f"    {k:10s}: {vals[0]}" + ("  (3次一致)" if len(set(vals)) == 1
                                              else f"  ⚠ 不一致 {vals}"))
        results["determinism"] = {"runs": det_runs, "bitwise_identical": identical}

        # ---------- harness 自校验：caliber (a) 走临时 CSV 应与原生逐位一致 ----------
        print("\n" + "=" * 70)
        print("harness 自校验：caliber(a) 经临时CSV vs 原生路径")
        print("=" * 70)
        seg_a_csv = run_once("masked", tmpdir, fit_ids=tr)   # train列 = 全集 == 原生
        a_csv = score(seg_a_csv, test_ids)
        same_a = all(repr(a_csv[k]) == repr(det_runs[0][k]) for k in keys)
        print(f"  原生 F1 = {det_runs[0]['f1']!r}")
        print(f"  临时CSV F1 = {a_csv['f1']!r}")
        print(f"  逐位一致: {same_a}（为 True 才说明临时CSV 手法可信）")
        assert same_a, "harness 自校验失败：临时CSV 路径与原生路径不一致"
        results["harness_selfcheck"] = {"native": det_runs[0], "via_tmp_csv": a_csv,
                                        "bitwise_identical": same_a}

        # ---------- 三种口径 ----------
        print("\n" + "=" * 70)
        print("三种口径对比（全部走真实 detect()）")
        print("=" * 70)
        calibers = {}

        calibers["a_train_full_1594"] = {
            "desc": "train 全集 1594 段（features_df['train']==1），生产现行口径",
            "n_fit_rows": len(tr), **det_runs[0],
        }
        print(f"  (a) train 全集 {len(tr)}: TP={calibers['a_train_full_1594']['tp']} "
              f"FP={calibers['a_train_full_1594']['fp']} "
              f"FN={calibers['a_train_full_1594']['fn']} "
              f"TN={calibers['a_train_full_1594']['tn']} "
              f"F1={calibers['a_train_full_1594']['f1']!r}")

        seg_b = run_once("masked", tmpdir, fit_ids=fit_ids)
        calibers["b_fit_1275"] = {
            "desc": "framework.load_split() 的 fit 1275 段，与实验侧权威划分同口径",
            "n_fit_rows": len(fit_ids), **score(seg_b, test_ids),
        }
        print(f"  (b) fit {len(fit_ids)}: TP={calibers['b_fit_1275']['tp']} "
              f"FP={calibers['b_fit_1275']['fp']} "
              f"FN={calibers['b_fit_1275']['fn']} "
              f"TN={calibers['b_fit_1275']['tn']} "
              f"F1={calibers['b_fit_1275']['f1']!r}")

        # (c) fit+val —— 集合与 (a) 相同，按 segment 排序后也应逐位相同
        seg_c = run_once("masked", tmpdir, fit_ids=set(fit_ids) | set(val_ids))
        calibers["c_fit_val_1594"] = {
            "desc": "fit+val 1594 段（集合与 (a) 完全相同）",
            "n_fit_rows": len(set(fit_ids) | set(val_ids)), **score(seg_c, test_ids),
        }
        print(f"  (c) fit+val {len(set(fit_ids)|set(val_ids))}: "
              f"TP={calibers['c_fit_val_1594']['tp']} "
              f"FP={calibers['c_fit_val_1594']['fp']} "
              f"FN={calibers['c_fit_val_1594']['fn']} "
              f"TN={calibers['c_fit_val_1594']['tn']} "
              f"F1={calibers['c_fit_val_1594']['f1']!r}")

        same_ac = all(repr(calibers["a_train_full_1594"][k]) ==
                      repr(calibers["c_fit_val_1594"][k]) for k in keys)
        print(f"  (a) 与 (c) 逐位一致: {same_ac}"
              f"  （集合相同 + 均按 segment 排序，故应一致）")

        # (c') 同一 1594 段集合，但按 fit-then-val 顺序喂入（探测行序敏感性）
        seg_cp = run_once("masked", tmpdir, fit_ids=set(fit_ids) | set(val_ids),
                          order="fit_then_val")
        print(f"  [c'] 输入 CSV sha={seg_cp.attrs['_sha']} "
              f"n_train={seg_cp.attrs['_n_train']}")
        calibers["cprime_fitval_order_1594"] = {
            "desc": "同 (c) 的 1594 段，但行序为 fit 段在前 val 段在后"
                    "（探测 detect 内部 sort_values('segment') 是否已消除行序影响）",
            "n_fit_rows": len(set(fit_ids) | set(val_ids)), **score(seg_cp, test_ids),
        }
        print(f"  (c') 同集合改行序: TP={calibers['cprime_fitval_order_1594']['tp']} "
              f"FP={calibers['cprime_fitval_order_1594']['fp']} "
              f"FN={calibers['cprime_fitval_order_1594']['fn']} "
              f"TN={calibers['cprime_fitval_order_1594']['tn']} "
              f"F1={calibers['cprime_fitval_order_1594']['f1']!r}")
        same_cc = all(repr(calibers["c_fit_val_1594"][k]) ==
                      repr(calibers["cprime_fitval_order_1594"][k]) for k in keys)
        print(f"  (c) 与 (c') 逐位一致: {same_cc}"
              f"  （True = 排序确实解耦了行序）")

        results["calibers"] = calibers
        results["a_equals_c"] = same_ac
        results["c_equals_cprime"] = same_cc

        # 与历史记录对比
        print("\n" + "=" * 70)
        print("与文档中既有数字对比")
        print("=" * 70)
        hist = {
            "code_comment_0.6446": 0.6446,
            "paper_gate_perchannel_0.6281": 0.6281,
            "production_fixed_json_0.4648": 0.46475195822454307,
        }
        for k, v in hist.items():
            print(f"  {k:34s} = {v}")
            for cn, cv in calibers.items():
                print(f"      vs {cn:30s} F1={cv['f1']:.6f}  Δ={cv['f1']-v:+.6f}")
        results["historical_comparison"] = hist
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {OUT}")


if __name__ == "__main__":
    main()

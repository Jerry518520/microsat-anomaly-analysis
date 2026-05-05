import json, os

results_dir = r"F:\微小卫星项目\microsat-anomaly-analysis\data\results"

print("=== fair_comparison_results.json ===")
with open(os.path.join(results_dir, "fair_comparison_results.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

summary = data.get("summary", {})
for k, v in summary.items():
    if isinstance(v, dict):
        print("  %s: %s" % (k, json.dumps(v, ensure_ascii=False)))
    else:
        print("  %s: %s" % (k, v))

files = [
    ("contamination_search_results.json", "Contamination"),
    ("rule_fallback_results.json", "Rule Fallback"),
    ("threshold_sweep_results.json", "Threshold Sweep"),
    ("subsampling_sweep_results.json", "Subsampling Sweep"),
]

for fname, label in files:
    fpath = os.path.join(results_dir, fname)
    print("\n=== %s (%s) ===" % (label, fname))
    if not os.path.exists(fpath):
        print("  NOT FOUND")
        continue
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            if "best" in d:
                print("  best: %s" % json.dumps(d["best"], ensure_ascii=False))
            for k, v in d.items():
                if k == "best":
                    continue
                if isinstance(v, dict):
                    f1 = v.get("seg_f1") or v.get("f1") or v.get("SegF1")
                    if f1:
                        print("  %s: SegF1=%.4f" % (k, f1))
                    else:
                        print("  %s: %s" % (k, str(v)[:80]))
                elif isinstance(v, (int, float, str)):
                    print("  %s: %s" % (k, v))
        elif isinstance(d, list):
            for item in d[:3]:
                print("  %s" % str(item)[:120])
    except Exception as e:
        print("  ERROR: %s" % e)

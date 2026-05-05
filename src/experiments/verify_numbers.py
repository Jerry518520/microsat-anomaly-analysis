import json

# 1. Baseline
with open('data/results/iforest_segment_baseline_results.json', 'r') as f:
    d = json.load(f)
print('=== 1. Baseline ===')
for k, v in d.items():
    print(f'  c={k}: F1={v["f1"]:.6f}')
print()

# 2. contamination_search_ws20
with open('data/results/contamination_search_ws20.json', 'r') as f:
    d = json.load(f)
print('=== 2. ws=20 contamination sweep ===')
for k, v in d['global_contamination_sweep'].items():
    seg = v['segment_level']
    print(f'  {k}: seg_F1={seg["f1"]:.6f}')
print()

# 3. rule_fallback
with open('data/results/rule_fallback_results.json', 'r') as f:
    d = json.load(f)
print('=== 3. rule_fallback (方案A-G) ===')
for k, v in d['results_summary'].items():
    seg = v.get('segment', {})
    f1 = seg.get('f1', 'N/A')
    if f1 != 'N/A':
        print(f'  {k}: seg_F1={f1:.6f}')
    else:
        print(f'  {k}: no segment data')
print()

# 4. threshold_sweep
with open('data/results/threshold_sweep_1.8_results.json', 'r') as f:
    d = json.load(f)
print('=== 4. threshold_sweep (方案G) ===')
for k, v in d['results'].items():
    print(f'  threshold={k}: F1={v["f1"]:.6f}')
print()

# 5. subsampling_sweep
with open('data/results/subsampling_sweep_1.9_results.json', 'r') as f:
    d = json.load(f)
print('=== 5. subsampling_sweep ===')
for k, v in d['results'].items():
    m = v['metrics']
    print(f'  psi={k}: F1={m["f1"]:.6f}')

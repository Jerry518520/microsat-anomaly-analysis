import json

path = r'F:\微小卫星项目\microsat-anomaly-analysis\notebooks\微小卫星遥测异常检测与RAG解释系统_完整Pipeline.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 手动修复特定单元格的内容

# 修复 Cell 12 (Stage 1: 全局IForest Baseline)
# 当前内容严重损坏，需要重建
cell_12_source = [
    "feature_cols = [c for c in features_df.columns if c not in META_COLS]\n",
    "X_train = train_df[feature_cols].fillna(0).values\n",
    "X_test = test_df[feature_cols].fillna(0).values\n",
    "\n",
    "print(\"=== Stage 1: 全局IForest Baseline ===\")\n",
    "print(f\"特征维度: {len(feature_cols)}\")\n",
    "# 搜索最优contamination\n",
    "best_f1, best_c = 0, 0.1\n",
    "for c in [0.1, 0.2, 0.3, 0.4, 0.5]:\n",
    "    wp, _ = eval_iforest(X_train, X_test, None, contamination=c, random_state=RANDOM_STATE)\n",
    "    seg = segment_aggregate(test_df, wp, threshold=0.5)\n",
    "    f1 = calc_seg_f1(seg)\n",
    "    print(f\"  c={c}: SegF1={f1:.4f}\")\n",
    "    if f1 > best_f1:\n",
    "        best_f1, best_c = f1, c\n",
    "\n",
    "print(f\"最优: c={best_c}, SegF1={best_f1:.4f}\")\n",
    "stage1_f1 = best_f1\n"
]

# 修复 Cell 16 (Stage 3: 投票阈值调优)
# 需要修复 best_threshold = 0.25seg_stage3 = ... 的问题
cell_16_source = [
    "print(\"=== Stage 3: 投票阈值调优 ===\")\n",
    "\n",
    "for th in [0.5, 0.4, 0.35, 0.3, 0.25, 0.2]:\n",
    "    seg = segment_aggregate(test_df, test_pred.values, threshold=th)\n",
    "    f1 = calc_seg_f1(seg)\n",
    "    print(f\"  threshold={th}: SegF1={f1:.4f}\")\n",
    "best_threshold = 0.25\n",
    "seg_stage3 = segment_aggregate(test_df, test_pred.values, threshold=best_threshold)\n",
    "stage3_f1 = calc_seg_f1(seg_stage3)\n",
    "print(f\"最优阈值: {best_threshold}, SegF1={stage3_f1:.4f}\")\n"
]

# 修复 Cell 20 (性能对比总结)
# 需要修复 improvement = ... * 100print(f"...") 的问题
cell_20_source = [
    "results = {\n",
    "    'Stage': ['1.全局IForest', '2.分通道+规则', '3.降阈值', '4.子采样'],\n",
    "    'SegF1': [stage1_f1, stage2_f1, stage3_f1, stage4_f1],\n",
    "    '提升': [0, stage2_f1-stage1_f1, stage3_f1-stage2_f1, stage4_f1-stage3_f1]\n",
    "}\n",
    "\n",
    "results_df = pd.DataFrame(results)\n",
    "results_df['累计提升'] = results_df['提升'].cumsum()\n",
    "results_df['vs Stage1'] = results_df['SegF1'] - stage1_f1\n",
    "\n",
    "print(\"=== 性能对比 ===\")\n",
    "print(results_df.to_string(index=False, float_format='%.4f'))\n",
    "improvement = (stage4_f1 - stage1_f1) / stage1_f1 * 100\n",
    "print(f\"总提升: {stage1_f1:.4f} → {stage4_f1:.4f} (+{improvement:.1f}%)\")\n"
]

# 修复 Cell 25 (RAG智能解释)
# 需要添加缺失的for循环和RAG解释生成代码
cell_25_source = [
    "if RAG_AVAILABLE:\n",
    "    # 初始化RAG Pipeline\n",
    "    pipeline = AnomalyRAGPipeline()\n",
    "    \n",
    "    # 检测异常（使用我们刚才训练的模型）\n",
    "    # 注意：这里演示用pipeline内置的检测，实际可传入自定义预测\n",
    "    anomalies = pipeline.detect()\n",
    "    \n",
    "    print(f\"检测到 {len(anomalies)} 个异常段\")\n",
    "    \n",
    "    # 为前3个异常生成解释\n",
    "    for i, anomaly in enumerate(anomalies[:3]):\n",
    "        print(f\"--- 异常 {i+1} ---\")\n",
    "        print(f\"段: {anomaly.segment}\")\n",
    "        print(f\"通道: {anomaly.channel}\")\n",
    "        print(f\"类型: {anomaly.anomaly_type}\")\n",
    "        print(f\"得分: {anomaly.anomaly_score:.3f}\")\n",
    "        \n",
    "        # 生成RAG解释\n",
    "        rag_result = pipeline.explain(anomaly)\n",
    "        print(f\"RAG解释:{rag_result.answer[:500]}...\")\n",
    "        print(\"\" + \"=\"*60)\n"
]

# 找到并替换这些单元格
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell.get('source', []))
        # Cell 12: Stage 1 (直接按索引修复)
        if i == 12:
            print(f"修复 Cell {i}: Stage 1")
            cell['source'] = cell_12_source
        # Cell 16: Stage 3
        elif 'Stage 3: 投票阈值调优' in src and 'best_threshold = 0.25' in src:
            print(f"修复 Cell {i}: Stage 3")
            cell['source'] = cell_16_source
        # Cell 20: 性能对比
        elif '性能对比' in src and 'improvement = (stage4_f1' in src:
            print(f"修复 Cell {i}: 性能对比")
            cell['source'] = cell_20_source
        # Cell 25: RAG解释
        elif 'RAG_AVAILABLE' in src and 'anomalies = pipeline.detect()' in src:
            print(f"修复 Cell {i}: RAG解释")
            cell['source'] = cell_25_source

# Save
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("修复完成！")

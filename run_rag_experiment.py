"""
RAG 实验脚本 — 2026-05-01
目标: 跑通异常检测 + RAG解释, 生成开题答辩用的实验数据

输出:
  data/results/anomaly_rag_results.json  — 完整结果
  data/results/rag_retrieval_stats.json  — 检索统计
  data/results/rag_example_report.md     — 3份样例报告(给答辩用)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# 设置 API key (从环境变量或硬编码)
if not os.environ.get("VOLCENGINE_API_KEY"):
    os.environ["VOLCENGINE_API_KEY"] = "[REDACTED]"

from src.rag.vectorstore import FAISSVectorStore
from src.rag.embedding import BGE_M3_Embedder
from src.rag.llm_client import LLMClient
from src.rag.prompts import PromptTemplates

# ============================
# Part 1: 异常检测 (复用已有结果)
# ============================

def load_detection_results():
    """从已有实验结果加载异常检测数据"""
    # 读取 segments 数据
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    import yaml
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    segments_file = raw_dir / config["data"]["segments_file"]
    features_file = raw_dir / config["data"]["features_file"]
    
    segments_df = pd.read_csv(segments_file, encoding="utf-8")
    dataset_df = pd.read_csv(features_file, encoding="utf-8")
    
    # 获取测试集中的异常段
    test_data = dataset_df[dataset_df["train"] == 0]
    anomaly_segments = test_data[test_data["anomaly"] == 1]
    
    print(f"测试集总段数: {len(test_data)}")
    print(f"异常段数: {len(anomaly_segments)}")
    print(f"通道分布:")
    for ch in sorted(anomaly_segments["channel"].unique()):
        cnt = len(anomaly_segments[anomaly_segments["channel"] == ch])
        print(f"  {ch}: {cnt}")
    
    return segments_df, anomaly_segments


# ============================
# Part 2: RAG 检索测试 (不需要 LLM, 快)
# ============================

def test_retrieval_quality(embedder, vectorstore, anomaly_segments, n_samples=30):
    """测试 RAG 检索质量 (不调用 LLM, 快速)"""
    print(f"\n{'='*60}")
    print(f"Part 2: RAG 检索质量测试 (抽样 {n_samples} 个异常段)")
    print(f"{'='*60}")
    
    # 通道物理含义
    CHANNEL_PHYSICS = {
        "CADC0872": "Magnetometer X-axis 磁力计X轴",
        "CADC0873": "Magnetometer Y-axis 磁力计Y轴",
        "CADC0874": "Magnetometer Z-axis 磁力计Z轴",
        "CADC0884": "Photodiode 1 angle 光电二极管1",
        "CADC0886": "Photodiode 2 angle 光电二极管2",
        "CADC0888": "Photodiode 3 angle 光电二极管3",
        "CADC0890": "Photodiode 4 angle 光电二极管4",
        "CADC0892": "Photodiode 5 angle 光电二极管5",
        "CADC0894": "Photodiode 6 angle 光电二极管6",
    }
    
    # 均匀抽样 (覆盖所有通道)
    channels = sorted(anomaly_segments["channel"].unique())
    samples = []
    per_ch = max(1, n_samples // len(channels))
    for ch in channels:
        ch_segs = anomaly_segments[anomaly_segments["channel"] == ch]
        n_pick = min(per_ch, len(ch_segs))
        picked = ch_segs.sample(n=n_pick, random_state=42)
        samples.append(picked)
    samples_df = pd.concat(samples).head(n_samples)
    
    retrieval_results = []
    total_time = 0
    
    for i, (_, row) in enumerate(samples_df.iterrows()):
        ch = row["channel"]
        seg = row["segment"]
        ch_desc = CHANNEL_PHYSICS.get(ch, ch)
        
        # 构造检索查询
        query = f"satellite telemetry anomaly {ch_desc} segment {seg}"
        
        start = time.time()
        results = vectorstore.search(query, top_k=5, score_threshold=0.3)
        elapsed = time.time() - start
        total_time += elapsed
        
        # 统计来源
        sources = set()
        for r in results:
            src = r["metadata"].get("filename", "unknown")
            sources.add(src)
        
        retrieval_results.append({
            "segment": int(seg),
            "channel": ch,
            "query": query,
            "n_results": len(results),
            "top_score": results[0]["score"] if results else 0,
            "avg_score": np.mean([r["score"] for r in results]) if results else 0,
            "sources": list(sources),
            "n_sources": len(sources),
            "retrieval_time_ms": elapsed * 1000,
        })
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples_df)}] 已完成")
    
    # 统计汇总
    stats = {
        "n_queries": len(retrieval_results),
        "avg_results_per_query": np.mean([r["n_results"] for r in retrieval_results]),
        "avg_top_score": np.mean([r["top_score"] for r in retrieval_results]),
        "avg_score": np.mean([r["avg_score"] for r in retrieval_results]),
        "avg_retrieval_time_ms": np.mean([r["retrieval_time_ms"] for r in retrieval_results]),
        "total_retrieval_time_s": total_time,
        "source_diversity": {},
        "per_channel": {},
    }
    
    # 来源多样性
    from collections import Counter
    all_sources = []
    for r in retrieval_results:
        all_sources.extend(r["sources"])
    stats["source_diversity"] = dict(Counter(all_sources).most_common())
    
    # 分通道统计
    for ch in channels:
        ch_results = [r for r in retrieval_results if r["channel"] == ch]
        if ch_results:
            stats["per_channel"][ch] = {
                "n": len(ch_results),
                "avg_top_score": np.mean([r["top_score"] for r in ch_results]),
                "avg_results": np.mean([r["n_results"] for r in ch_results]),
                "avg_time_ms": np.mean([r["retrieval_time_ms"] for r in ch_results]),
            }
    
    return retrieval_results, stats


# ============================
# Part 3: RAG 生成测试 (调用 LLM, 慢)
# ============================

def run_rag_generation(embedder, vectorstore, anomaly_segments, n_generate=20):
    """为异常段生成 RAG 解释 (调用 LLM)"""
    print(f"\n{'='*60}")
    print(f"Part 3: RAG 生成 (LLM解释 {n_generate} 个异常段)")
    print(f"{'='*60}")
    
    CHANNEL_PHYSICS = {
        "CADC0872": "Magnetometer X-axis 磁力计X轴",
        "CADC0873": "Magnetometer Y-axis 磁力计Y轴",
        "CADC0874": "Magnetometer Z-axis 磁力计Z轴",
        "CADC0884": "Photodiode 1 光电二极管1",
        "CADC0886": "Photodiode 2 光电二极管2",
        "CADC0888": "Photodiode 3 光电二极管3",
        "CADC0890": "Photodiode 4 光电二极管4",
        "CADC0892": "Photodiode 5 光电二极管5",
        "CADC0894": "Photodiode 6 光电二极管6",
    }
    
    # 初始化 LLM
    llm = LLMClient()
    prompts = PromptTemplates()
    
    # 均匀抽样
    channels = sorted(anomaly_segments["channel"].unique())
    samples = []
    per_ch = max(1, n_generate // len(channels))
    for ch in channels:
        ch_segs = anomaly_segments[anomaly_segments["channel"] == ch]
        n_pick = min(per_ch, len(ch_segs))
        picked = ch_segs.sample(n=n_pick, random_state=42)
        samples.append(picked)
    samples_df = pd.concat(samples).head(n_generate)
    
    results = []
    total_retrieval_time = 0
    total_generation_time = 0
    
    for i, (_, row) in enumerate(samples_df.iterrows()):
        ch = row["channel"]
        seg = row["segment"]
        ch_desc = CHANNEL_PHYSICS.get(ch, ch)
        is_strong = ch in {"CADC0872", "CADC0873", "CADC0874"}
        
        query = f"satellite telemetry anomaly {ch_desc} segment {seg}"
        
        # Step 1: 检索
        t0 = time.time()
        search_results = vectorstore.search_with_context(query, top_k=5, max_tokens=2000)
        retrieval_time = time.time() - t0
        context, sources = search_results
        
        # Step 2: 生成
        anomaly_type = "unusual_shapes"  # 默认
        channel_id = ch
        anomaly_desc = f"通道 {ch} ({ch_desc}) 在 segment {seg} 出现异常信号"
        
        prompt_data = prompts.get_anomaly_analysis_prompt(
            channel_id=channel_id,
            anomaly_type=anomaly_type,
            anomaly_description=anomaly_desc,
            context=context
        )
        
        t0 = time.time()
        try:
            answer = llm.generate(
                prompt=prompt_data["user"],
                system_prompt=prompt_data["system"],
                max_tokens=2048
            )
            generation_time = time.time() - t0
            success = True
        except Exception as e:
            answer = f"生成失败: {e}"
            generation_time = time.time() - t0
            success = False
        
        total_retrieval_time += retrieval_time
        total_generation_time += generation_time
        
        # 来源信息
        src_list = []
        for s in sources:
            src_list.append({
                "filename": s["metadata"].get("filename", "?"),
                "page": s["metadata"].get("page", "?"),
                "score": s["score"],
            })
        
        result = {
            "segment": int(seg),
            "channel": ch,
            "channel_desc": ch_desc,
            "is_strong_channel": is_strong,
            "anomaly_type": anomaly_type,
            "answer": answer,
            "sources": src_list,
            "retrieval_time_s": retrieval_time,
            "generation_time_s": generation_time,
            "success": success,
        }
        results.append(result)
        
        status = "[OK]" if success else "[FAIL]"
        print(f"  [{i+1}/{n_generate}] {status} seg={seg} {ch} | 检索={retrieval_time:.1f}s 生成={generation_time:.1f}s")
    
    # 统计
    successful = [r for r in results if r["success"]]
    stats = {
        "n_total": len(results),
        "n_success": len(successful),
        "n_failed": len(results) - len(successful),
        "avg_retrieval_time_s": np.mean([r["retrieval_time_s"] for r in results]),
        "avg_generation_time_s": np.mean([r["generation_time_s"] for r in successful]) if successful else 0,
        "avg_total_time_s": np.mean([r["retrieval_time_s"] + r["generation_time_s"] for r in successful]) if successful else 0,
        "total_time_s": total_retrieval_time + total_generation_time,
        "avg_answer_length": np.mean([len(r["answer"]) for r in successful]) if successful else 0,
        "avg_sources_per_query": np.mean([len(r["sources"]) for r in results]),
    }
    
    return results, stats


# ============================
# Part 4: 生成样例报告
# ============================

def generate_example_reports(rag_results, output_path):
    """生成3份样例报告 (给答辩用)"""
    successful = [r for r in rag_results if r["success"]]
    if not successful:
        print("没有成功的RAG结果，跳过报告生成")
        return
    
    # 选3份: 1个强通道 + 1个弱通道 + 1个score最高的
    strong = [r for r in successful if r["is_strong_channel"]]
    weak = [r for r in successful if not r["is_strong_channel"]]
    
    examples = []
    if strong:
        examples.append(("强通道示例 (磁力计)", strong[0]))
    if weak:
        examples.append(("弱通道示例 (光电二极管)", weak[0]))
    if len(successful) > 2:
        examples.append(("完整分析示例", successful[min(2, len(successful)-1)]))
    
    report = "# RAG 异常分析样例报告\n\n"
    report += f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    for title, r in examples:
        report += f"---\n\n## {title}\n\n"
        report += f"- **通道**: {r['channel']} ({r['channel_desc']})\n"
        report += f"- **Segment**: {r['segment']}\n"
        report += f"- **强通道**: {'是' if r['is_strong_channel'] else '否'}\n"
        report += f"- **检索耗时**: {r['retrieval_time_s']:.2f}s\n"
        report += f"- **生成耗时**: {r['generation_time_s']:.2f}s\n\n"
        report += f"### 检索来源 ({len(r['sources'])}条)\n\n"
        for s in r["sources"]:
            report += f"- {s['filename']} (p.{s['page']}, score={s['score']:.3f})\n"
        report += f"\n### RAG 生成的分析报告\n\n{r['answer']}\n\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"样例报告已保存: {output_path}")


# ============================
# Main
# ============================

def main():
    start_time = time.time()
    
    print("="*60)
    print("RAG 实验 — 2026-05-01")
    print("="*60)
    
    # Step 1: 加载数据
    print("\nPart 1: 加载数据...")
    segments_df, anomaly_segments = load_detection_results()
    
    # Step 2: 初始化 RAG 组件
    print("\n初始化 RAG 组件...")
    embedder = BGE_M3_Embedder()
    vectorstore = FAISSVectorStore(embedder=embedder)
    loaded = vectorstore.load()
    print(f"FAISS 索引加载: {'成功' if loaded else '失败'}")
    print(f"索引文档块数: {vectorstore.index.ntotal if vectorstore.index else 0}")
    
    # Step 3: 检索质量测试 (快速, 不调 LLM)
    retrieval_results, retrieval_stats = test_retrieval_quality(
        embedder, vectorstore, anomaly_segments, n_samples=30
    )
    
    # 保存检索统计
    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "rag_retrieval_stats.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n检索统计已保存: {results_dir / 'rag_retrieval_stats.json'}")
    
    # Step 4: RAG 生成 (调 LLM, 慢)
    rag_results, rag_stats = run_rag_generation(
        embedder, vectorstore, anomaly_segments, n_generate=20
    )
    
    # 保存完整结果
    with open(results_dir / "anomaly_rag_results.json", "w", encoding="utf-8") as f:
        json.dump(rag_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"RAG结果已保存: {results_dir / 'anomaly_rag_results.json'}")
    
    with open(results_dir / "rag_generation_stats.json", "w", encoding="utf-8") as f:
        json.dump(rag_stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"生成统计已保存: {results_dir / 'rag_generation_stats.json'}")
    
    # Step 5: 样例报告
    generate_example_reports(rag_results, results_dir / "rag_example_report.md")
    
    # 汇总
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"实验完成! 总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'='*60}")
    print(f"\n检索统计:")
    print(f"  平均top-1分数: {retrieval_stats['avg_top_score']:.4f}")
    print(f"  平均检索耗时: {retrieval_stats['avg_retrieval_time_ms']:.0f}ms")
    print(f"  来源多样性: {retrieval_stats['source_diversity']}")
    print(f"\n生成统计:")
    print(f"  成功率: {rag_stats['n_success']}/{rag_stats['n_total']}")
    print(f"  平均生成耗时: {rag_stats['avg_generation_time_s']:.1f}s")
    print(f"  平均回答长度: {rag_stats['avg_answer_length']:.0f}字")


if __name__ == "__main__":
    main()

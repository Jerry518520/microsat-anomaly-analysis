"""RAG optimization test with Volcengine DeepSeek V3.2 + enhanced prompts + experiment knowledge."""
import os, sys, time

os.environ['VOLCENGINE_API_KEY'] = 'ark-e1423552-067c-4537-80a2-ad6c764c3bbe-52da8'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

os.chdir(r'F:\微小卫星项目\microsat-anomaly-analysis')
sys.path.insert(0, '.')

import yaml
with open('configs/rag_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

from sentence_transformers import SentenceTransformer
import faiss, pickle, requests

# Load components
print('Loading embedder...')
embedder = SentenceTransformer(config['embedding']['model_name'], device='cuda')
embedder.max_seq_length = 512

print('Loading FAISS index...')
save_dir = config['vectorstore']['persist_directory']
index = faiss.read_index(os.path.join(save_dir, 'faiss_index.bin'))
with open(os.path.join(save_dir, 'documents_metadata.pkl'), 'rb') as f:
    metadata_list = pickle.load(f)
print(f'Index: {index.ntotal} vectors')

API_KEY = os.environ['VOLCENGINE_API_KEY']
headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}

# Enhanced system prompt
enhanced_system = """你是一个卫星异常诊断专家，专门分析OPS-SAT微小卫星的遥测数据异常。
请基于提供的卫星领域知识，给出专业、准确的异常原因分析。

## OPS-SAT 遥测通道速查
9个遥测通道分为两类：
- **磁力计(强通道)**：CADC0872=磁力计X轴, CADC0873=磁力计Y轴, CADC0874=磁力计Z轴
  - 异常含义：磁干扰、磁力矩器故障、姿态控制异常
- **光电二极管(弱通道)**：CADC0884/0886/0888/0890/0892/0894 = 光电二极管1-6角度
  - 异常含义：姿态偏差、传感器遮挡、光照条件变化

## 异常类型分类
1. Unusual shapes(异常形状) 2. Peaks(尖峰) 3. Zero values(零值) 4. Gaps(数据间隙)
CADC0874通道异常以"长数据间隙"为主。

## 回答要求
1. 必须基于提供的知识片段进行分析，不编造
2. 明确标注知识来源（文档名 + 页码）
3. 如果知识片段不足，请说明"现有知识库信息不足"，并给出基于领域常识的推理
4. 使用中文回答，保持专业性和可读性
5. 分析应包含：异常类型判断→可能原因(按可能性排序)→影响范围→建议措施
6. 如果查询涉及具体CADC通道，先说明该通道的物理含义再分析"""

# Test queries
test_queries = [
    'OPS-SAT卫星CADC0874磁力计Z轴通道出现异常读数，可能的原因是什么？',
    '光电二极管传感器出现零值异常可能的原因',
    'CADC0872磁力计X轴检测到异常形状信号，如何分析？',
    'OPS-SAT异常检测系统使用什么算法，效果如何？',
]

results = []

for qi, q in enumerate(test_queries):
    print(f'\n--- Query {qi+1}/{len(test_queries)} ---')
    # Retrieve
    q_emb = embedder.encode([q], normalize_embeddings=True)
    scores, ids = index.search(q_emb.astype('float32'), 5)
    context_parts = []
    sources = []
    for score, idx in zip(scores[0], ids[0]):
        if score >= 0.3:
            meta = metadata_list[idx]
            src = meta['metadata'].get('source_name', '?')
            page = meta['metadata'].get('page', '?')
            content = meta['content'][:600].replace('\xa0', ' ')
            context_parts.append(f'[Source: {src}, Page {page}, Score: {score:.3f}]\n{content}')
            sources.append(f'{src} p{page}')

    context = '\n\n---\n\n'.join(context_parts)
    print(f'Retrieved {len(context_parts)} chunks: {", ".join(sources[:3])}')

    user_prompt = f"""请基于以下卫星领域知识，回答用户的问题：

参考知识：
{context}

用户问题：{q}

请按以下结构回答：
1. **通道定位**（如涉及CADC编号）：该通道属于什么传感器，测量什么物理量
2. **异常类型判断**：属于哪种异常类型
3. **可能原因**（按可能性排序）
4. **影响评估**
5. **建议措施**

每个分析点必须标注知识来源。"""

    payload = {
        'model': 'deepseek-v3-2-251201',
        'messages': [
            {'role': 'system', 'content': enhanced_system},
            {'role': 'user', 'content': user_prompt}
        ],
        'max_tokens': 1024,
        'temperature': 0.1
    }

    try:
        t0 = time.time()
        resp = requests.post('https://ark.cn-beijing.volces.com/api/v3/chat/completions',
                            headers=headers, json=payload, timeout=120)
        latency = time.time() - t0
        if resp.status_code == 200:
            answer = resp.json()['choices'][0]['message']['content']
            usage = resp.json().get('usage', {})
            print(f'LLM OK ({latency:.1f}s, tokens: {usage.get("total_tokens", "?")})')
        else:
            answer = f'[ERROR] HTTP {resp.status_code}: {resp.text[:200]}'
            latency = -1
            print(f'LLM ERROR: HTTP {resp.status_code}')
    except Exception as e:
        answer = f'[ERROR] {e}'
        latency = -1
        print(f'LLM ERROR: {e}')

    results.append({
        'query': q,
        'sources': sources,
        'answer': answer,
        'latency': latency
    })

# Write results
with open('test_rag_optimized.txt', 'w', encoding='utf-8') as f:
    f.write('RAG优化后测试结果（火山引擎DeepSeek V3.2 + Prompt增强 + 实验知识入库）\n')
    f.write('=' * 60 + '\n\n')
    for i, r in enumerate(results, 1):
        f.write(f'## 测试 {i}\n')
        f.write(f'问题: {r["query"]}\n')
        f.write(f'检索来源: {", ".join(r["sources"])}\n')
        f.write(f'LLM响应时间: {r["latency"]:.1f}s\n')
        f.write(f'回答:\n{r["answer"]}\n')
        f.write('-' * 60 + '\n\n')

print('\nResults written to test_rag_optimized.txt')

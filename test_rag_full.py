"""Full RAG pipeline test - output to file for UTF-8 reading."""
import os, sys

os.environ['NVIDIA_API_KEY'] = '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'
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
embedder = SentenceTransformer(config['embedding']['model_name'], device='cuda')
embedder.max_seq_length = 512

save_dir = config['vectorstore']['persist_directory']
index = faiss.read_index(os.path.join(save_dir, 'faiss_index.bin'))
with open(os.path.join(save_dir, 'documents_metadata.pkl'), 'rb') as f:
    metadata_list = pickle.load(f)

API_KEY = os.environ['NVIDIA_API_KEY']
headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}

# Test queries
test_queries = [
    'OPS-SAT卫星CADC0874磁力计Z轴通道出现异常读数，可能的原因是什么？',
    'What are the common anomaly types in satellite magnetometer data?',
    'ADCS安全模式下反作用飞轮和磁力矩器如何协同工作？',
    '光电二极管传感器出现零值异常可能的原因',
]

results = []

for q in test_queries:
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
            content = meta['content'][:500].replace('\xa0', ' ')
            context_parts.append(f'[Source: {src}, Page {page}, Score: {score:.3f}]\n{content}')
            sources.append(f'{src} p{page}')

    context = '\n\n---\n\n'.join(context_parts)

    # Generate
    system_prompt = """你是卫星遥测数据分析专家。基于提供的知识库上下文回答问题。
始终引用来源。如果上下文信息不足，请明确说明。用中文回答。"""

    user_prompt = f"""基于以下知识库上下文，回答问题：

## 上下文：
{context}

## 问题：
{q}

请详细解释异常原因并引用具体来源。"""

    payload = {
        'model': config['llm']['model'],
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        'max_tokens': 1024,
        'temperature': 0.1
    }

    try:
        resp = requests.post(config['llm']['api_base'], headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            answer = resp.json()['choices'][0]['message']['content']
        else:
            answer = f'[ERROR] HTTP {resp.status_code}: {resp.text[:200]}'
    except Exception as e:
        answer = f'[ERROR] {e}'

    results.append({
        'query': q,
        'sources': sources,
        'answer': answer
    })

# Write results to file
with open('test_rag_results.txt', 'w', encoding='utf-8') as f:
    f.write('RAG端到端测试结果\n')
    f.write('=' * 60 + '\n\n')
    for i, r in enumerate(results, 1):
        f.write(f'## 测试 {i}\n')
        f.write(f'问题: {r["query"]}\n')
        f.write(f'检索来源: {", ".join(r["sources"])}\n')
        f.write(f'回答:\n{r["answer"]}\n')
        f.write('-' * 60 + '\n\n')

print('Results written to test_rag_results.txt')

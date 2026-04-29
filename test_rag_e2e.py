"""End-to-end RAG pipeline test with NVIDIA API."""
import os, sys

# Set API key
os.environ['NVIDIA_API_KEY'] = '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'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

os.chdir(r'F:\微小卫星项目\microsat-anomaly-analysis')
sys.path.insert(0, '.')

# Load config
import yaml
with open('configs/rag_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

LLM_MODEL = config['llm']['model']
LLM_URL = config['llm']['api_base']  # already full URL
API_KEY = os.environ['NVIDIA_API_KEY']

print('=== Step 1: Load Embedder ===')
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer(config['embedding']['model_name'], device='cuda')
embedder.max_seq_length = 512
print(f'Embedder loaded: {config["embedding"]["model_name"]}')

print('\n=== Step 2: Load FAISS Index ===')
import faiss, pickle
save_dir = config['vectorstore']['persist_directory']
index = faiss.read_index(os.path.join(save_dir, 'faiss_index.bin'))
with open(os.path.join(save_dir, 'documents_metadata.pkl'), 'rb') as f:
    metadata_list = pickle.load(f)
print(f'Index: {index.ntotal} vectors, dim={index.d}')

print('\n=== Step 3: Test LLM Client ===')
import requests
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}
payload = {
    'model': LLM_MODEL,
    'messages': [{'role': 'user', 'content': 'Say "API connection successful" in exactly those words.'}],
    'max_tokens': 20,
    'temperature': 0.1
}
try:
    resp = requests.post(LLM_URL, headers=headers, json=payload, timeout=30)
    if resp.status_code == 200:
        msg = resp.json()['choices'][0]['message']['content']
        print(f'LLM response: {msg}')
        print('NVIDIA API connection: SUCCESS')
    else:
        print(f'LLM error: HTTP {resp.status_code}')
        print(resp.text[:500])
except Exception as e:
    print(f'LLM connection failed: {e}')

print('\n=== Step 4: Full RAG Query ===')
query = 'magnetometer anomaly on OPS-SAT satellite, what could cause unusual readings on CADC0874 Z-axis channel?'

# Retrieve
q_emb = embedder.encode([query], normalize_embeddings=True)
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
print(f'Retrieved {len(context_parts)} relevant chunks')
for s in sources:
    print(f'  - {s}')

# Generate
system_prompt = """You are an expert satellite telemetry analyst. Answer questions about satellite anomalies based on the provided context. 
Always cite your sources. If the context doesn't contain enough information, say so clearly.
Answer in Chinese (Simplified)."""

user_prompt = f"""Based on the following knowledge base context, answer the question:

## Context:
{context}

## Question:
{query}

Please explain what could cause the anomaly and cite specific sources."""

rag_payload = {
    'model': LLM_MODEL,
    'messages': [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ],
    'max_tokens': 1024,
    'temperature': 0.1
}

try:
    resp = requests.post(LLM_URL, headers=headers, json=rag_payload, timeout=60)
    if resp.status_code == 200:
        answer = resp.json()['choices'][0]['message']['content']
        # Safe print
        answer_safe = answer.encode('ascii', 'replace').decode('ascii')
        print(f'\nRAG Answer:\n{answer_safe}')
    else:
        print(f'RAG generation error: HTTP {resp.status_code}')
        print(resp.text[:500])
except Exception as e:
    print(f'RAG generation failed: {e}')

print('\n=== TEST COMPLETE ===')

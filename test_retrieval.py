"""Quick retrieval test for built FAISS index."""
import os, sys, pickle, yaml
import faiss
import numpy as np

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.chdir(r'F:\微小卫星项目\microsat-anomaly-analysis')

# Load config
with open('configs/rag_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Load index + metadata
save_dir = config['vectorstore']['persist_directory']
index = faiss.read_index(os.path.join(save_dir, 'faiss_index.bin'))
with open(os.path.join(save_dir, 'documents_metadata.pkl'), 'rb') as f:
    metadata_list = pickle.load(f)
print(f'Index: {index.ntotal} vectors, dim={index.d}')
print(f'Metadata: {len(metadata_list)} entries')

# Stats by source
from collections import Counter
source_counts = Counter(m['metadata'].get('source_name', '?') for m in metadata_list)
for src, cnt in source_counts.most_common():
    print(f'  {src}: {cnt} chunks')

# Load embedder
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer(config['embedding']['model_name'], device='cuda')
embedder.max_seq_length = 512

# Test queries
print('\n=== Retrieval Test ===')
test_queries = [
    'magnetometer anomaly detection',
    'ADCS safe mode reaction wheel',
    'photodiode zero value gap',
    'satellite attitude control magnetic torquer',
    'CADC0874 magnetometer Z axis',
]
for q in test_queries:
    q_emb = embedder.encode([q], normalize_embeddings=True)
    scores, ids = index.search(q_emb.astype('float32'), 5)
    print(f'\nQ: "{q}"')
    for i, (score, idx) in enumerate(zip(scores[0], ids[0])):
        if score < 0.3:
            break
        meta = metadata_list[idx]
        src = meta['metadata'].get('source_name', '?')
        page = meta['metadata'].get('page', '?')
        # Safe print - replace non-ascii
        preview = meta['content'][:80].replace('\n', ' ').replace('\xa0', ' ')
        preview = preview.encode('ascii', 'replace').decode('ascii')
        print(f'  [{i+1}] score={score:.4f} | {src} p{page} | {preview}...')

print('\n=== DONE ===')

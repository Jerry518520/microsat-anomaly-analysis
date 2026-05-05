"""Build FAISS index from knowledge base PDFs/MDs and test retrieval."""
import os, sys, yaml, pickle
import faiss, numpy as np

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Change to project root
os.chdir(r'F:\微小卫星项目\microsat-anomaly-analysis')

# 1. Load config
with open('configs/rag_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
print('Config loaded')

# 2. Load PDFs and MDs
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

all_docs = []
kb_list = config['document']['knowledge_base']
chunk_cfg = config['document']['chunking']

for kb in kb_list:
    file_path = kb['file_path']
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.getcwd(), file_path)
    if not os.path.exists(file_path):
        print(f'SKIP: {kb["name"]} - not found: {file_path}')
        continue
    file_size = os.path.getsize(file_path)/1e6
    print(f'Loading: {kb["name"]} ({file_size:.1f}MB)')
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif ext == '.md':
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        docs = [Document(
            page_content=content,
            metadata={'source': file_path, 'page': 0, 'page_label': '1', 'total_pages': 1}
        )]
    else:
        print(f'  SKIP unsupported format: {ext}')
        continue
    for d in docs:
        d.metadata['source_name'] = kb['name']
        d.metadata['priority'] = kb['priority']
        d.metadata['language'] = kb['language']
        d.metadata['filename'] = os.path.basename(file_path)
    all_docs.extend(docs)
    print(f'  -> {len(docs)} pages')

print(f'\nTotal pages loaded: {len(all_docs)}')

# 3. Split
separators = ['\n\n', '\n', '\u3002', '.', ' ', '']
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_cfg['chunk_size'],
    chunk_overlap=chunk_cfg['chunk_overlap'],
    separators=separators
)
chunks = splitter.split_documents(all_docs)
print(f'After splitting: {len(chunks)} chunks')

# 4. Embed
print('\nLoading BGE-M3 model...')
from sentence_transformers import SentenceTransformer
model_path = config['embedding']['model_name']
embedder = SentenceTransformer(model_path, device='cuda')
embedder.max_seq_length = config['embedding']['max_length']

print('Encoding chunks (this may take a few minutes)...')
texts = [c.page_content for c in chunks]
embeddings = embedder.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
print(f'Embeddings shape: {embeddings.shape}')

# 5. Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity after normalization
index.add(embeddings.astype('float32'))
print(f'FAISS index built: {index.ntotal} vectors, dim={dim}')

# 6. Persist
save_dir = config['vectorstore']['persist_directory']
os.makedirs(save_dir, exist_ok=True)
faiss.write_index(index, os.path.join(save_dir, 'faiss_index.bin'))
metadata_list = [{'content': c.page_content, 'metadata': c.metadata} for c in chunks]
with open(os.path.join(save_dir, 'documents_metadata.pkl'), 'wb') as f:
    pickle.dump(metadata_list, f)
print(f'Saved index to {save_dir}/faiss_index.bin')
print(f'Saved metadata to {save_dir}/documents_metadata.pkl')

# 7. Retrieval test
print('\n=== Retrieval Test ===')
test_queries = [
    'magnetometer anomaly detection',
    'ADCS safe mode reaction wheel',
    'photodiode zero value gap anomaly',
    'satellite attitude control magnetic torquer',
]
for q in test_queries:
    q_emb = embedder.encode([q], normalize_embeddings=True)
    scores, ids = index.search(q_emb.astype('float32'), 3)
    print(f'\nQ: "{q}"')
    for i, (score, idx) in enumerate(zip(scores[0], ids[0])):
        meta = metadata_list[idx]
        src = meta['metadata'].get('source_name', '?')
        page = meta['metadata'].get('page', '?')
        preview = meta['content'][:80].replace('\n', ' ')
        print(f'  [{i+1}] score={score:.4f} | {src} p{page} | {preview}...')

print('\n=== ALL DONE ===')

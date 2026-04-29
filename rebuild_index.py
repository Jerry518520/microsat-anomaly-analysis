"""Rebuild FAISS index with PDF + Markdown knowledge entries."""
import os, sys, yaml, pickle, textwrap
import faiss, numpy as np

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

os.chdir(r'F:\微小卫星项目\microsat-anomaly-analysis')

# 1. Load config
with open('configs/rag_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
print('Config loaded')

# 2. Load documents (PDF + MD)
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

all_docs = []
kb_list = config['document']['knowledge_base']
chunk_cfg = config['document']['chunking']

for kb in kb_list:
    pdf_path = kb['file_path']
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.join(os.getcwd(), pdf_path)
    if not os.path.exists(pdf_path):
        print(f'SKIP: {kb["name"]} - not found: {pdf_path}')
        continue
    
    ext = os.path.splitext(pdf_path)[1].lower()
    print(f'Loading: {kb["name"]} ({os.path.getsize(pdf_path)/1e3:.0f}KB, {ext})')
    
    try:
        if ext == '.pdf':
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
        elif ext in ('.md', '.markdown', '.txt'):
            # Simple text loading for markdown
            with open(pdf_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Split by --- separators (H2 sections)
            sections = content.split('\n---\n')
            docs = []
            for i, section in enumerate(sections):
                section = section.strip()
                if section:
                    docs.append(Document(
                        page_content=section,
                        metadata={'page': i, 'total_pages': len(sections)}
                    ))
        else:
            print(f'  SKIP: unsupported format {ext}')
            continue
        
        for d in docs:
            d.metadata['source_name'] = kb['name']
            d.metadata['priority'] = kb['priority']
            d.metadata['language'] = kb['language']
        all_docs.extend(docs)
        print(f'  -> {len(docs)} sections/pages')
    except Exception as e:
        print(f'  ERROR loading {kb["name"]}: {e}')

print(f'\nTotal sections/pages loaded: {len(all_docs)}')

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

print('Encoding chunks (CUDA)...')
texts = [c.page_content for c in chunks]
embeddings = embedder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
print(f'Embeddings shape: {embeddings.shape}')

# 5. Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings.astype('float32'))
print(f'FAISS index built: {index.ntotal} vectors, dim={dim}')

# 6. Persist
save_dir = config['vectorstore']['persist_directory']
os.makedirs(save_dir, exist_ok=True)
faiss.write_index(index, os.path.join(save_dir, 'faiss_index.bin'))
metadata_list = [{'content': c.page_content, 'metadata': c.metadata} for c in chunks]
with open(os.path.join(save_dir, 'documents_metadata.pkl'), 'wb') as f:
    pickle.dump(metadata_list, f)
print(f'Saved index ({os.path.getsize(os.path.join(save_dir, "faiss_index.bin"))/1e6:.1f}MB)')
print(f'Saved metadata ({os.path.getsize(os.path.join(save_dir, "documents_metadata.pkl"))/1e6:.1f}MB)')

# 7. Quick retrieval test
print('\n=== Retrieval Test ===')
test_queries = [
    'CADC0874 magnetometer Z-axis anomaly',
    'OPS-SAT ADCS hardware components',
    'photodiode zero value anomaly cause',
    'Isolation Forest anomaly detection results',
    '9-channel physical meaning CADC',
]
for q in test_queries:
    q_emb = embedder.encode([q], normalize_embeddings=True)
    scores, ids = index.search(q_emb.astype('float32'), 3)
    print(f'\nQ: "{q}"')
    for i, (score, idx) in enumerate(zip(scores[0], ids[0])):
        meta = metadata_list[idx]
        src = meta['metadata'].get('source_name', '?')
        page = meta['metadata'].get('page', '?')
        preview = meta['content'][:100].replace('\n', ' ').replace('\xa0', ' ')
        try:
            print(f'  [{i+1}] score={score:.4f} | {src} p{page} | {preview}...')
        except UnicodeEncodeError:
            print(f'  [{i+1}] score={score:.4f} | {src} p{page} | (content ok, print encoding issue)')

print('\n=== INDEX REBUILD COMPLETE ===')

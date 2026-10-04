"""Build FAISS index from knowledge base PDFs/MDs and test retrieval."""
import os, sys, yaml, pickle
import faiss, numpy as np, torch

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# Change to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载 .env（EMBEDDING_MODEL_PATH 等在此配置；参见 src/ui/app.py 的做法）
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), '.env'), override=True)
except ImportError:
    print('WARN: python-dotenv 未安装，EMBEDDING_MODEL_PATH 将取不到值')

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

# 4b.剔除低质量片段（PDF 换页残片 / 目录页）
# 这类 chunk 维度低、极易与任意查询相似，实测曾多次占据 top1
# （如 ITU EN p69 的孤立句子 'returned due to non-completion of coordination.'）
# 规则定义在 src/rag/retrieval_policy.py，与检索时过滤保持一致
sys.path.insert(0, os.getcwd())
from src.rag.retrieval_policy import is_low_quality

retrieval_cfg = config.get('retrieval', {})
min_chunk_chars = retrieval_cfg.get('min_chunk_chars', 80)
max_junk_ratio = retrieval_cfg.get('max_junk_ratio', 0.55)

before = len(chunks)
kept, dropped = [],0
for c in chunks:
    if is_low_quality(c.page_content, min_chunk_chars, max_junk_ratio):
        dropped += 1
    else:
        kept.append(c)
chunks = kept
print(f'Filtered {dropped} low-quality chunks (min_chars={min_chunk_chars}, '
      f'max_junk_ratio={max_junk_ratio}), kept {len(chunks)}')

# 4. Embed
print('\nLoading BGE-M3 model...')

# 模型路径解析：复用 src/rag/embedding.py 的**唯一实现**，不再各读各的。
# 修复记录：原代码在此处自带一份 "EMBEDDING_MODEL_PATH or config['model_name']"
# 逻辑，与 src/rag/embedding.py:73 的同名逻辑重复。两者虽在字符串层面碰巧
# 一致，但没有任何机制保证它们**永远**一致——一旦有人只改 configs/rag_config.yaml
# 的 model_name，build_index 与运行时就会加载不同权重，已建索引与查询向量
# 分属不同向量空间，检索静默劣化。现改为单一真相源。
sys.path.insert(0, os.getcwd())
from src.rag.embedding import resolve_embedding_model_path, EmbeddingModelPathError
from sentence_transformers import SentenceTransformer

try:
    model_path = resolve_embedding_model_path(config.get('embedding', {}))
except EmbeddingModelPathError as e:
    print(f'\n[FATAL] 嵌入模型路径不可用：\n{e}')
    sys.exit(1)
print(f'Model path: {model_path}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
embedder = SentenceTransformer(model_path, device=device)
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

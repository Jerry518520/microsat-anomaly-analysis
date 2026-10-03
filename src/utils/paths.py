"""
FAISS 向量索引路径 —— 唯一定义源（P0-A6）

## 为什么需要这个模块

历史上索引路径在 4 处各写一遍字符串，其中 `src/ui/pages/detection.py` 写错了
（`data/faiss_index/index.faiss`，该目录从不存在）：

    src/ui/pages/detection.py   data/faiss_index/index.faiss   ← 错，目录不存在
    src/api/routes/detection.py data/vectorstore/faiss_index.bin
    src/api/routes/dashboard.py data/vectorstore/faiss_index.bin
    src/ui/app.py               data/vectorstore/faiss_index.bin

更危险的是错误路径被 `if os.path.exists(...)` 包着，且异常被 `except: pass`
吞掉，导致索引加载失败时**不报错、不提示**，页面照常渲染，只是 Chunk 数显示
"未索引"。用户以为在用 RAG 解释，实际处于无 RAG 的降级状态——这是静默降级，
比直接崩溃更难发现。

真实路径由 `configs/rag_config.yaml` 的 `vectorstore.persist_directory`
（`data/vectorstore`）+ 文件名 `faiss_index.bin` 决定，与
`src/rag/vectorstore.py` 的 `save()` / `load()` 和 `scripts/build_index.py`
保持一致。

## 约定

新增调用方一律用本模块的函数取路径，不要再手写字符串拼接。
若将来索引目录变更，只改 `configs/rag_config.yaml` +
`FAISS_INDEX_BASENAME` 一处即可。
"""

import os
import shutil
import tempfile

# 与 configs/rag_config.yaml: vectorstore.persist_directory 保持一致
# （相对项目根目录）
FAISS_PERSIST_DIR = "data/vectorstore"

# 与 src/rag/vectorstore.py 的 save()/load() 及 scripts/build_index.py 一致
FAISS_INDEX_BASENAME = "faiss_index.bin"
FAISS_METADATA_BASENAME = "documents_metadata.pkl"

# 本文件位于 <root>/src/utils/paths.py，故上溯两级即项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_faiss_index_path() -> str:
    """返回 FAISS 索引文件的绝对路径。"""
    return os.path.join(PROJECT_ROOT, FAISS_PERSIST_DIR, FAISS_INDEX_BASENAME)


def get_faiss_metadata_path() -> str:
    """返回 FAISS 文档元数据文件的绝对路径。"""
    return os.path.join(PROJECT_ROOT, FAISS_PERSIST_DIR, FAISS_METADATA_BASENAME)


def _has_non_ascii(path: str) -> bool:
    return any(ord(ch) > 127 for ch in path)


def faiss_readable_path(path: str) -> str:
    """把索引路径转成 faiss 在本平台能真正打开的路径。

    ## 为什么要这一步（本轮修复中实测发现的第二个隐性故障）

    faiss 1.14.2 在 Windows 上通过 C++ `fopen(const char*)` 打开索引文件，
    该接口按系统 ANSI 代码页解释字节，**无法处理非 ASCII 路径**。而本项目
    实际部署路径是 `F:\\F盘\\微小卫星项目\\...`，含中文，于是：

        faiss.read_index("F:\\F盘\\...\\faiss_index.bin")
        -> Error in __cdecl faiss::FileIOReader::FileIOReader ... 'f' failed:
           could not open ... for reading: No such file or directory

    文件明明存在（os.path.exists 为 True，20MB），faiss 却报"文件不存在"。
    由于旧代码把这个异常吞掉，表现同样是"Chunk 总数: 未索引"，与索引真的
    缺失**无法区分**——这是比路径写错更隐蔽的一层故障。

    已验证的解法（按优先级）：
      1. 路径全 ASCII → 直接用，零开销（Linux / 纯英文部署路径走这条）
      2. Windows 且路径含非 ASCII → 复制到系统临时目录下的 ASCII 缓存路径后读取。
         实测 20MB 索引复制+读取合计约 0.02s，且以源文件 mtime 作为缓存有效性
         判据，源索引更新后会自动重新复制，不会读到过期索引。

    注：8.3 短路径（GetShortPathNameW）在本机返回的还是原路径（该卷未启用
    短文件名），故不采用。
    """
    if not _has_non_ascii(path):
        return path

    if os.name != "nt":
        # 非 Windows 平台的 faiss 通常能处理 UTF-8 路径，无需暂存
        return path

    cache_dir = os.path.join(tempfile.gettempdir(), "microsat_faiss_cache")
    os.makedirs(cache_dir, exist_ok=True)
    staged = os.path.join(cache_dir, os.path.basename(path))

    # 以源文件 mtime 判定缓存是否有效；源更新后重新复制
    if not (
        os.path.exists(staged)
        and os.path.getmtime(staged) >= os.path.getmtime(path)
    ):
        shutil.copy2(path, staged)

    return staged


def read_faiss_index_status():
    """检查索引是否可用，返回 (状态, chunk 数, 原因) 三元组。

    状态取值：
      "ok"          —— 索引存在且成功读取
      "missing"     —— 索引文件不存在（RAG 无法提供解释）
      "unreadable"  —— 文件存在但 faiss 读取失败（版本不匹配/文件损坏）

    设计意图：调用方**必须**根据状态给出可见反馈（Streamlit 用
    st.warning / st.error），不允许静默降级。历史上正是因为这里
    `except: pass` + 上层 `os.path.exists` 双重掩盖，索引缺失时页面
    看起来一切正常却悄悄丢了 RAG 能力。

    返回原因字符串而非抛异常，是为了让 UI 能把"路径 + 原因"一并展示给
    用户定位问题；本函数不吞异常细节，`unreadable` 分支带上原始报错。
    """
    index_path = get_faiss_index_path()

    if not os.path.exists(index_path):
        return "missing", None, f"未找到索引文件：{index_path}"

    try:
        import faiss
    except ImportError as e:
        return "unreadable", None, f"faiss 未安装，无法读取索引：{e}"

    try:
        idx = faiss.read_index(faiss_readable_path(index_path))
        return "ok", int(idx.ntotal), ""
    except Exception as e:
        return "unreadable", None, f"索引存在但读取失败：{index_path}（{e}）"

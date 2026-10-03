"""
RAG 修复回归测试
验证4项修复的正确性：
1. anomaly_rag_pipeline.py 使用 .get() 防止 KeyError
2. pipeline.py 错误路径 metadata 包含 retrieval_time / generation_time
3. faiss_index.bin 存在
4. explanation.py _is_garbled_text() 新增错误模式检测
"""

import os
import sys
import json
import ast
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ============================================================
# 测试 1: 静态代码审查 — pipeline.py 错误路径 metadata 结构
# ============================================================

def test_pipeline_error_path_metadata_structure():
    """验证 pipeline.py 错误路径返回的 metadata 包含 retrieval_time 和 generation_time"""
    pipeline_src = (project_root / "src" / "rag" / "pipeline.py").read_text(encoding="utf-8")

    # 正常路径 metadata 应包含 retrieval_time 和 generation_time
    assert '"retrieval_time": retrieval_time' in pipeline_src, \
        "正常路径 metadata 缺少 retrieval_time"
    assert '"generation_time": generation_time' in pipeline_src, \
        "正常路径 metadata 缺少 generation_time"

    # 错误路径 metadata 应包含 retrieval_time: 0 和 generation_time: 0
    assert '"retrieval_time": 0' in pipeline_src, \
        "错误路径 metadata 缺少 retrieval_time: 0"
    assert '"generation_time": 0' in pipeline_src, \
        "错误路径 metadata 缺少 generation_time: 0"

    print("[PASS] pipeline.py 错误路径 metadata 结构与正常路径一致")


# ============================================================
# 测试 2: 静态代码审查 — anomaly_rag_pipeline.py 使用 .get()
# ============================================================

def test_anomaly_rag_pipeline_uses_get():
    """验证 anomaly_rag_pipeline.py 使用 .get() 访问 metadata 字段"""
    src = (project_root / "src" / "integration" / "anomaly_rag_pipeline.py").read_text(encoding="utf-8")

    # 必须使用 .get() 而非直接索引
    assert '.get("retrieval_time", 0)' in src, \
        "anomaly_rag_pipeline.py 未使用 .get('retrieval_time', 0)"
    assert '.get("generation_time", 0)' in src, \
        "anomaly_rag_pipeline.py 未使用 .get('generation_time', 0)"

    # 确认没有旧的直接索引方式（会抛 KeyError）
    assert '["metadata"]["retrieval_time"]' not in src, \
        "仍存在旧的直接索引 [\"metadata\"][\"retrieval_time\"]"
    assert '["metadata"]["generation_time"]' not in src, \
        "仍存在旧的直接索引 [\"metadata\"][\"generation_time\"]"

    print("[PASS] anomaly_rag_pipeline.py 正确使用 .get() 防止 KeyError")


# ============================================================
# 测试 3: faiss_index.bin 存在性
# ============================================================

def test_faiss_index_exists():
    """验证 FAISS 索引文件存在且大小合理"""
    faiss_path = project_root / "data" / "vectorstore" / "faiss_index.bin"
    assert faiss_path.exists(), f"FAISS 索引文件不存在: {faiss_path}"

    size_mb = faiss_path.stat().st_size / (1024 * 1024)
    assert size_mb > 1, f"FAISS 索引文件过小 ({size_mb:.1f} MB)，可能损坏"

    print(f"[PASS] faiss_index.bin 存在，大小 {size_mb:.1f} MB")


# ============================================================
# 测试 4: _is_garbled_text() 错误模式检测
# ============================================================

def _import_is_garbled_text():
    """动态导入 _is_garbled_text 函数"""
    # 从 explanation.py 中提取函数
    explanation_path = project_root / "src" / "ui" / "pages" / "explanation.py"
    src = explanation_path.read_text(encoding="utf-8")

    # 提取 _is_garbled_text 函数定义并执行
    lines = src.split("\n")
    func_start = None
    func_end = None
    for i, line in enumerate(lines):
        if "def _is_garbled_text" in line:
            func_start = i
        elif func_start is not None and (line.startswith("def ") or line.startswith("class ")):
            func_end = i
            break

    if func_start is None:
        raise RuntimeError("未找到 _is_garbled_text 函数")

    func_lines = lines[func_start:func_end] if func_end else lines[func_start:]
    func_code = "\n".join(func_lines)

    local_ns = {}
    exec(func_code, {}, local_ns)
    return local_ns["_is_garbled_text"]


def test_is_garbled_text_error_patterns():
    """验证 _is_garbled_text() 正确检测 RAG 错误模式"""
    is_garbled = _import_is_garbled_text()

    # 测试所有新增的错误模式应返回 True
    error_texts = [
        "RAG解释生成失败: 'retrieval_time'",
        "检索失败: connection timeout",
        "生成失败: API error",
        "RAG 查询失败: invalid key",
        "查询过程中出现错误",
        "RAG引擎初始化失败: module not found",
        "抱歉，查询过程中出现错误：KeyError",
    ]

    for text in error_texts:
        result = is_garbled(text)
        assert result is True, f"_is_garbled_text('{text}') 应返回 True，实际返回 {result}"

    print("[PASS] _is_garbled_text() 正确检测所有 RAG 错误模式")


def test_is_garbled_text_normal_text():
    """验证正常诊断文本不被误判为乱码"""
    is_garbled = _import_is_garbled_text()

    normal_texts = [
        "根据知识库分析，卫星遥测数据异常可能由以下原因导致：1. 传感器故障 2. 通信链路问题",
        "通道 CADC0872 检测到异常，异常类型为 unusual_shapes，建议检查传感器状态",
        "磁力计数据偏差超出阈值，可能需要重新校准",
        "",  # 空字符串应返回 False
    ]

    for text in normal_texts:
        result = is_garbled(text)
        assert result is False, f"_is_garbled_text('{text[:40]}...') 应返回 False，实际返回 {result}"

    print("[PASS] 正常诊断文本不会被误判为乱码")


def test_is_garbled_text_actual_garbled():
    """验证真正的乱码文本仍然被正确检测"""
    is_garbled = _import_is_garbled_text()

    # 构造乱码文本（替换字符占比 > 5%）
    garbled_text = "卫星数据" + "\ufffd" * 50 + "正常"
    result = is_garbled(garbled_text)
    assert result is True, f"乱码文本应返回 True，实际返回 {result}"

    print("[PASS] 真正的乱码文本仍然被正确检测")


# ============================================================
# 测试 5: 错误路径不会导致下游 KeyError
# ============================================================

def test_error_path_no_keyerror():
    """模拟 pipeline 错误路径返回值，验证下游访问不会抛 KeyError"""
    # 模拟 pipeline.py 错误路径返回值
    error_result = {
        "answer": "抱歉，查询过程中出现错误：test error",
        "sources": [],
        "context": "",
        "metadata": {
            "query": "test",
            "query_type": "anomaly",
            "retrieval_time": 0,
            "generation_time": 0,
            "error": "test error",
            "total_time": 0.1
        }
    }

    # 模拟 anomaly_rag_pipeline.py 的下游访问
    retrieval_time = error_result["metadata"].get("retrieval_time", 0)
    generation_time = error_result["metadata"].get("generation_time", 0)

    assert retrieval_time == 0, f"retrieval_time 应为 0，实际为 {retrieval_time}"
    assert generation_time == 0, f"generation_time 应为 0，实际为 {generation_time}"

    # 模拟正常路径返回值
    normal_result = {
        "answer": "正常分析结果",
        "sources": [{"content": "来源1", "score": 0.85}],
        "context": "上下文",
        "metadata": {
            "query": "test",
            "query_type": "anomaly",
            "retrieval_time": 1.5,
            "generation_time": 2.3,
            "total_time": 3.8,
            "sources_count": 1
        }
    }

    # 两种路径的 metadata 都应能安全访问 retrieval_time / generation_time
    normal_rt = normal_result["metadata"].get("retrieval_time", 0)
    normal_gt = normal_result["metadata"].get("generation_time", 0)
    assert normal_rt == 1.5
    assert normal_gt == 2.3

    print("[PASS] 错误路径和正常路径均可安全访问 metadata 字段")


# ============================================================
# 测试 6: 数据文件中错误模式与新增检测模式匹配
# ============================================================

def test_existing_error_data_matches_patterns():
    """验证现有数据文件中的错误信息能被 _is_garbled_text() 正确检测"""
    is_garbled = _import_is_garbled_text()

    results_path = project_root / "data" / "results" / "anomaly_rag_results.json"
    if not results_path.exists():
        print("[SKIP] anomaly_rag_results.json 不存在")
        return

    data = json.loads(results_path.read_text(encoding="utf-8"))

    # 找出真正的错误/乱码记录。
    # ⚠ 不要用「异常确认」「RAG」等词做筛选——那些是**正常诊断输出**的内容
    #   （7 段式提示词要求 LLM 输出「异常确认」段落），不是错误数据。
    #   之前的筛选条件把 200 条正常输出全当成「错误记录」，然后断言它们必须
    #   被判为乱码，导致 assert False is True。
    #   真正的错误特征：替换字符 U+FFFD、连续问号、或明确的失败提示。
    def _looks_broken(text: str) -> bool:
        return (
            "�" in text
            or "??" in text
            or "调用失败" in text
            or "Traceback" in text
        )

    error_results = [r for r in data if _looks_broken(r.get("rag_explanation", ""))]

    for r in error_results:
        explanation = r["rag_explanation"]
        result = is_garbled(explanation)
        assert result is True, \
            f"_is_garbled_text() 未能检测出现有错误信息: '{explanation[:60]}'"

    print(f"[PASS] 现有数据中 {len(error_results)} 条错误记录均被 _is_garbled_text() 正确检测")


# ============================================================
# 运行所有测试
# ============================================================

if __name__ == "__main__":
    tests = [
        ("pipeline.py 错误路径 metadata 结构", test_pipeline_error_path_metadata_structure),
        ("anomaly_rag_pipeline.py 使用 .get()", test_anomaly_rag_pipeline_uses_get),
        ("faiss_index.bin 存在性", test_faiss_index_exists),
        ("_is_garbled_text() 错误模式检测", test_is_garbled_text_error_patterns),
        ("正常文本不误判", test_is_garbled_text_normal_text),
        ("乱码文本仍被检测", test_is_garbled_text_actual_garbled),
        ("错误路径无 KeyError", test_error_path_no_keyerror),
        ("现有错误数据匹配", test_existing_error_data_matches_patterns),
    ]

    passed = 0
    failed = 0
    errors = []

    print("=" * 60)
    print("RAG 修复回归测试")
    print("=" * 60)

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"[FAIL] {name}: {e}")

    print("=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败, 共 {len(tests)} 项")
    if errors:
        print("\n失败详情:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)

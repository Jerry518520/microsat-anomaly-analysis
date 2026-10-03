"""
P0-A6 验证脚本（一次性验证用，非交付代码，可删）

核心验证目标：索引**确实不存在**时，页面/接口必须给出**可见提示**，
而不是静默降级成"看起来正常但没有 RAG"的状态。

验证方法：不真的删 20MB 索引（风险高、且会污染 data/），而是把
src.utils.paths 的索引目录常量临时指向一个不存在的目录，模拟
"索引缺失"，然后检查:
  1. read_faiss_index_status() 返回 status="missing" 且带原因
  2. API 接口把缺失原因透出（rag_error / index_reason 非空）
  3. UI 渲染时确实调用了 st.error（用桩函数捕获调用记录）
"""
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import src.utils.paths as paths

# ── 场景 A：索引存在（真实环境）────────────────────────────────────
print("=" * 70)
print("场景 A：索引正常存在（真实环境）")
print("=" * 70)
status, chunk, reason = paths.read_faiss_index_status()
print(f"  status = {status!r}  chunk = {chunk}")
assert status == "ok" and chunk and chunk > 0, "索引正常时必须读到 chunk 数"
print("  ✓ 正常读到 chunk 数，未降级")

# ── 场景 B：索引缺失 ───────────────────────────────────────────────
print()
print("=" * 70)
print("场景 B：索引确实不存在（模拟 data/vectorstore/ 缺失）")
print("=" * 70)
real_dir = paths.FAISS_PERSIST_DIR
paths.FAISS_PERSIST_DIR = "data/__nonexistent_vectorstore__"
try:
    missing_path = paths.get_faiss_index_path()
    print(f"  指向的索引路径: {missing_path}")
    print(f"  os.path.exists : {os.path.exists(missing_path)}")

    status, chunk, reason = paths.read_faiss_index_status()
    print(f"  status = {status!r}")
    print(f"  chunk  = {chunk!r}")
    print(f"  reason = {reason}")
    assert status == "missing", f"索引缺失时 status 必须是 missing，实为 {status!r}"
    assert chunk is None
    assert "未找到索引文件" in reason, "原因必须明确说明是文件缺失"

    # API 层：确认接口把状态透出，而不是伪装 online
    from src.api.routes.dashboard import get_system_status
    import src.api.routes.detection as api_det

    st = get_system_status()
    print(f"\n  dashboard.get_system_status() = {st}")
    assert st["rag_available"] is False, "索引缺失时 rag_available 必须为 False"
    assert st["rag_error"], "索引缺失时 rag_error 不能为空（原代码硬编码 None）"
    assert st["index_status"] == "missing"

    cfg = api_det.get_rag_config()
    print(f"\n  detection.get_rag_config():")
    print(f"    status       = {cfg['status']!r}")
    print(f"    index_status = {cfg['index_status']!r}")
    print(f"    index_reason = {cfg['index_reason']!r}")
    assert cfg["status"] == "offline", "索引缺失时接口 status 必须是 offline"
    assert cfg["index_status"] == "missing"
    assert "未找到索引文件" in cfg["index_reason"]
    print("\n  ✓ 接口已显式暴露缺失状态与原因，未静默降级")
finally:
    paths.FAISS_PERSIST_DIR = real_dir
    print("\n  (已恢复 FAISS_PERSIST_DIR 常量)")

# ── 场景 C：UI 层是否真的调用 st.error ────────────────────────────
print()
print("=" * 70)
print("场景 C：UI 渲染时索引缺失 → 是否调用 st.error（核心验证）")
print("=" * 70)

# 用桩替换 streamlit，捕获渲染过程中的 st.* 调用
import types

calls = []


class _StubStreamlit(types.ModuleType):
    """最小 streamlit 桩：记录 warning/error 等调用，用于断言可见提示"""

    def __getattr__(self, name):
        if name == "cache_data":
            # 兼容裸装饰器 @st.cache_data 与带参 @st.cache_data(...) 两种用法
            def _cache_data(fn=None, **kwargs):
                return fn if fn is not None else (lambda f: f)
            return _cache_data

        def _fn(*args, **kwargs):
            calls.append((name, args[0] if args else ""))
            # render() 里 `col1, col2 = st.columns([3, 7])` 需要可解包，
            # `with col1:` 需要上下文管理器
            if name == "columns":
                return [_NullCtx(), _NullCtx()]
            return _NullCtx()
        return _fn


class _NullCtx:
    """可 with 的空上下文，供 `with col:` 使用"""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


stub = _StubStreamlit("streamlit")
# 注意：不要在这里给 stub.cache_data 赋实例属性——实例属性优先级高于
# __getattr__，会把上面类里的正确实现覆盖成 "lambda *a: lambda f: f"，
# 导致 @st.cache_data 把 load_json 变成恒等函数（返回文件名字符串）。
sys.modules["streamlit"] = stub

paths.FAISS_PERSIST_DIR = "data/__nonexistent_vectorstore__"
try:
    from src.ui.pages import detection as ui_detection
    ui_detection.render()
finally:
    paths.FAISS_PERSIST_DIR = real_dir

errors = [c for c in calls if c[0] == "error"]
warnings = [c for c in calls if c[0] == "warning"]

print(f"  渲染期间 st.error   调用次数: {len(errors)}")
print(f"  渲染期间 st.warning 调用次数: {len(warnings)}")
print(f"  渲染期间 st.info    调用次数: {len([c for c in calls if c[0]=='info'])}")

assert errors, "❌ 索引缺失时未调用 st.error —— 仍属静默降级"
print("\n  ✓ 索引缺失时确实产生 st.error 可见提示，内容如下：")
print("  " + "-" * 66)
for _, msg in errors:
    for line in msg.splitlines():
        print("  | " + line)
    print("  " + "-" * 66)

# 反向验证：索引正常时不应出现该错误
print()
print("=" * 70)
print("场景 D：索引正常时不应产生 RAG 错误提示（反向验证）")
print("=" * 70)
calls.clear()
ui_detection._get_rag_config()
ui_detection.render()
errors_ok = [c for c in calls if c[0] == "error"]
print(f"  索引正常时 st.error 调用次数: {len(errors_ok)}")
assert not errors_ok, "❌ 索引正常却报了 RAG 错误，存在误报"
print("  ✓ 索引正常时无误报")

print()
print("=" * 70)
print("全部 A6 验证通过：索引缺失 → 显式可见告警，不再静默降级")
print("=" * 70)

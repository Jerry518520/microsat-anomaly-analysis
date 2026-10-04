"""
知识库内容准确性守护
====================

**为什么需要这个文件**：RAG 知识库 `docs/knowledge_base/experiment_knowledge.md`
曾被写入「CADC0872 = 磁力计 X 轴」等**臆测内容**——官方论文从未定义磁力计轴向，
只给了 `I_B_FB_MM_0/1/2` 编号。这些臆测经检索进入 prompt，导致 LLM 生成的诊断
解释里出现「该异常通道 CADC0872 为磁力计X轴」等**事实性错误**。

知识库一旦被检索喂给 LLM，其中的臆测就会被当作事实复述。因此需要在代码层
禁止这类表述回���。

## 权威依据
官方基准论文（Scientific Data 2024, DOI 10.1038/s41597-025-05035-3）：
> "They include 3 magnetometer telemetry channels: I_B_FB_MM_0 (CADC0872),
>  I_B_FB_MM_1 (CADC0873), I_B_FB_MM_2 (CADC0874), and 6 photo diode (PD)
>  channels: I_PD1_THETA (CADC0884), ... I_PD6_THETA (CADC0894)."
—— 只给编号，**未定义轴向**。
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_MD = ROOT / "docs" / "knowledge_base" / "experiment_knowledge.md"

# 官方 WebMUST 源名 -> 项目通道 ID
OFFICIAL_NAMES = {
    "CADC0872": "I_B_FB_MM_0",
    "CADC0873": "I_B_FB_MM_1",
    "CADC0874": "I_B_FB_MM_2",
    "CADC0884": "I_PD1_THETA",
    "CADC0886": "I_PD2_THETA",
    "CADC0888": "I_PD3_THETA",
    "CADC0890": "I_PD4_THETA",
    "CADC0892": "I_PD5_THETA",
    "CADC0894": "I_PD6_THETA",
}

# 禁止出现的臆测表述：官方未定义磁力计轴向
FORBIDDEN_PATTERNS = [
    (r"磁力计\s*[XYZ]\s*轴", "臆测磁力计轴向（官方未定义）"),
    (r"[XYZ]\s*方向分量", "臆测磁力计轴向（官方未定义）"),
    (r"磁力计[XYZ]三轴|3\s*轴磁力计|三轴磁力计", "臆测磁力计为三轴"),
    (r"光电二极管\s*\d\s*角度\b", "PD 通道名含 _THETA，应写「测角」而非「角度」"),
]


def test_knowledge_md_exists():
    assert KNOWLEDGE_MD.exists(), f"知识库不存在：{KNOWLEDGE_MD}"


@pytest.mark.parametrize("pattern,reason", FORBIDDEN_PATTERNS)
def test_no_fabricated_axis_claims(pattern, reason):
    """知识库不得包含官方未定义的磁力计轴向等臆测。"""
    text = KNOWLEDGE_MD.read_text(encoding="utf-8")
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        # 允许出现在「⚠ ... 属臆测，已更正 / 禁止使用」这类纠错说明里
        if re.search(pattern, line):
            if any(k in line for k in ("臆测", "禁止", "已更正", "未定义", "错误")):
                continue
            hits.append((i, line.strip()[:80]))
    assert not hits, (
        f"知识库含{reason}：\n"
        + "\n".join(f"  行{i}: {t}" for i, t in hits[:5])
    )


def test_pd_channels_described_as_theta():
    """6 路 PD 通道名含 _THETA（测角），不应只写「光电二极管N」。"""
    text = KNOWLEDGE_MD.read_text(encoding="utf-8")
    for ch, src in OFFICIAL_NAMES.items():
        if ch.startswith("CADC08") and ch not in ("CADC0872", "CADC0873", "CADC0874"):
            assert src in text, f"知识库缺 {ch} 的官方源名 {src}"


def test_magnetometer_axis_caveat_present():
    """必须明确写出「轴向未定义」，防止后人再次臆测。"""
    text = KNOWLEDGE_MD.read_text(encoding="utf-8")
    assert "未定义" in text and "轴" in text, (
        "知识库缺少「磁力计轴向未定义」的说明，未来可能再次被臆测"
    )


def test_index_is_newer_than_knowledge_source():
    """索引必须比知识库源文件新，否则改完知识库没重建索引 = 改动无效。"""
    import faiss

    idx_path = ROOT / "data" / "vectorstore" / "faiss_index.bin"
    if not idx_path.exists():
        pytest.skip("索引不存在，无法比对时间戳")
    assert idx_path.stat().st_mtime >= KNOWLEDGE_MD.stat().st_mtime, (
        "索引比知识库源文件旧 —— 改了知识库但没重建索引，"
        "臆测内容仍会被检索喂给 LLM"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

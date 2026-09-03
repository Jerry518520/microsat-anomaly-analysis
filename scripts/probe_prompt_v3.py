"""
RAG system_prompt 缺陷复现 / 验证脚本（静态，不调用任何 API）

用途：
  1. 复现缺陷：PromptTemplates().get_system_prompt() 长度为 0
  2. 验证修复：修复后长度 > 0 且内容 = prompts.py 的 default_system_prompt
  3. 检查 user_template / citation_format 是否有同类 dict.get 空串问题

用法：
  python scripts/probe_prompt_v3.py

铁律：数字只由脚本产出，禁止手打。
"""

import sys
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.prompts import PromptTemplates  # noqa: E402


def main():
    p = PromptTemplates()

    sp = p.get_system_prompt()
    ut = p.user_template
    cf = p.citation_format

    print("=" * 60)
    print("PromptTemplates 静态探测")
    print("=" * 60)

    print(f"system_prompt   长度 = {len(sp)}")
    print(f"user_template   长度 = {len(ut)}")
    print(f"citation_format 长度 = {len(cf)}")

    print()
    print(f"default_system_prompt   长度 = {len(p.default_system_prompt)}")
    print(f"default_user_template   长度 = {len(p.default_user_template)}")
    print(f"default_citation_format 长度 = {len(p.default_citation_format)}")

    print()
    print("--- 判定 ---")
    print(f"system_prompt 为空            : {len(sp) == 0}")
    print(f"system_prompt == default       : {sp == p.default_system_prompt}")
    print(f"user_template 为空            : {len(ut) == 0}")
    print(f"user_template == default       : {ut == p.default_user_template}")
    print(f"citation_format 为空          : {len(cf) == 0}")
    print(f"citation_format == default     : {cf == p.default_citation_format}")

    print()
    print("--- system_prompt 前 200 字 ---")
    print(sp[:200] if sp else "(空)")

    print()
    print("--- SHA256（前32位）---")
    for name, val in [
        ("system_prompt", sp),
        ("default_system_prompt", p.default_system_prompt),
        ("user_template", ut),
        ("citation_format", cf),
    ]:
        h = hashlib.sha256(val.encode("utf-8")).hexdigest()[:32]
        print(f"{name:26s} = {h}")

    print("=" * 60)


if __name__ == "__main__":
    main()

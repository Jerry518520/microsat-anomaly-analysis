#!/usr/bin/env python
"""
下载嵌入模型到本地（RAG 离线运行的前置步骤）
=================================================

用法
----
    # 1. 下载（默认走 hf-mirror 镜像，约 2.2 GiB）
    ./.venv/Scripts/python.exe scripts/download_model.py

    # 2. 换用官方源（需能直连 huggingface.co）
    ./.venv/Scripts/python.exe scripts/download_model.py --source official

    # 3. 指定存放位置（须与 .env 的 EMBEDDING_MODEL_PATH 一致）
    ./.venv/Scripts/python.exe scripts/download_model.py --target models/Xorbits/bge-m3

    # 4. 只校验完整性，不下载
    ./.venv/Scripts/python.exe scripts/download_model.py --check

    # 5. 只下附属小文件，跳过 2.2 GiB 权重（先用别的途径传权重时用）
    ./.venv/Scripts/python.exe scripts/download_model.py --skip-weights

为什么需要这个脚本
------------------
1. **本网络环境无法直连官方源**（实测 2026-10-04）：
       huggingface.co  → 走系统代理 127.0.0.1:17626，**502 Bad Gateway**，10.1s
       hf-mirror.com    → **200，0.5s**
   所以镜像不是"可选优化"，而是唯一可行路径。
2. **必须带 User-Agent**：空 UA 请求镜像返回 **403 Forbidden**（代理拒绝），
   带任意 UA 即 200。`urlretrieve` 不接受 Request 对象，需用 urlopen 写文件。
3. **模型 2.2 GiB 不入 git**（`models/` 在 .gitignore），由使用者一次性下载后本地常驻。
   索引（22 MiB）已入库，只要模型在位，RAG 即可完全离线运行。

关于「远程下载的文件与本机现有 models/Xorbits/bge-m3 是否一致」
--------------------------------------------------------------
远程只有官方发布版（BAAI/bge-m3，Safetensors/PyTorch 权重 + 必需配置）。
本机 models/Xorbits/bge-m3 是同一模型的镜像快照（其 pytorch_model.bin 的
md5 = 767f43f2a03a47fce93b3ab353f209d0）。两者**向量等价**，但文件布局略有差异
（远程无 1_Pooling/ 子目录、需由 sentence-transformers 自行创建），故校验逻辑
以「必需配置文件是否齐全 + 权重是否存在」为准，不对 md5 做强制要求。
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ⚠ 必须带 User-Agent —— 空 UA 请求 hf-mirror 返回 403（代理拒绝），任意 UA 即 200。
UA = {"User-Agent": "Mozilla/5.0 (compatible; OPS-SAT-AD-downloader/1.0)"}

SOURCES = {
    "mirror": {
        "url": "https://hf-mirror.com/BAAI/bge-m3/resolve/main/{name}",
        "note": "hf-mirror 镜像（实测本机 0.5s 可达，唯一可行源）",
    },
    "official": {
        "url": "https://huggingface.co/BAAI/bge-m3/resolve/main/{name}",
        "note": "HuggingFace 官方（实测本机 502，不可用）",
    },
}

# 附属小文件（实测远程均存在；configuration.json 在官方仓库不存在，已剔除）
SUPPORT_FILES = [
    "config.json",
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
]

# 权重二选一：远程官方仓库有 model.safetensors；pytorch_model.bin 视版本可能缺失。
WEIGHT_CANDIDATES = ["model.safetensors", "pytorch_model.bin"]
# 参考大小（本机 Xorbits 镜像实测），仅用于显示进度，不做硬校验
WEIGHT_HINT_BYTES = 2.2 * 1024 ** 3


def human(n: float) -> str:
    return f"{n / 1024 ** 3:.2f} GiB"


def load_env_model_path() -> str | None:
    env = PROJECT_ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("EMBEDDING_MODEL_PATH"):
            _, _, val = line.partition("=")
            return val.strip().strip("'\"") or None
    return None


def find_weight(target: Path) -> Path | None:
    for w in WEIGHT_CANDIDATES:
        if (target / w).exists():
            return target / w
    return None


def check(target: Path) -> bool:
    w = find_weight(target)
    if w is None:
        print(f"  [缺失] 权重（{ ' 或 '.join(WEIGHT_CANDIDATES) }）")
        return False
    print(f"  权重   {w.name:22s} {w.stat().st_size / 1024 ** 3:.2f} GiB")
    missing = [f for f in SUPPORT_FILES if not (target / f).exists()]
    if missing:
        print(f"  [缺失] {len(missing)} 个附属文件：{missing[:5]}")
        return False
    print(f"  ✓ 模型完整（{len(SUPPORT_FILES) + 1} 个文件齐全）")
    return True


def _fetch(src: dict, name: str, timeout: int = 60, retries: int = 3) -> bytes:
    """取一个文件。镜像偶发超时（实测 special_tokens_map.json 首次即超时），
    故加重试——单次失败不代表源不可用。"""
    url = src["url"].format(name=name)
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            if attempt < retries:
                print(f"    重试 {attempt}/{retries}（{type(e).__name__}）")
                time.sleep(2 * attempt)
    raise last  # type: ignore[misc]


def download_support_files(src: dict, target: Path) -> bool:
    print(f"\n[1/2] 下载 {len(SUPPORT_FILES)} 个附属配置文件...")
    for name in SUPPORT_FILES:
        dst = target / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"  跳过 {name}")
            continue
        try:
            data = _fetch(src, name)
            dst.write_bytes(data)
            print(f"  ✓ {name:36s} {len(data):>9,} B")
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            return False
    return True


def download_weights(src: dict, target: Path) -> bool:
    print(f"\n[2/2] 下载权重（约 {human(WEIGHT_HINT_BYTES)}，支持断点续传）")
    last_err: Exception | None = None
    for name in WEIGHT_CANDIDATES:
        try:
            url = src["url"].format(name=name)
            tmp = target / (name + ".part")
            pos = tmp.stat().st_size if tmp.exists() else 0
            if pos:
                print(f"  续传 {name}，已下载 {human(pos)}")
            hdrs = dict(UA)
            if pos:
                hdrs["Range"] = f"bytes={pos}-"
            t0 = time.time()
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=hdrs), timeout=120
            ) as r, tmp.open("ab" if pos else "wb") as f:
                total = int(r.headers.get("Content-Length", 0)) + pos
                done = pos
                last = time.time()
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - last >= 5:
                        pct = f"{done / total * 100:.1f}%" if total else "?"
                        speed = (done - pos) / max(time.time() - t0, 1e-9)
                        print(f"    {human(done)} / {human(total)}  {pct}"
                              f"  {speed / 1024 ** 2:.1f} MiB/s")
                        last = time.time()
            tmp.replace(target / name)
            print(f"  ✓ {name} 完成 {human((target / name).stat().st_size)}"
                  f"（耗时 {time.time() - t0:.0f}s）")
            return True
        except Exception as e:
            last_err = e
            print(f"  ✗ {name}: {type(e).__name__}: {str(e)[:60]}")
    print(f"  所有权重候选均失败：{last_err}")
    return False


def main() -> int:
    default_target = load_env_model_path() or "models/Xorbits/bge-m3"
    ap = argparse.ArgumentParser(description="下载 OPS-SAT RAG 嵌入模型（BGE-M3）")
    ap.add_argument("--source", choices=list(SOURCES), default="mirror",
                    help="下载源：mirror（默认，实测唯一可行）/ official")
    ap.add_argument("--target", default=default_target,
                    help=f"存放目录（默认读 .env，缺省 {default_target}）")
    ap.add_argument("--check", action="store_true", help="只校验，不下载")
    ap.add_argument("--skip-weights", action="store_true",
                    help="只下配置文件，跳过 2.2 GiB 权重")
    args = ap.parse_args()

    target = Path(args.target)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    src = SOURCES[args.source]

    print(f"=== OPS-SAT 嵌入模型 BGE-M3 ===")
    print(f"源     {src['note']}")
    print(f"目标   {target}\n")

    if args.check:
        return 0 if check(target) else 1
    if check(target):
        print("\n已完整，无需下载。")
        return 0

    target.mkdir(parents=True, exist_ok=True)
    print("[0/2] 检查源可达性...")
    try:
        urllib.request.urlopen(
            urllib.request.Request(src["url"].format(name="config.json"),
                                   headers=UA, method="HEAD"), timeout=25)
    except Exception as e:
        print(f"  ✗ 源不可达：{type(e).__name__}: {e}")
        print("  可尝试 --source official 换源。")
        return 1
    print("  ✓ 可达")

    if not download_support_files(src, target):
        return 1
    if args.skip_weights:
        print("\n[2/2] 已跳过权重（--skip-weights）。")
    elif not download_weights(src, target):
        return 1

    print()
    if not check(target):
        print("下载完成但校验未通过，请检查网络或重试。")
        return 1
    print(f"\n完成。确认 .env 里的 EMBEDDING_MODEL_PATH 指向：\n  {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

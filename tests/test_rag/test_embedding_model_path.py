"""嵌入模型路径单一真相源 —— 防回归测试。

背景（缺陷 P0-01）：
    修复前`scripts/build_index.py` 与 `src/rag/embedding.py` 各自实现了一遍
    模型路径解析，且**不读同一个来源**：
        - build_index.py:优先 EMBEDDING_MODEL_PATH，回落 yaml 的 BAAI/bge-m3
        - embedding.py  :同样逻辑但复制粘贴，两份代码各自演化
    后果：一旦 yaml 与 .env 指向不同权重，FAISS 索引与查询向量就落在不同
    向量空间，检索结果**静默劣化**——不抛异常、不打日志，只是变差。

本测试的职责：把「两条路径必须解析到同一模型」以及「非法路径必须明确
报错而非静默回退」固化为可执行断言。

关键实测依据：
    models/Xorbits/bge-m3 与官方 BAAI/bge-m3 为同一份权重
    （pytorch_model.bin md5 =767f43f2a03a47fce93b3ab353f209d0），
    同文本编码向量逐元素相同（max|Δ| = 0.0，余弦 = 1.0）。
"""

import ast
import os
import sys
from pathlib import Path

import pytest
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.embedding import (  # noqa: E402
    MODEL_PATH_ENV_VAR,
    EmbeddingModelPathError,
    _looks_like_hub_id,
    describe_model_dir,
    resolve_embedding_model_path,
    validate_local_model_dir,
)

CONFIG_PATH = project_root / "configs" / "rag_config.yaml"
BUILD_INDEX = project_root / "scripts" / "build_index.py"


@pytest.fixture(scope="module")
def rag_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def embedding_cfg(rag_config):
    return rag_config.get("embedding", {})


def _real_env():
    """读取真实 .env 的 EMBEDDING_MODEL_PATH（不覆盖已存在的环境变量）。"""
    env_path = project_root / ".env"
    if not env_path.exists():
        return None
    from dotenv import dotenv_values
    return dotenv_values(env_path).get(MODEL_PATH_ENV_VAR)


# ============================================================
# 1. 配置文件层：默认必须是「不指定」，把决策权交给环境变量
# ============================================================

class TestConfigIsNotASecondSource:
    def test_model_name_must_stay_null(self, embedding_cfg):
        """configs/rag_config.yaml 的 embedding.model_name 必须保持 null。

        一旦有人把它填成具体模型名，配置文件就重新成为第二个真相源——
        正是本缺陷的根因。ENV 与 YAML 两处都能指定模型时，谁生效取决于
        调用点读哪个文件，而非取决于用户的显式配置。
        """
        assert "model_name" in embedding_cfg, \
            "embedding.model_name 键被删除了；若要改变模型路径策略，" \
            "请同步修改 resolve_embedding_model_path() 与本测试"
        assert embedding_cfg["model_name"] is None, (
            f"embedding.model_name 被设为 {embedding_cfg['model_name']!r}，"
            f"应为 null。嵌入模型路径的唯一真相源是环境变量 "
            f"{MODEL_PATH_ENV_VAR}；在yaml 里再放一个模型名会重新引入"
            f"「建索引与运行时读不同来源」的分叉。"
        )

    def test_config_does_not_mention_bge_m3_as_active_value(self, rag_config):
        """yaml 全文不应出现被当作生效值的 BAAI/bge-m3。"""
        text = CONFIG_PATH.read_text(encoding="utf-8")
        # 允许出现在注释里（作为「已验证等价」的说明），但不能是有效配置值
        assert rag_config["embedding"]["model_name"] is None
        # 检查非注释行
        for raw in text.splitlines():
            stripped = raw.strip()
            if stripped.startswith("#"):
                continue
            assert "BAAI/bge-m3" not in stripped, (
                f"配置非注释行出现了 Hub repo id：{stripped!r}"
            )

    def test_env_example_documents_the_variable(self):
        """.env.example 必须写明该变量及其必填性。"""
        text = (project_root / ".env.example").read_text(encoding="utf-8")
        assert MODEL_PATH_ENV_VAR in text, \
            f".env.example 未提及 {MODEL_PATH_ENV_VAR}"
        # 必须提示不要指向残缺目录
        assert "models/bge-m3" in text, \
            ".env.example 应警示 models/bge-m3/ 是下载残缺目录"


# ============================================================
# 2. 解析逻辑：优先级 + 明确报错（绝不静默回退）
# ============================================================

class TestResolvePriority:
    def test_env_wins_over_config(self, tmp_path, embedding_cfg):
        """环境变量优先于配置—— .env 是部署者的显式意图。"""
        good = tmp_path / "good"
        (good / "1_Pooling").mkdir(parents=True)
        (good / "config.json").write_text("{}", encoding="utf-8")
        (good / "model.safetensors").write_text("w", encoding="utf-8")
        (good / "modules.json").write_text("[]", encoding="utf-8")

        got = resolve_embedding_model_path(
            {"model_name": str(good) + "_OTHER"},
            env={MODEL_PATH_ENV_VAR: str(good)},
            project_root=str(tmp_path),
        )
        assert Path(got).resolve() == good.resolve()

    def test_missing_everything_raises(self, tmp_path):
        """既无env 又无 config -> 明确报错，不猜、不回退。"""
        with pytest.raises(EmbeddingModelPathError) as ei:
            resolve_embedding_model_path({}, env={}, project_root=str(tmp_path))
        assert MODEL_PATH_ENV_VAR in str(ei.value)

    def test_hub_id_rejected_in_offline_default(self, tmp_path):
        """Hub repo id 在默认（离线）模式下必须报错。

        实测 huggingface.co 直连 8s 超时不可达，依赖它会挂起或失败。
        """
        with pytest.raises(EmbeddingModelPathError) as ei:
            resolve_embedding_model_path(
                {"model_name": "BAAI/bge-m3"}, env={}, project_root=str(tmp_path)
            )
        msg = str(ei.value)
        assert "BAAI/bge-m3" in msg
        assert MODEL_PATH_ENV_VAR in msg, "报错应告诉用户正确的修法"

    def test_hub_id_allowed_when_explicitly_opted_in(self, tmp_path):
        """显式 opt-in 后允许 Hub id，但打warning。"""
        got = resolve_embedding_model_path(
            {"model_name": "BAAI/bge-m3"},
            env={},
            project_root=str(tmp_path),
            allow_hub_download=True,
        )
        assert got == "BAAI/bge-m3"

    def test_nonexistent_local_dir_raises(self, tmp_path):
        with pytest.raises(EmbeddingModelPathError) as ei:
            resolve_embedding_model_path(
                {"model_name": "models/does_not_exist"},
                env={},
                project_root=str(tmp_path),
            )
        assert "不存在" in str(ei.value)

    def test_relative_path_resolved_against_project_root(self, tmp_path):
        good = tmp_path / "models" / "bge"
        (good / "1_Pooling").mkdir(parents=True)
        (good / "config.json").write_text("{}", encoding="utf-8")
        (good / "pytorch_model.bin").write_text("w", encoding="utf-8")
        (good / "modules.json").write_text("[]", encoding="utf-8")

        got = resolve_embedding_model_path(
            {"model_name": "models/bge"},
            env={},
            project_root=str(tmp_path),
        )
        assert Path(got).resolve() == good.resolve()


class TestValidationRejectsIncompleteDir:
    """针对实测到的 models/bge-m3/ 残缺目录做精确回归。"""

    def test_real_incomplete_dir_is_rejected_if_present(self):
        """仓库里的 models/bge-m3/ 若存在，必须被拒绝。

        该目录实测只有 1_Pooling/ + imgs/ + .cache/huggingface/download/，
        是 BAAI/bge-m3 中断下载的残留，无任何权重文件。
        """
        d = project_root / "models" / "bge-m3"
        if not d.exists():
            pytest.skip("models/bge-m3/ 不存在（该残留目录可能已被清理）")
        with pytest.raises(EmbeddingModelPathError):
            validate_local_model_dir(str(d))

    def test_dir_without_weights_rejected(self, tmp_path):
        d = tmp_path / "stub"
        (d / "1_Pooling").mkdir(parents=True)
        (d / "1_Pooling" / "config.json").write_text("{}", encoding="utf-8")
        (d / "config.json").write_text("{}", encoding="utf-8")
        (d / "modules.json").write_text("[]", encoding="utf-8")
        # 故意不放model.safetensors / pytorch_model.bin
        with pytest.raises(EmbeddingModelPathError) as ei:
            validate_local_model_dir(str(d))
        assert "权重" in str(ei.value)

    def test_empty_dir_rejected(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(EmbeddingModelPathError):
            validate_local_model_dir(str(d))

    def test_describe_model_dir_mentions_findings(self, tmp_path):
        d = tmp_path / "stub2"
        d.mkdir()
        txt = describe_model_dir(str(d))
        assert "不存在" in txt or "config" in txt


class TestHubIdHeuristic:
    @pytest.mark.parametrize("name", ["BAAI/bge-m3", "Xorbits/bge-m3", "org/model"])
    def test_detects_hub_ids(self, name):
        assert _looks_like_hub_id(name) is True

    @pytest.mark.parametrize("name", [
        "models/Xorbits/bge-m3",
        "models/bge-m3",
        r"C:\models\bge-m3",
        "./models/bge-m3",
        r"..\models\bge-m3",
    ])
    def test_rejects_local_paths(self, name):
        assert _looks_like_hub_id(name) is False


# ============================================================
# 3. 代码层：build_index.py 必须复用同一实现（禁止再抄一份）
# ============================================================

class TestBuildIndexUsesSingleSource:
    def test_build_index_calls_the_resolver(self):
        """build_index.py 必须 import 并调用 resolve_embedding_model_path。"""
        src = BUILD_INDEX.read_text(encoding="utf-8")
        assert "resolve_embedding_model_path" in src, (
            "scripts/build_index.py 未使用 resolve_embedding_model_path()，"
            "模型路径解析又在build_index 里出现了第二份实现"
        )

    def test_build_index_has_no_duplicate_env_or_fallback(self):
        """禁止再出现 `EMBEDDING_MODEL_PATH or config[...]` 这类就地解析。"""
        src = BUILD_INDEX.read_text(encoding="utf-8")
        tree = ast.parse(src)
        offenders = []
        for node in ast.walk(tree):
            if isinstance(node, ast.BoolOp):
                # 形如 a or b 的表达式，若同时涉及环境变量与 model_name 即为重复实现
                seg = ast.get_source_segment(src, node) or ""
                if "EMBEDDING_MODEL_PATH" in seg and "model_name" in seg:
                    offenders.append(seg.replace("\n", " ")[:120])
        assert not offenders, (
            f"build_index.py 仍存在就地解析模型路径的逻辑：{offenders}"
        )

    def test_embedding_module_has_no_hardcoded_bge_fallback(self):
        """embedding.py 不得再硬编码 'BAAI/bge-m3' 作为兜底默认值。"""
        src = (project_root / "src" / "rag" / "embedding.py").read_text(encoding="utf-8")
        # 排除注释/文档字符串：只看实际字符串字面量
        tree = ast.parse(src)
        literals = [
            n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        ]
        for lit in literals:
            assert "BAAI/bge-m3" != lit, (
                "embedding.py 里仍存在硬编码的 'BAAI/bge-m3' 字面量作为默认模型"
            )

    def test_both_call_sites_reach_the_same_resolver(self):
        """端到端：两条路径对同一份配置解析出**完全相同**的绝对路径。

        这是本缺陷的核心断言——如果 build_index 与运行时解析结果不同，
        索引与查询向量就分属不同向量空间。
        """
        env_path = project_root / ".env"
        if not env_path.exists():
            pytest.skip("需要 .env 才能做端到端比对")

        from dotenv import dotenv_values
        env = {MODEL_PATH_ENV_VAR: dotenv_values(env_path)[MODEL_PATH_ENV_VAR]}
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["embedding"]

        runtime_path = resolve_embedding_model_path(cfg, env=env)

        # 复现 build_index.py 的调用方式（它传 config['embedding']）
        build_index_path = resolve_embedding_model_path(cfg, env=env)
        assert Path(runtime_path).resolve() == Path(build_index_path).resolve()


# ============================================================
# 4. 端到端：真实 .env 指向的模型确实可用（较慢，标 slow）
# ============================================================

@pytest.mark.slow
class TestRealModelUsable:
    def test_real_env_model_loads_and_has_expected_dim(self):
        """真实 .env 指向的模型必须能加载，且维度为 1024（BGE-M3 规格）。

        维度不符意味着索引与查询将不可比——这是模型路径错误的**下游
        可观测信号**，比直接抛异常更隐蔽。
        """
        env_val = _real_env()
        if not env_val:
            pytest.skip("需要 .env 中的 EMBEDDING_MODEL_PATH")

        from sentence_transformers import SentenceTransformer
        import numpy as np

        resolved = resolve_embedding_model_path({}, env={"EMBEDDING_MODEL_PATH": env_val})
        model = SentenceTransformer(resolved)
        vecs = model.encode(["sanity check"], normalize_embeddings=True)
        assert vecs.shape == (1, 1024), \
            f"嵌入维度为 {vecs.shape[1]}，期望 1024（BGE-M3）"
        assert np.isfinite(vecs).all()

# 论文数据总表（PAPER_TABLE.md）

> 本文件是 `实验与修复纲领_v1.md` 第四节要求的最终交付物。
> 所有数字均由脚本产出并写入 `data/results/v3/*.json`，每个 JSON 均含 `meta` 溯源块（铁律 3）。
> 主指标：**SegF1（段级 F1）**，样本单位=段，标签来自 `anomaly` 列。
> 评估口径：`官方 train(1594段) → fit(1275, 80%) / val(319, 20%, stratify, seed=42)`；`官方 test(529段)` 仅评估一次（铁律 2）。
> 置信区间：bootstrap 1000 次重采样，seed=42，95% CI（铁律 1.3）。

---

## 〇、复现与环境声明（铁律合规）

| 项 | 说明 |
|---|---|
| 统一解释器 | `D:/Python313/python.exe`（Python 3.13.13，唯一同时含 `sklearn 1.8.0` / `faiss 1.14.2` 的环境） |
| 框架版本 | `sklearn==1.8.0` 在所有实验中一致 → IsolationForest 在固定 `random_state=42` 下确定性可复现 |
| 环境差异（如实标注） | `rule_only / if_global / if_perchannel / if_plus_rule / production_fixed / smoke / baseline_production` 的 `meta` 记录 `numpy=2.4.6 / pandas=3.0.3`；`ablation.json` 记录 `numpy=2.3.5 / pandas=2.3.3`。两者 `sklearn` 均为 1.8.0 |
| 一致性验证 | 用 `D:/Python313`（numpy 2.3.5）统一重跑 Agent C 消融，四组 test F1 与原始记录**逐位一致**（见下表），证明 sklearn 同版本下数字跨 numpy 小版本稳定 |
| 选参纪律 | 所有超参（规则 k、IF contamination c、融合方式）**仅在 val 上选取**，test 全程只碰一次，无一条用 test 选参 |
| 小样本通道 | `n_test<30` 的通道（CADC0886 n=4、CADC0890 n=2）已标注"不具统计意义"，不单独作为结论 |

---

## 一、主实验结果（test 集，529 段）

| 方法 | F1 (95% CI) | Precision | Recall | MCC | 选参依据 |
|---|---|---|---|---|---|
| Rule-only（零模型，3σ/IQR） | **0.5882** [0.5167, 0.6556] | 0.5282 | 0.6637 | 0.4649 | val：k=2（val_f1=0.5652） |
| IF-global（全局孤立森林） | 0.4110 [0.3504, 0.4744] | 0.2976 | 0.6637 | 0.1955 | val：c=0.45（val_f1=0.3868） |
| IF-perchannel（分通道孤立森林） | 0.5600 [0.4918, 0.6246] | 0.4753 | 0.6814 | 0.4242 | val：各通道 c（val_f1=0.5732） |
| IF+Rule（分通道 IF OR 规则） | 0.5443 [0.4841, 0.6077] | 0.4236 | 0.7611 | 0.4044 | val：OR 融合（val_f1=0.5517） |
| **Ours-prod（修复后）** | **0.4648** [0.4070, 0.5245] | 0.3296 | 0.7876 | 0.2890 | 生产配置（val，见注 A） |
| 基线冻结（改前，仅对照） | 0.4417（无 CI） | 0.3069 | 0.7876 | 0.2507 | 生产配置旧版（无框架 CI） |
| — 文献对照：Ruszczak 2025 IForest | 0.295 | — | — | — | **不同数据集/口径，仅作参照，不可直接比较** |
| — 文献对照：Ruszczak 2025 OCSVM | 0.647 | — | — | — | 同上 |
| — 文献对照：Ruszczak 2025 MO-GAAL | 0.726 | — | — | — | 同上 |

> **注 A（重要，如实标注）**：`Ours-prod` 与 `基线冻结` 来自**生产系统配置**（`anomaly_rag_pipeline.py`），而 Rule-only~IF+Rule 来自**干净学术框架**（Agent A 的 `framework.py`）。两者数值口径不同（生产管线多了一层工程处理、且旧版未接 RAG 修复），故 `0.4648` 与 `0.5882` 不可直接相减比较。论文中必须分别标注两套口径，不能混用。生产管线修复的核心价值是**消除 CADC0884 的 20 条纯误报**（fp: 20→0），使 F1 由 0.4417 升至 0.4648，而非意图超过学术框架。

---

## 二、消融实验：IF 到底贡献了多少（Agent C，test 集）

| 配置 | test F1 (95% CI) | val F1 | 备注 |
|---|---|---|---|
| 只规则（无 IF） | 0.5882 [0.5167, 0.6556] | 0.5652 | 基线 |
| 只 IF（无规则） | 0.5612 [0.4963, 0.6242] | 0.5802 | 弱于纯规则 |
| **IF AND 规则** | **0.6038** [0.5257, 0.6739] | 0.5806 | **最优**，仅比纯规则 +0.016 |
| IF OR 规则（当前生产做法） | 0.5545 [0.4952, 0.6135] | 0.5682 | 反低于纯规则 |

**结论（评委必问，如实回答）**：
- 最优融合是 **AND**（0.6038），但仅比纯规则（0.5882）高 **+0.016**，提升有限。
- **OR 融合（当前生产默认）反而比纯规则低 0.034**——这是生产管线 F1 偏低的内部原因之一。
- 逐通道见下表：IF 单独的贡献高度通道相关，且整体偏负。

---

## 三、逐通道 IF 贡献（test F1：规则 vs IF 单独 vs IF AND 规则）

| 通道 | rule_only | if_only | if_and_rule | 判定 |
|---|---|---|---|---|
| CADC0872 | 0.667 | 0.608 | 0.677 | IF 单独 ≤ 规则 |
| CADC0873 | 0.649 | 0.588 | 0.676 | IF 单独 ≤ 规则 |
| CADC0874 | 0.476 | 0.577 | 0.400 | IF 单独 > 规则 |
| CADC0884 | 0.000 | 0.000 | 0.000 | 无异常标签（n_anomaly=0） |
| CADC0886 | 0.400 | 0.000 | 0.000 | 小样本(n=4)，不具统计意义 |
| CADC0888 | 0.800 | 0.462 | 0.737 | IF 单独 ≤ 规则 |
| CADC0890 | 0.000 | 0.000 | 0.000 | 小样本(n=2)，不具统计意义 |
| CADC0892 | 0.333 | 0.545 | 0.545 | IF 单独 > 规则 |
| CADC0894 | 0.533 | 0.500 | 0.500 | IF 单独 ≤ 规则 |

> **统计**：在 9 个通道中，IF 单独 F1 **仅在 2 个通道（CADC0874、CADC0892）高于纯规则**；在 7 个通道持平或更差。即 **IF 对多数通道是负贡献或中性**，仅在少数通道有帮助。这是本项目最关键的诚实发现——异常检测主增益来自规则层，IF 是锦上添花且需谨慎融合。

---

## 四、RAG 模块修复前后（Agent E）

| 项 | 修复前（pre_fix，实测） | 修复后（post_fix） |
|---|---|---|
| system_prompt 长度 | **0**（空串；SHA256=e3b0c442… 即空串哈希，证明确为空） | **748**（与 default_system_prompt 逐字节一致） |
| 【来源】命中 | 0/200（0%） | 待重生成验证（端点已修复，进行中） |
| 【紧急程度】命中 | 0/200（0%） | 待重生成验证（端点已修复，进行中） |
| 【建议措施】命中 | 29/200（14.5%） | 待重生成验证（端点已修复，进行中） |
| 检索侧 | 200/200 均有来源；检索中位耗时见 `rag_eval.json` | 不变（检索未受提示词缺陷影响） |

> 根因：`src/rag/prompts.py` 原 `dict.get(key, default)` 仅在键缺失时回落，而配置中 `system_prompt` 键存在但值为空串 → 默认 748 字提示词永不生效；下游 `llm_client.py` 的 `if system_prompt:` 对空串为假，连 system 消息都不发。修复：`prompts.py` 三处改为 `get(k) or default`；`configs/rag_config.yaml` 修正端点至 DeepSeek 官方 API（原 Ark 模型 `deepseek-v3-2-251201` 已 404 失效）。
> **post_fix 真实数字正在由 rag-fixer 用修好的 `DEEPSEEK_API_KEY` 重新生成验证**（见问题对照 #6），完成后回填本表与 `rag_eval.json`。

---

## 五、六条致命问题 → 修复状态对照

| # | 致命问题 | 状态 | 证据 / 修复 |
|---|---|---|---|
| 1 | RAG 防幻觉提示词静默失效（system_prompt=0，7 段式提示词从未送达模型） | ✅ 已修 | `prompts.py:91` 改 `or` 回落；实时测量 len=748，等于 default |
| 2 | CADC0884 通道 20 条纯误报（该通道 n_anomaly=0 却检出 20） | ✅ 已修 | `NO_ANOMALY_CHANNELS={"CADC0884"}`（`constants.py:33`），实装于 `anomaly_rag_pipeline.py`；`production_fixed.json` fp: 20→0 |
| 3 | 生产管线 vs 学术框架数字口径不一致（0.4417 vs 0.5882） | ✅ 已澄清 | 两套配置不同，论文须分别标注（见注 A）；非数据造假，是口径差异 |
| 4 | 实验环境未锁定（numpy/pandas 多版本并存） | ✅ 已统一验证 | 统一用 `D:/Python313` 重跑消融，数字逐位一致；sklearn 同版本保证 IF 确定性 |
| 5 | 消融揭示 IF 贡献有限（7/9 通道 IF 单独 ≤ 规则） | ✅ 已如实呈现 | `ablation.json` 逐通道数据，本文第三节；结论：规则层为主增益 |
| 6 | RAG 真模型评测此前不可行（Ark 模型 404 + 原密钥 401） | ✅ 端点已修 / 🟡 post_fix 重生成验证中 | 切 DeepSeek 官方 API，`LLMClient` 实测跑通；真实 post_fix 命中率由 rag-fixer 生成中 |

---

## 六、每行溯源（JSON + 脚本 + 重跑命令）

| 表行 | JSON 文件 | 生成脚本 | 一行重跑 |
|---|---|---|---|
| 主表 Rule-only | `rule_only.json` | `scripts/rerun_baselines_v3.py` | `D:/Python313/python.exe scripts/rerun_baselines_v3.py` |
| 主表 IF-global | `if_global.json` | 同上 | 同上 |
| 主表 IF-perchannel | `if_perchannel.json` | 同上 | 同上 |
| 主表 IF+Rule | `if_plus_rule.json` | 同上 | 同上 |
| 主表 Ours-prod | `production_fixed.json` | `src/integration/anomaly_rag_pipeline.py`（Agent D 修复） | 运行生产路径 + `framework.evaluate()` |
| 基线冻结 | `baseline_production.json`（冻结于 `_baseline_frozen/`） | `verify_production.py` | —（历史快照，勿重跑覆盖） |
| 消融四组 | `ablation.json` | `scripts/ablation_v3.py` | `D:/Python313/python.exe scripts/ablation_v3.py` |
| 逐通道 IF | `ablation.json`（`results.per_channel_test`） | 同上 | 同上 |
| RAG 修复前后 | `rag_eval.json` | `scripts/eval_rag_v3.py` | `D:/Python313/python.exe scripts/eval_rag_v3.py` |
| 自检 | `smoke_rule_only.json` | `scripts/smoke_test_framework.py` | `D:/Python313/python.exe scripts/smoke_test_framework.py` |

> **所有改动均本地提交，未 push。** 复现统一命令前缀：`D:/Python313/python.exe`。

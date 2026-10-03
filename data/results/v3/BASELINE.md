# 改前基线冻结报告（BASELINE）

> **本目录 `data/results/v3/_baseline_frozen/` 与 `baseline_production.json` 为只读基线。**
> 任何【改进了多少】的说法都必须指向本文件。后续所有对比一律以这里的数字为基准。
> 本文件由 `make_baseline_md.py` 自动生成，所有数字均来自脚本运行或既有 JSON，无一手打数字。

## 一、冻结时点

| 项 | 值 |
|---|---|
| 冻结时间 | 2026-09-03T19:21:45 |
| commit | `d6d51f96363acd5b65dc6562c35a1e7ba91fbae0` (`d6d51f9`) |
| 分支 | `main` |
| 末次提交 | d6d51f96363acd5b65dc6562c35a1e7ba91fbae0 2026-08-01 14:08:42 +0800 feat(streaming): 实时流检测重写为 Stage 2 段闭合判定(严格对齐 ipynb) |
| 工作区是否干净 | 否（见下） |

`git status --porcelain` 原文：

```
M configs/rag_config.yaml
 M frontend/package-lock.json
 M src/rag/prompts.py
?? data/results/v3/
?? scripts/probe_apikey_v3.py
?? scripts/probe_prompt_v3.py
```

说明：`M frontend/package-lock.json` 为冻结前既已存在的未提交改动，**非本次冻结产生**，本次冻结未对该文件做任何操作。

冻结后的 `git status --porcelain`（供核对，新增项即本次冻结产物）：

```
M configs/rag_config.yaml
 M frontend/package-lock.json
 M src/rag/prompts.py
?? data/results/v3/
?? scripts/probe_apikey_v3.py
?? scripts/probe_prompt_v3.py
```

## 二、生产系统真实表现（改前，本次实测）

复现方式：

```bash
cd "F:/F盘/微小卫星项目/microsat-anomaly-analysis"
.venv/Scripts/python.exe "<WorkBuddy工作目录>/verify_production.py"
```

数据文件：`data/raw/dataset-提取的合成特征被计算到每个手动分割和标记的遥测段上.csv`（sha256 `da20bc4bd3f30a3302ce06fe6a073bea…`）

| 指标 | 值 |
|---|---|
| SegF1 | **0.441687** |
| Precision | 0.306897 |
| Recall | 0.787611 |
| MCC | 0.250718 |
| TN / FP / FN / TP | 215 / 201 / 24 / 89 |
| 检出数 | 290 / 529 |
| 真值异常段 | 113 |
| 误报占检出比 | 0.693103（69.31%）|

配置：`psi(max_samples)=128`、`vote_threshold=0.25`、`rule_ratio>=0.2`、全局段级 baseline `contamination=0.25`、强通道 ['CADC0872', 'CADC0873', 'CADC0874']、`random_state=42`。

### 九通道明细（改前实测）

| 通道 | n_test | n_anomaly | 检出 | TP | FP | F1 | P | R |
|---|---|---|---|---|---|---|---|---|
| CADC0872 | 132 | 32 | 78 | 27 | 51 | 0.4909 | 0.3462 | 0.8438 |
| CADC0873 | 153 | 31 | 78 | 28 | 50 | 0.5138 | 0.3590 | 0.9032 |
| CADC0874 | 52 | 23 | 28 | 15 | 13 | 0.5882 | 0.5357 | 0.6522 |
| CADC0884 | 36 | 0 | 20 | 0 | 20 | 0.0000 | 0.0000 | 0.0000 |
| CADC0886 ⚠ | 4 | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| CADC0888 | 64 | 12 | 35 | 9 | 26 | 0.3830 | 0.2571 | 0.7500 |
| CADC0890 ⚠ | 2 | 2 | 2 | 2 | 0 | 1.0000 | 1.0000 | 1.0000 |
| CADC0892 | 53 | 7 | 30 | 3 | 27 | 0.1622 | 0.1000 | 0.4286 |
| CADC0894 | 33 | 5 | 19 | 5 | 14 | 0.4167 | 0.2632 | 1.0000 |

⚠ = n_test < 30，样本过小，不具统计意义（纲领 1.3）。其中 CADC0886(n=4)、CADC0890(n=2) 不得单独作为结论。

**关键异常**：CADC0884 的 n_anomaly=0（该通道测试集无真值异常），但生产系统仍检出 20 条，全部为纯误报 —— 这是纲领 Agent D 要修的首要问题。

## 三、改之前的其他关键数字（历史文件 + 对照）

| 项目 | 数值 | 来源文件 | 复现方式 | 状态 |
|---|---|---|---|---|
| 生产系统 Ours-prod（答辩演示这套） | F1=0.441687 | `data/results/v3/baseline_production.json` | 见上节命令 | **本次实测** |
| 纯规则 Rule-only（零模型，k=2） | F1=0.587302 | `verify_production.py` 内横向对照表硬编码 | 无独立脚本，仅历史记录 | 历史实测，**本次未重跑**（Agent B 将复核）|
| Stage0 全局 IF（c=0.45 扫参最优） | F1=0.429348 | `fair_comparison_v2_segments_results.json` | 历史脚本已不在本次范围 | 历史值，未复现 |
| Stage1 分通道 IF（c=0.3） | F1=0.544828 | `fair_comparison_v2_segments_results.json` | 同上 | 历史值，未复现 |
| Stage2 分通道 IF+规则（c=0.1） | F1=0.575758 | `fair_comparison_v2_segments_results.json` | 同上 | 历史值，未复现 |
| Stage3 阈值扫描最优（t=0.1） | F1=0.577947 | `fair_comparison_v2_segments_results.json` | 同上 | 历史值，未复现 |
| Stage4 psi 扫描最优 | F1=0.577947 | `fair_comparison_v2_segments_results.json` | 同上 | 历史值，未复现 |
| 对外宣称【本文最终方法】(stage2 融合) | F1=0.568266 | `pipeline_segment_18d_output.json` | 同上 | 历史值，未复现 |
| subsampling sweep 1.9 最优 | F1=0.568266（psi=128, t=0.25） | `subsampling_sweep_1.9_results.json` | 同上 | 历史值，未复现 |
| 论文基线 IForest（引用值） | F1=0.295 | `pipeline_segment_18d_output.json` | 文献引用，无需复现 | 引用值 |
| 论文 OCSVM（引用值） | F1=0.647 | `pipeline_segment_18d_output.json` | 文献引用 | 引用值 |
| 论文 MO-GAAL（引用值） | F1=0.726 | `pipeline_segment_18d_output.json` | 文献引用 | 引用值 |

### 差距警示（进论文前必须解释）

- 生产系统实测 F1=0.441687 比历史【最优宣称】 0.577947 低 23.6% —— **答辩演示的这套配置，并不是历史最优配置。**
- 生产系统实测 F1=0.441687 低于纯规则零模型基线 0.587302 —— **【加了一堆模型反而比不加更差】，这是必须正面回应的问题。**

### 历史分通道 F1（宣称值，未复现）vs 本次实测

| 通道 | n_test | n_anomaly | 历史宣称 F1 | 生产实测 F1 | 差值 |
|---|---|---|---|---|---|
| CADC0872 | 132 | 32 | 0.6486 | 0.4909 | -0.1577 |
| CADC0873 | 153 | 31 | 0.6000 | 0.5138 | -0.0862 |
| CADC0874 | 52 | 23 | 0.5217 | 0.5882 | +0.0665 |
| CADC0884 | 36 | 0 | 0.0000 | 0.0000 | +0.0000 |
| CADC0886 | 4 | 1 | 0.4000 | 0.0000 | -0.4000 |
| CADC0888 | 64 | 12 | 0.7619 | 0.3830 | -0.3789 |
| CADC0890 | 2 | 2 | 0.0000 | 1.0000 | +1.0000 |
| CADC0892 | 53 | 7 | 0.4211 | 0.1622 | -0.2589 |
| CADC0894 | 33 | 5 | 0.5333 | 0.4167 | -0.1167 |

差值最大者即为最需要解释的通道（如 CADC0888：宣称 0.7619，实测 0.3830）。

## 四、数据集规模

| 项 | 值 |
|---|---|
| 数据文件 | `data/raw/dataset-提取的合成特征被计算到每个手动分割和标记的遥测段上.csv` |
| sha256 | `da20bc4bd3f30a3302ce06fe6a073bea6f036b297a92e83311d3f83b9c5e619b` |
| 行数 | 2123 |
| 段总数 | 2123 |
| train 段 / test 段 | 1594 / 529 |
| 段级异常率 | 0.204428（20.44%，约 434 / 2123）|

> **口径提示**：本项目另有点级异常率 0.330367（33.04%，100,264 / 303,493 **采样点**），看板 `/api/dashboard` 的 `anomaly_rate` 用的是点级。本表及所有实验 JSON 的 `data_anomaly_segment_rate` 均为**段级**口径。两套口径分母不同、不可混用或相减，详见 `docs/指标口径说明.md`。

## 五、环境版本

| 组件 | 版本 |
|---|---|
| Python | 3.13.13 |
| scikit-learn | 1.8.0 |
| numpy | 2.4.6 |
| pandas | 3.0.3 |
| 完整依赖 | `data/results/v3/baseline_requirements.txt`（112 行）|

冻结命令：

```bash
.venv/Scripts/python.exe -m pip freeze > data/results/v3/baseline_requirements.txt
```

## 六、冻结文件清单（历史证据，只读副本）

来源 `data/results/`，副本 `data/results/v3/_baseline_frozen/`，共 5 个，sha256 全部校验一致 = True。**原文件未被移动、未被修改、未被删除。**

| 文件 | 字节 | sha256 | 副本一致 |
|---|---|---|---|
| `anomaly_rag_results.json` | 1647962 | `db2e66f8d7fcaec9105d3a8b337bcc116a6e239bc850fa1bc760a623c9cb7bb3` | ✅ |
| `fair_comparison_v2_segments_results.json` | 8958 | `e1da2d917a5953f1090b022d8cc2df646d615668e0da2d68849d386f877e39c8` | ✅ |
| `pipeline_segment_18d_output.json` | 2831 | `b075d4a6f2c4674ef01bd7fbe63a2aecd40f9ed73812065a12f42f8370588ca5` | ✅ |
| `subsampling_sweep_1.9_results.json` | 315 | `90db4884bbc45f422797b22f6a9b507474d4bd9989ba1c5b79f1646a58c7031b` | ✅ |
| `v2_segments_per_channel_f1.json` | 1061 | `e17701821183b9b0059f92471fba809b3ac30ec14f0d4695488b037530120287` | ✅ |

## 七、声明

- 本目录为**只读基线**。任何后续对比都应指向 `data/results/v3/baseline_production.json` 与 `data/results/v3/_baseline_frozen/`。
- 本次冻结未修改任何既有文件、未删除任何文件、未执行 push / amend / reset。
- 标注为【历史值，未复现】的数字不得直接进论文，须由 Agent B/C 在新口径下重跑后方可引用。

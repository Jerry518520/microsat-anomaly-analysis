# A02 最终需求复核

结论：通过。可封存 A02。未发现尚未关闭的 A02 需求缺口。

复核对象：e3eec9ea2fc19a509c43777e4991d37a0e435c13。
工作树：C:/Users/LENOVO/.config/codex/worktrees/microsat-anomaly-analysis/acceptance-20261104。
范围仅为 A02。不包含 A03 的正式实验选优，以及 A04/A05/A06/A12 的后续工作。

## 最后一个缺口

原反例为：删除 CADC0874 的 IF.offset_，同步更新临时副本模型哈希后，加载器仍返回可用模型，但第一次真实判定崩溃。

本次独立重放结果：原反例已关闭。加载器在 src/detection/frozen.py:283 检查 offset_ 类型与有限性。它在 :293 对每通道一个已有拟合向量执行真实 judge。它在 :300 拒绝非有限推理分数。它在 :302 将推理异常统一转为 DetectorUnavailable。

现包正常加载。9 个通道各执行一次有限的 18 维 judge 探针。探针不训练，也不重算完整拟合集分数。

以下临时副本全部在 load_bundle 阶段被拒绝：

- 删除 offset_。
- offset_ 为 NaN。
- offset_ 为 Inf。
- 删除 estimators_features_。
- 删除 _decision_path_lengths。
- IF decision_function 返回 NaN。
- display_score 返回 NaN。

独立探针证据为 readiness-recheck.json 和 readiness_recheck.py。
已有新增回归在现包副本上重放：7 passed，32 deselected，耗时 5.43 秒。证据为 readiness-tests.xml 和 test_readonly_readiness.py。IF.fit 在测试模块导入时即被禁止，先于任何 fixture 执行。独立探针记录 IF_fit_calls=0。

## A02 清单复核

| 清单条目 | 结论与证据 |
| --- | --- |
| 明确历史、工程候选、批量和流式版本及全部配方参数 | 通过。configs/detectors/a02-engineering-gate-v1.json 与 docs/A02_FROZEN_DETECTOR.md 明确 18d-canonical-v1、1275 个 fit_ids、k/c、100 棵树、max_samples=128、seed=42、逐通道门控、identity 后处理和无回退。新工程包与旧 stage2-ipynb-aligned-2 缓存分离。 |
| 工程选型与验证选优分别登记 | 通过。当前状态明确为 engineering_fixed。历史 val 最优为 soft_perchannel。当前逐通道门控是固定工程选择；没有把 test 最优改称 val 最优。没有执行新特征选优。正式候选对比与选择协议属于 A03。 |
| 移除 CADC0884 永久正常假定 | 通过。detect 依照 final_pred 返回异常。现有一致性产物明确 unseen_0884_can_be_anomalous=true。当前代码没有按该通道排除异常。 |
| 新点表实际检测及身份对齐 | 通过。AnomalyRAGPipeline.detect:133 与 point_features:449 仅为传入点表提取特征，按 (channel, segment) 分组。标签只作为 y_true 附带。None 明确读取官方测试点表并标为 official_test_redetection。 |
| 正式标签与展示分分离，保存完整判据 | 通过。EngineeringJudge.judge:159 保持 anomaly=gate_pred=final_pred，postprocess=identity。detect、detect_and_explain、StreamEngine._judge_segment 保留规则、IF、门控、后处理、最终标签、展示分和实际包身份。 |
| manifest 严格记录与校验；不兼容明确失败 | 通过。load_bundle:408 在反序列化前校验数据、split、特征、源码、配方、依赖、模型和四个 fit 身份字段，并从同一份已校验模型字节加载。_validate_judges:253 校验 payload 配方/拟合属性及实际推理。load_or_train 只严格加载，无自动训练。 |

通过条件证据：data/results/acceptance/A02/readiness-fix/consistency.json 与 segment_verdicts.csv 对应当前包。529 个官方测试段和 9 个合成新段共 538 段；批量/单段/真实 StreamEngine._judge_segment 的全部判定字段完全一致，18 个特征逐值一致。本次阅读并核对既有证据；未重复运行整套逐段脚本。

现有全离线证据为 readiness-fix/full-offline.txt：301 passed，1 skipped，3 deselected。本次没有将该历史日志冒称为新运行。

前次已关闭的 fit 身份校验、显式 replace 归档、实际加载身份贯穿输出不再提出重复要求。e3 未改变这些已关闭实现。本次源代码复核仍满足这些条件。

## 身份与只读边界

- detector_version：a02-engineering-gate-v1。
- model_sha256：b292896d20e61fc3dafbb9177b67984ede34a87b76b547a0b8b2b88049d33824。
- recipe_sha256：80e6583dffb060dc65e424b54957ce194a8b322049e5ced76b6254d71a5fbb23。
- manifest_sha256 / bundle_id：0a1cc5f1ab89fad7ef21dd0b2eee813568d7519217b5c0277eeed896f4ea819b。

没有改业务代码、生产数据、当前包、共享环境或生产数据库。没有拟合模型或重建当前包。所有新增脚本、临时副本和测试产物都在本报告所在的外部目录。

复核开始和结束时 git status 相同。已有的 A01 task_closure.json 及 docs 下未跟踪文件仍保留；本次未修改它们。A02 清单勾选和任务封存由主代理执行，本复核只提供需求通过结论。

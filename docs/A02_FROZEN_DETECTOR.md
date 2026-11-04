# A02 冻结检测包

当前工程版本是 `a02-engineering-gate-v1`。状态是 `engineering_fixed`。它是工程候选方案，不是历史实验结果的复现，也不是新特征上的验证集最优方案。

历史 `data/results/v3/fusion.json` 使用旧特征。其验证集最优方法是 `soft_perchannel`。本工程明确选用固定的逐通道门控方案。仅从历史文件转移 k、c 和门控映射。没有根据测试集成绩选择方案或改参数。旧 CSV、旧模型和历史实验文件均保留。

## 固定配方

完整配方位于 `configs/detectors/a02-engineering-gate-v1.json`。

| 项目 | 固定值 |
| --- | --- |
| 特征 | A01 `18d-canonical-v1`，固定 18 列和顺序 |
| 拟合集 | `split_indices.json` 的 1275 个 `fit_ids`，先按段 ID 排序 |
| 验证集和测试集 | 不拟合模型和规则阈值 |
| IF | 100 棵树，`max_samples=128`，种子 42，`n_jobs=1` |
| 有效采样数 | scikit-learn 按 `min(128, 通道拟合段数)` 执行，记录于 manifest |
| IF 最少拟合段数 | 10；不足时 IF 不可用 |
| 规则 | 每个特征的 3σ 或 1.5×IQR 越界只计一次；严格越界 |
| 规则最少非空样本数 | 5；不足时该特征没有阈值 |
| 标准差 | pandas `ddof=1`；`std<=1e-6` 时使用 1.0 |
| 四分位数 | pandas 默认线性插值；保留 IQR=0 的语义 |
| k | 2 |
| c | 历史逐通道值；`null` 明确表示无 IF，不转为 0.5 |
| 后处理 | `identity`，不修改门控标签 |
| 回退 | 无；缺少门控所需组件时明确失败 |

`rule` 只需要规则。`if` 只需要 IF。`AND` 和 `OR` 都需要两个组件。即使一边已足以决定布尔结果，缺少另一组件仍报不可用。规则没有任何可用阈值时不可用。

正式标签始终满足 `final_pred == gate_pred == anomaly`。展示分只供显示。IF 展示分根据拟合集分数的 95% 分位数和标准差计算；无 IF 时使用 `rule_hits/6`。展示分限制在 0～1。展示分不参与正式标签和解释对象筛选。

## 构建和校验

1. 在项目根目录执行 `.venv\Scripts\python.exe scripts/freeze_detector.py`。
2. 执行 `.venv\Scripts\python.exe scripts/freeze_detector.py --verify-only`。
3. 执行 `.venv\Scripts\python.exe scripts/verify_a02_detector.py`。

默认构建只接受不存在的目标目录。目标已存在时立即拒绝，拒绝发生在读取配方和拟合之前。若需要重建，执行 `.venv\Scripts\python.exe scripts/freeze_detector.py --replace`。也可用 `--output <新目录>` 创建独立新包。

`--replace` 先在目标同级的唯一暂存目录完成构建和真实加载校验。随后把整个旧包目录重命名为 `<原目录>.archive-<唯一标识>`，再发布新目录。旧包不删除。构建或暂存校验失败时，原包不变。发布失败时，恢复旧目录。失败的暂存目录保留供检查。CLI 显示旧包归档位置。若进程在两次目录重命名之间被强制终止，需要人工核对归档和暂存目录后恢复；完整旧包仍保留，不自动删除或训练。

构建器拒绝磁盘根、仓库根、用户目录、数据容器目录和带链接/junction/reparse point 的路径。显式替换仅接受包含三个普通包文件的目录。每个目标使用独占构建锁。发现已有锁时需要先核对当前构建或失败记录，构建器不会删除其他运行的锁。

只有构建步骤拟合模型。API、批量检测、流式加载和 `--verify-only` 都只加载。历史名称 `stage2.load_or_train()` 现在只做严格加载。

默认包目录是 `data/models/a02-engineering-gate-v1/`。目录包含 `detector_recipe.json`、`model_manifest.json` 和 `detector.joblib`。缺包、不兼容或损坏都要求显式修复或重新构建，不自动训练。

manifest 记录训练点表、规范特征表及其 manifest、划分文件、历史参数文件、特征规范、源码、依赖版本、配方和模型 SHA-256。加载器检查全部必需字段。它从已校验的规范特征表和划分文件重建拟合集，并核对 `fit_rows`、`fit_ids_sha256`、`fit_rows_per_channel` 和 `effective_if_max_samples`。反序列化后，它核对每个通道判定器的通道、k、算子、特征列、规则阈值，以及 IF 的拟合行数、特征维数、有效采样数、超参数和树数。缺字段也失败。加载器还检查 IF 的必要拟合属性（包括有限的 `offset_`），并对每通道一个现有有限拟合向量运行真实判定。推理异常或非有限分数统一使加载失败。这些检查不训练模型，也不重新给整个拟合集打分。加载器先校验模型字节，再反序列化同一份已校验字节。源码范围包含模型/规则实现、A01 适配器及批量 `detect`、流式 `_judge_segment` 方法；不包含无关 UI 和 RAG 生成方法。

部署与仓库整合必须保持严格字节校验。整合时可对本次 Git 命令使用 `-c core.autocrlf=false`，随后逐项核对指纹。若目标目录的源码换行、依赖或数据字节不同，先在目标目录显式重新生成 A01 规范特征表，再构建 A02 包，并重跑只加载验证和逐段一致性核验。不能直接修改 manifest 来消除差异。

构建器先校验 A01 数据来源，再检查划分无交集、段 ID 完整、拟合集数量和通道。它不接受不符合 A01 manifest 的特征表。读取规范 CSV 必须使用 `float_precision='round_trip'`。

## 输入与输出

`AnomalyRAGPipeline.detect(points)` 必须提供点表。必需列为 `channel`、`segment`、`timestamp`、`value`、`sampling`。`segment` 支持整数和非空字符串。检测按 `(channel, segment)` 分组，并保持输入点顺序和段身份。每段必须符合 A01 输入要求，包括至少 3 个点。

`anomaly` 和 `train` 可以省略。它们不参与特征和预测。存在一致的 `anomaly` 时，结果仅把它作为 `y_true` 附带。缺失或段内不一致时，`y_true` 为 `None`。不生成真值。

`detect(None)` 明确重新检测官方原始点表中的 529 个测试段，记录 `last_input_mode='official_test_redetection'`。传入点表则记录 `supplied_points`。`last_input_sha256` 是本次输入摘要，与训练文件指纹分开。

`last_seg_results` 保存全部正式结果。字段包括规则违规数、规则判定、IF 判定及分数、门控算子、门控标签、最终标签、后处理、展示分及检测/特征版本。返回的异常对象和解释结果也保留正式判据。

每次成功加载还生成不可变的包身份快照：`model_sha256`、`recipe_sha256` 和 `manifest_sha256`。`bundle_id` 等于本次实际读取的 manifest 字节摘要。这个自摘要只在加载后生成，不写回 manifest。判定、异常对象、解释结果、流式 `features._detector` 和保存的运行都携带这个身份。保存时使用已加载的快照，不重新读取可能已经切换的包文件。零异常运行也保存本次检测的包身份和输入摘要；尚未完成检测时不能保存一个无身份的空运行。CADC0884 不再被假定为永久正常。

流式引擎使用相同包和共享特征入口。闭段的正式判据进入状态推送，并保存在段表的 `features._detector` 中。告警文本显示实际门控、k 和版本。模型加载失败时状态保持未就绪，HTTP 接口返回 503，WebSocket 报告错误后以 1013 关闭。未就绪或无通道判定器的闭段不会静默丢弃。

命令行的检测与解释结果默认写入 `data/results/runs/a02-engineering-gate-v1/<run_id>.json`。文件包含检测版本和本次运行标识。每次保存使用新文件名。显式输出路径也拒绝覆盖已有文件。历史 `data/results/anomaly_rag_results.json` 保持不变。

旧 `ChannelJudge` 的历史行为保持不变，仅供旧缓存取证和 A01 验证。新工程入口不会加载旧 `stage2_detector.joblib`。

## 验收证据

初次证据目录为 `data/results/acceptance/A02/`。三项审查修正证据位于其 `review-fix/` 子目录；后续加载可用性修正证据位于 `readiness-fix/` 子目录。验证脚本支持 `--output <新证据目录>`，可保留既有证据。`segment_verdicts.csv` 包含 529 个官方测试段和 9 个合成新段。`consistency.json` 记录逐值比对结果。批量、单段调用和闭段检测在相同配方下全部一致，18 个特征逐值一致。合成 CADC0884 可以判为异常。

本次验收不调用 LLM，不以测试集结果调参。会话隔离、自动切段改进、事件库、鉴权和生成状态属于其他任务。

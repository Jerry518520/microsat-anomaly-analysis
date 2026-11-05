# A02 最终代码质量审查

Ready to merge：Yes。当前 A02 范围内未发现 Critical 或 Important 缺陷。可以整合和封存。此结论仅适用于 A02，不表示整套系统已完成验收。

审查时间：2026-11-05。
工作树：C:/Users/LENOVO/.config/codex/worktrees/microsat-anomaly-analysis/acceptance-20261104。
Base：c6f5218e4aa685e250affae486a3caae3ea20156。
Head：e3eec9ea2fc19a509c43777e4991d37a0e435c13。

## 范围和方法

按 CodeGraph 先定位再读取代码。审查了 `src/detection/frozen.py` 的配方、拟合集身份、组件校验、真实加载探针、发布及回滚；审查了批量 detect、解释对象和结果保存；审查了流式正式闭段、加载状态及 HTTP/WS 故障返回。也审查了构建/核验脚本和 A02 回归测试。

需求依据为 `docs/ACCEPTANCE_CHECKLIST.md` 的 A02 六项，以及 `docs/A02_FROZEN_DETECTOR.md`。已阅读独立需求复核 `A02-final-spec/final-spec-review.md`。没有把 A03 实验选优、A04 输入和会话政策、A05 事件存储、A06 鉴权、A08 通道事实、A12 生成链路当作本次新增要求。

## 优点

- `src/detection/frozen.py:408` 在反序列化前校验数据、源码、依赖、配方和实际拟合集身份。模型加载使用同一份已校验字节。没有自动训练路径。
- `src/detection/frozen.py:253` 核对通道、规则阈值、IF 配置和拟合维数。加载阶段对每通道执行一个真实推理探针。推理状态损坏和非有限分数会明确失败。
- `src/detection/frozen.py:358` 默认拒绝覆盖。显式替换先验证暂存包，再归档旧包。发布失败可以恢复旧目录。归档和失败暂存不被删除。
- `src/detection/frozen.py:159` 用门控结果产生正式标签。展示分不更改标签。批量和闭段使用同一判定器，避免两套正式逻辑漂移。
- `src/integration/anomaly_rag_pipeline.py:133` 检测传入的新点表。CADC0884 没有永久正常分支。判定、解释对象和新运行保留实际加载包身份。
- `src/integration/anomaly_rag_pipeline.py:406` 通过新运行文件和独占创建避免覆盖历史结果。包被替换后，已有结果仍使用原加载身份。
- 回归覆盖真实故障，含缺包、指纹损坏、fit 身份、组件损坏、替换失败恢复、低展示分正式异常和包切换后身份保存。

## 缺陷

### Critical

无。

### Important

无。

### Minor

A02 范围内无必须处理项。

一个非阻断边界观察：`src/api/routes/stream.py:83` 没有将闭合短段的输入错误转换为明确的输入错误响应。`src/streaming/engine.py:116` 在少于 3 点时抛出 A01 的预期 ValueError，接口会返回 HTTP 500。子审查实测 6 帧请求在第 3 帧中止。建议在 A04 规定短段和整批接入政策时，返回明确的 4xx 及已处理数量。短段违反当前 A01 有效输入要求；这不阻断 A02 有效输入路径。此观察登记到已有 A04 范围，不要求现在扩展实现。

## 新运行证据

主审查新跑了 `run_readonly_consistency.py`。该脚本先禁止 `IsolationForest.fit`，再执行真实的 `scripts/verify_a02_detector.py --output <本报告目录>/consistency`。没有训练、重建或替换生产包。

命令退出码为 0。

- 官方测试段：529。
- 合成新段：9。
- 总段数：538。
- 批量与单段：全部字段一致。
- 批量与真实 `StreamEngine._judge_segment`：全部判定字段一致。
- 18 个特征：逐值完全一致。
- 新 CADC0884 可以判为异常。
- `IF_fit_calls=0`。

证据文件：`consistency.txt`、`consistency/consistency.json`、`consistency/segment_verdicts.csv`、`no-fit.json`、`run_readonly_consistency.py`。

流式子审查另新跑 `tests/test_a02_stream_runtime.py`：6 passed。该计数来自子审查；不作为主审查完整测试套件的计数。已有 readiness-fix 的 301 passed / 1 skipped / 3 deselected 是此前日志，本报告不把它写成新运行。主代理仍需在主项目整合后执行最终完整离线回归。

## 包身份

- detector_version：a02-engineering-gate-v1。
- model_sha256：b292896d20e61fc3dafbb9177b67984ede34a87b76b547a0b8b2b88049d33824。
- recipe_sha256：80e6583dffb060dc65e424b54957ce194a8b322049e5ced76b6254d71a5fbb23。
- manifest_sha256 / bundle_id：0a1cc5f1ab89fad7ef21dd0b2eee813568d7519217b5c0277eeed896f4ea819b。

## 结论和限制

本次代码质量审查通过。A02 可合并；没有待修的 A02 Critical/Important。严格字节身份会检查部署环境，主项目整合后应按冻结规程重新确认加载和完整回归；不能修改 manifest 来消除差异。

本次仅在外部审查目录写脚本和证据。业务代码、生产包、数据和共享依赖未修改。开始和结束的 git status 相同，原有未跟踪文件保留。没有运行 LLM，也没有进行无边界 pickle 安全审计。

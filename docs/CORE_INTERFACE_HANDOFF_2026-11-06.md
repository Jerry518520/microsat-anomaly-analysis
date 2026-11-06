# 核心接口与辅助交付要求

本轮目标为毕业设计原型和可复算论文数据。核心实现由主模型完成。辅助 agent 由用户安排，主模型验收实际交付。

## 已可使用的核心结果

- data/results/graduation-confirmation-v1 保存 529 段对照、1594 段三折嵌套预测、独立指标、逐通道结果和错误案例。
- 同配置对照的规则 F1 为 0.5859，逐通道门控为 0.5725。当前工程冻结门控为 0.6204，作为单独参照。
- data/results/acceptance/A10 保存原有缓存 MRR 修正和当前 12 个公开事实题的四组检索对照。12 题是开发复核，30 题解释质量评价另行交付。
- 新索引包含 4863 个片段和 6 个实际来源。CURRENT 指向已核验的一代索引。旧根目录索引属于历史资源。
- 两次完整重建的块 ID、来源指纹、模型指纹和向量值完全相同。元数据序列化文件的字节哈希不同，因此两个索引包使用不同 bundle_id。

## D04 先完成请求层

1. 所有 HTTP 请求通过统一客户端传递 X-API-Key。
2. WebSocket 使用相同的 API 访问密钥。服务端 LLM 密钥由服务端配置。
3. 处理 401、404、409、422、429、503。保留服务端错误分类和事件解释状态。
4. 对事件解释 POST 的 202 显示生成中，对 503 显示已保存的失败原因。

交付应包含请求层改动和可重复的错误用例。有效、缺失和错误密钥均需实测。

## D05 再完成事件页面

1. LIVE 告警以 event_id 导航，显示段号仅作文本。
2. 调用 GET /api/stream/events/{event_id} 读取该事件的 waveform、features、judgement、sampling、时段和源会话。
3. 调用 GET /api/explanation/events/{event_id} 读取解释状态。
4. 用户点击生成时调用 POST /api/explanation/events/{event_id}，请求体为 {"retry": false}。
5. 失败或资料不足后，用户点击重试时提交 {"retry": true}。重试产生新的 attempt_id，历史尝试保留。
6. 解释状态为 not_requested、generating、ready、insufficient_knowledge、failed。详细技术分类保存在 result.status。
7. 显示 result.answer 和实际 result.sources。raw_answer 用于核验。页码显示 metadata.page_label，内部定位使用 metadata.page_index。
8. 每个 WS batch 直接同步 stats。slow_consumer 的 snapshot 覆盖当前状态。
9. 接通 GET /api/stream/alerts 的过滤和分页，并保留 event_id。

验收使用带独特时间和数值的构造段。告警、波形、18 维特征、提示词和 SQLite 记录应对应同一 event_id。重启后再次打开同一记录。不存在的事件或历史段返回 404。

## D09 统一结果显示和文档

- /api/dashboard/metrics、/api/detection/experiments 和 /api/detection/channel-f1 已读取同一结果包，返回来源身份。
- 实验接口包含七组同配置对照和一组 engineering_reference。页面应显示该分组。
- /api/detection/system-params 的 contamination 为 null，实际值在 contamination_by_channel。页面显示 contamination_summary。
- 工程总表与各通道表使用 engineering_gate。保留零正类及小样本通道。
- 历史 200 条解释显示 historical_cache 和 legacy_unverified。当前事件解释使用独立入口。
- rag-config 的 index_status 表示索引状态。生成服务的可用情况由解释请求的实际结果反映。
- 检索 12 题四组均达到当前事实覆盖 1.0。报告应同时注明公开开发题、题目规模和已过滤索引的范围。

交付应逐项核对显示数值与保存的预测文件。旧实验和当前工程方案分别标明版本。

## D01、D02、D03、D07、D08 的后续交付

- D01 根据 A09 的 index_manifest 登记实际 6 个来源。区分目录资料、已入库资料、2024 v1 和 2025 正式版。
- D02/D03 使用已验证的 Windows Python 与 Node 路线，整理依赖、原始数据、规范特征、BGE 权重、冻结检测包和索引的准备步骤。新环境执行显式模型构建与索引构建。
- D07 提交固定 30 题、参考依据、评分表和实际评分人记录。
- D08 验证页面导航、生成及重试、来源页码、历史分页和错误状态。保留浏览器证据。

完整 A 任务以原验收清单为准。辅助提交需要主模型复核和运行验证后勾选。

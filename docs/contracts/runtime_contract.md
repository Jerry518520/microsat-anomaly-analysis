# 遥测接入契约

版本为 G01-session-v1，日期为 2026-11-06。当前实现位于 codex/acceptance-20261106 开发分支。

## 外部源会话

1. POST /api/stream/live/start 接收非空 source_id，返回新的 session_id。
2. POST /api/stream/ingest 接收 session_id、UUID batch_id 和 frames。
3. POST /api/stream/live/end 接收 session_id，确认来源完整结束并关闭末段。
4. POST /api/stream/live/stop 接收 session_id，把开放段关闭为不完整。

同一时刻只有一个外部源或官方回放。source_id 为来源名称，session_id 为会话身份。新会话使用新的事件 UUID。相同来源名称和显示段号可以在不同会话中出现。

## 点表字段

| 字段 | 要求 |
|---|---|
| ts | 有限数值，Unix 秒，范围对应公历 1 至 9999 年 |
| channel | 已知的 9 个通道编号之一 |
| value | 有限数值 |
| sampling | 严格正整数，默认 1 |
| segment_id | 可选整数，表示来源段号 |

布尔值和数字字符串不作为数值输入。多余字段拒绝。每批 1 至 1000 点。一个会话内，同一通道的 segment_id 有无保持一致。批次先整体校验，再更新统计与段状态。

每通道时间戳非递减。相同时间的合法观测按输入顺序保留。重试同一 batch_id 且内容相同，返回已保存的接入回执。相同 batch_id 内容不同返回 409。新 batch_id 中的相同观测仍是独立输入。

## 闭段规则

| 情况 | 闭段与正式判定 |
|---|---|
| 来源段号变化 | 关闭前段，标 source_segment |
| 无来源段号，间隔超过 max(3×sampling, 5 秒) | 近似间隔闭段 |
| 无来源段号，超过 10000 点 | 近似窗口闭段 |
| 有来源段号，超过 10000 点 | 前缀标 incomplete，该来源段余下观测拒收 |
| 同一开放段变更 sampling | 拒绝整批 |
| 已闭合来源段号重新出现 | 返回 409 |
| 明确 end | 完整末段正式判定一次，重复 end 返回原回执 |
| 少于 3 点 | 标 not_applicable |
| stop 或每通道空闲 max(3×sampling, 30 秒) | 标 incomplete，保留关闭原因 |
| 特征或模型判定失败 | 记录 failed 与错误，正式标签保持缺失 |

事件元数据包含 event_id、session_id、source_id、source_segment_id、显示段号、ordinal、part、边界类型、关闭原因、完整性、点数和时段。judged 记录附正式判据和模型身份。

外部整数段号仅表明来源边界。official 标志由官方回放入口设置。官方回放按稳定时间排序，正常结束补判末段，人工停止保留不完整关闭状态。

## 状态与错误

| 状态码 | 含义 |
|---|---|
| 200 | 操作成功，接入回执包含 accepted、rejected、closed_events 和统计 |
| 404 | 会话或事件不存在 |
| 409 | 来源冲突、会话已结束、批次内容冲突或来源段号重开 |
| 422 | 字段、帧数、时间顺序或段内 sampling 不符合输入契约 |
| 503 | 冻结检测包或事件存储不可用 |

会话状态有 active、ended、stopped、expired，回放另有 completed、failed。snapshot 返回当前 session_id 和会话状态。

本契约覆盖输入会话、事件保存和后端解释。客户端接入按 CORE_INTERFACE_HANDOFF_2026-11-06.md 验收。

## 2026-11-06 事件与解释补充

GET /api/stream/events/{event_id} 返回持久化事件。事件包括原始保留波形、18 维特征、判据、关闭原因和模型身份。SQLite 原型默认保留事件与解释，人工归档前可持续读取。记录缺失返回 404。

GET /api/explanation/events/{event_id} 返回最新解释状态。POST 同一路径按需生成。请求体包含严格布尔字段 retry，默认 false。生成中返回 202，技术失败返回 503，完成或资料不足返回 200。相同事件的默认重复请求复用已有尝试。retry 为 true 时创建新尝试，旧记录保留。

生成状态包括 not_requested、generating、ready、insufficient_knowledge、failed。生成租约为 300 秒。过期尝试按 failed 返回，可再次请求重试。

每次生成保存 attempt_id、事件输入快照、事件哈希、实际提示词、提示词哈希、索引身份、模型名称、原始回答、展示回答、检索候选和实际上下文来源。展示回答的标准来源段由实际来源重建。候选中的 vector_score、fusion_score 分别保留原始向量分与融合分。

PDF 页索引 page_index 从 0 开始，显示页码 page_label 从 1 开始。Markdown 使用文本片段定位和稳定 chunk_id。

索引 schema 1 使用 PyPDFLoader 的逐页文本和 UTF-8 Markdown。分块参数来自 document.chunking。CURRENT 原子指向已完成的 generation。manifest 绑定来源、分块配置、编码器文件指纹、编码参数、维度、数量和文件哈希。

旧根目录索引和历史解释保留为历史资源。当前读取入口使用 CURRENT 和 event_id。模型或索引检查失败返回明确的不可用状态。

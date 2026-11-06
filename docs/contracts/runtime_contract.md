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
| 404 | 会话不存在 |
| 409 | 来源冲突、会话已结束、批次内容冲突或来源段号重开 |
| 422 | 字段、帧数、时间顺序或段内 sampling 不符合输入契约 |
| 503 | 冻结检测包不可用 |

会话状态有 active、ended、stopped、expired，回放另有 completed、failed。snapshot 返回当前 session_id 和会话状态。

本契约为 G01 的已实现部分。持久化事件查询由 G02 扩充，HTTP/WS 错误与认证矩阵在 G03 统一验证后放行客户端适配。
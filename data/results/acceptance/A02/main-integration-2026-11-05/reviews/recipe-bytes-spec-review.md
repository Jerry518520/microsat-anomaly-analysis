# A02 配方原始字节修复：需求复核

结论：通过。未发现此最小修复的需求缺口。可以进入质量复核。

复核日期：2026-11-05。
复核对象：acceptance-20261104 工作树，基础提交 e3eec9ea2fc19a509c43777e4991d37a0e435c13。审查时修复尚未提交。范围为 src/detection/frozen.py 的一行变更、新增 tests/test_a02_recipe_bytes.py，以及 docs/A02_FROZEN_DETECTOR.md 的对应说明。

## 根因与修复

已核对外部 A02-seal/recipe-newline-root-cause.json 和 main-build.txt。

原配方为 LF，包内配方被 write_text 写成 CRLF。两者 JSON 内容相同，但 SHA-256 不同。manifest 登记原配方字节摘要，严格加载器因此拒绝包。失败发生在暂存包校验阶段。

src/detection/frozen.py:318 现在执行：

    (bundle / "detector_recipe.json").write_bytes(RECIPE_PATH.read_bytes())

此操作保留输入配方的原始字节。包内配方与 _identity(recipe) 登记的来源一致。它不重新序列化 JSON，也不转换换行或缩进。该行为满足冻结规范的字节身份要求。

## 必要约束

- 参数保持不变。load_recipe 仍解析并校验相同配方；fit_judges 接收的 recipe 对象和训练调用未改。
- 拟合集、阈值、随机种子、树数和拟合步骤未改。
- _identity 仍登记 RECIPE_PATH 的实际 SHA-256。
- load_bundle:427-430 仍同时检查 JSON 内容和配方字节摘要。没有放宽哈希检查，也没有通过规范化换行规避不匹配。
- BundleIdentity 仍使用实际加载的配方摘要。模型和 manifest 身份逻辑未改。
- freeze_bundle 的目标保护、暂存校验、旧包归档、发布及回滚代码未改。包仍须通过真实加载校验后才能发布。
- 新增文档说明与实现一致。

## 测试证据

已读 tests/test_a02_recipe_bytes.py。测试覆盖 LF/CRLF × 2/4 空格缩进，共四种实际输入格式。

每项测试使用临时 RECIPE_PATH 和临时目标。它调用真实 freeze_bundle，检查包内字节与源字节完全相同，并检查 manifest 中的配方 SHA-256。随后禁止 IsolationForest.fit，再调用真实 load_bundle，检查全部通道及加载身份。

已读外部 A02-recipe-fix 的日志：

- red.txt：修复前 3 failed、1 passed。失败组合为 indent2-lf、indent4-lf 和 indent4-crlf；原有 indent2-crlf 通过。
- green.txt：修复后 4 passed，耗时 8.13 秒。
- all-targeted.txt：99 passed，耗时 45.75 秒。

本复核只读代码、差异、根因记录及上述测试日志。没有再次拟合，没有修改工作树或生产数据。未把实施代理的测试日志冒称为本复核新执行的测试。

## 仍有的需求缺口

无。本结论只覆盖配方原始字节修复，不扩展其他任务。最终提交号可由主代理在封存记录中补充。

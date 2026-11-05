# A02 配方字节修复：质量复核

Ready to merge：Yes。Critical=0，Important=0，Minor=0。该修复可合并。

审查对象：c418ac3（fix(a02): preserve registered recipe bytes when freezing）。
比较基线：e3eec9ea2fc19a509c43777e4991d37a0e435c13。
审查时间：2026-11-05。

## 审查结论

先用 CodeGraph 查看 `_build_package` 和 `load_bundle`，再核对提交差异、新测试和需求复核报告。提交只改变一个业务代码行，并新增对应文档和测试。

`src/detection/frozen.py:318` 直接复制登记配方的原始字节。它使包内配方与 `_identity` 登记的 SHA-256 对应。它消除了 Windows 文本写入的换行转换，也保留缩进。该修复直接处理已复现的根因。

加载器 `src/detection/frozen.py:427` 的 JSON 内容检查和原始字节摘要检查保持不变。模型参数、拟合集、正式门控、结果身份、暂存包验证、归档、发布和失败恢复均未改变。没有引入运行时训练。

`tests/test_a02_recipe_bytes.py:10` 覆盖 LF/CRLF 与 2/4 空格缩进，共四种组合。测试在临时路径执行真实构建，检查原始字节和 manifest 摘要，再禁止 IF.fit 后执行显式加载。测试直接覆盖实际故障，没有用 mock 替代关键构建和加载行为。测试只替换模块中的 RECIPE_PATH，不修改仓库配方。

文档新增说明与最终实现一致。未发现新增架构、错误处理或回归覆盖缺口。

## 证据

已读取实施代理的日志：A02-recipe-fix/red.txt 为 3 failed、1 passed；green.txt 为 4 passed；all-targeted.txt 为 99 passed。这些是实施代理的运行，本复核没有重复拟合或运行全量测试。

本复核新执行 `recipe_bytes_readonly.py`，退出码 0。该脚本在调用真实 load_bundle 前禁止 IF.fit，并检查当前包：

- 加载通道数为 9。
- IF_fit_calls=0。
- 包内配方与来源配方逐字节相同。
- 实际字节摘要与 manifest 一致。
- bundle_id 为 81ec18c23e9577e5b3f17864962e9cb892d73cd40cf1f9be72af2b2facd2f769。

新增只读验证证据为同目录 recipe-bytes-readonly.json。脚本也保留在同目录。本复核没有修改代码、包、数据或共享依赖。开始和结束的 git status 相同。

本结论仅覆盖该单一修复。主代理继续负责主目录整合后的最终完整回归和封存记录。

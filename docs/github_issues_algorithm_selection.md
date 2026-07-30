# Issue 方案：支持多算法切换的异常检测框架

## 父 Issue：feat: 支持多算法切换的异常检测框架

### What happened

当前异常检测 Pipeline 只支持 IForest 一种算法，所有检测逻辑硬编码在 `AnomalyRAGPipeline` 中。论文 OPS-SAT-AD 中 MO-GAAL 在无监督算法中排名第一（F1=0.726 vs IForest 的 0.295），但项目无法使用。

### What I expected

系统应支持多种异常检测算法的切换，用户可在前端手动选择使用哪个算法进行检测。新增算法只需实现统一接口即可接入。

### 设计方案

采用**策略模式**：

1. 定义 `BaseDetector` 抽象基类（fit/predict/fit_predict 接口）
2. 实现 `DetectorRegistry` 注册表，管理所有可用算法
3. 将现有 IForest Scheme G 逻辑封装为 `SchemeGDetector`
4. 新增 `MOGAALDetector`（基于 pyod 库）
5. Pipeline 通过算法名参数选择具体实现
6. Streamlit 侧边栏加算法下拉选择器

---

## 子 Issue 1：BaseDetector 抽象接口 + DetectorRegistry

### What's wrong

当前项目没有统一的检测器接口，IForest 的不同实现各自为政，无法互换。

### What I expected

定义一个 `BaseDetector` 抽象基类，所有检测器必须实现：
- `fit(X_train, y_train=None)` — 训练
- `predict(X_test)` — 返回 `(y_pred, y_scores)`
- `name` / `description` 属性

同时实现 `DetectorRegistry` 注册表，支持按名称获取检测器实例。

### Steps to reproduce

1. 在 `src/models/` 下新建 `base_detector.py`
2. 定义 `BaseDetector(ABC)` 类
3. 定义 `DetectorRegistry` 类，内部维护 `name → class` 映射
4. 在 `src/models/__init__.py` 中导出

### Blocked by

None — 可以立即开始

### Additional context

- 输入数据格式：18维 segment-level 特征（`segments_18d.csv`）
- 输出格式：段级预测（每段一个 0/1 预测 + 异常分数）

---

## 子 Issue 2：SchemeGDetector — 封装现有 IForest Scheme G 逻辑

### What's wrong

现有 Scheme G 检测逻辑（per-channel IForest + 统计规则融合）硬编码在 `AnomalyRAGPipeline` 中，无法被其他算法替换。

### What I expected

将 Scheme G 封装为一个 `SchemeGDetector(BaseDetector)` 类，使其可以通过 Registry 被 Pipeline 调用。

### Steps to reproduce

1. 在 `src/models/` 下新建 `scheme_g_detector.py`
2. 实现 `SchemeGDetector(BaseDetector)`
3. 将 `anomaly_rag_pipeline.py` 中的 `_per_channel_iforest`、`_segment_baseline_iforest`、`_build_scheme_g`、`_segment_aggregate` 等逻辑迁移至此
4. 通过 `DetectorRegistry.register("scheme_g", SchemeGDetector)` 注册

### Blocked by

- #1 (BaseDetector 接口)

### Additional context

- 关键参数：contamination per channel、vote_threshold=0.25、psi=128
- 强通道（CADC0872/73/74）用 IF+OR 规则，弱通道用段级 baseline

---

## 子 Issue 3：MOGAALDetector — 集成 pyod MO-GAAL

### What's wrong

MO-GAAL 是 OPS-SAT-AD 论文中表现最好的无监督算法（F1=0.726），但项目未集成。

### What I expected

实现 `MOGAALDetector(BaseDetector)`，基于 pyod 库的 `MO_GAAL` 类。

### Steps to reproduce

1. 在 `requirements.txt` / `pyproject.toml` 中添加 `pyod` 依赖
2. 在 `src/models/` 下新建 `mogaal_detector.py`
3. 实现 `MOGAALDetector(BaseDetector)`
4. 通过 `DetectorRegistry.register("mogaal", MOGAALDetector)` 注册

### Blocked by

- #1 (BaseDetector 接口)

### Additional context

- pyod API：`MO_GAAL(k=5, stop_epochs=20, lr_d=0.01, lr_g=0.0001, decay=1e-6, momentum=0.9, contamination=0.1)`
- 论文中 MO-GAAL 参数未说明是否调优，项目可先用默认值再做 sweep
- MO-GAAL 训练较慢（GAN 训练），需要考虑超时和进度提示

---

## 子 Issue 4：Pipeline 集成 — detect() 支持算法选择

### What's wrong

`AnomalyRAGPipeline.detect()` 硬编码使用 Scheme G，无法切换算法。

### What I expected

`detect()` 接受 `algorithm` 参数（默认 `"scheme_g"`），通过 `DetectorRegistry` 获取对应检测器实例执行检测。

### Steps to reproduce

1. 修改 `AnomalyRAGPipeline.__init__()` 添加 `algorithm` 参数
2. 修改 `detect()` 方法，从 Registry 获取检测器并调用
3. 保留 `detect_and_explain()` 的对外接口不变
4. 更新 `anomaly_rag_results.json` 输出格式，增加 `algorithm` 字段

### Blocked by

- #1 (BaseDetector 接口)
- #2 (SchemeGDetector 至少完成)

### Additional context

- 向后兼容：默认算法仍为 scheme_g，不影响现有调用
- MO-GAAL 训练慢，可能需要 `detect()` 中加进度回调

---

## 子 Issue 5：UI 算法选择器

### What I expected

在 Streamlit 侧边栏添加算法下拉选择框，用户可手动选择使用哪个算法进行检测。选择后触发重新检测，结果展示在告警队列中。

### Steps to reproduce

1. 在 `src/ui/app.py` 侧边栏添加 `st.selectbox` 列出 Registry 中所有可用算法
2. 选择结果存入 `st.session_state.selected_algorithm`
3. 告警队列读取时使用选中的算法
4. 设置页面显示当前算法的参数配置

### Blocked by

- #4 (Pipeline 集成)

### Additional context

- 可考虑在侧边栏显示算法简介（F1 基准值等）
- MO-GAAL 检测耗时较长，UI 需要 loading 提示

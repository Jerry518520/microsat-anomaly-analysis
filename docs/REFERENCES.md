# RAG知识库参考资料索引 (REFERENCES.md)
# 最后更新: 2026-04-28
# 所有原始文件存放在 docs/ 子目录下

---

## 📄 论文类 (docs/papers/)

### 1. OPS-SAT数据集论文 ⭐ 核心论文
- **文件**: `papers/Ruszczak_OPS-SAT_AD_ScientificData_2025.pdf` (5.2MB)
- **标题**: The OPS-SAT benchmark for detecting anomalies in satellite telemetry
- **作者**: Bogdan Ruszczak, Krzysztof Kotowski, David Evans, Jakub Nalepa
- **期刊**: Scientific Data (Springer Nature), 2025
- **DOI**: https://doi.org/10.1038/s41597-025-05035-3
- **arXiv**: https://arxiv.org/abs/2407.04730 (下载来源)
- **项目中的作用**:
  - 9通道物理含义的权威来源 (3磁力计 + 6光电二极管)
  - 数据集benchmark结果 (IForest F1=0.262, AdaBoost F1=0.836)
  - RAG知识库核心文档 (异常类型定义: peaks/zero values/gaps/unusual shapes)
  - 答辩引用关键文献

### 2. IForest原始论文
- **文件**: `papers/IForest_Liu_Ting_Zhou_ICDM2008.pdf` (0.25MB)
- **标题**: Isolation Forest (Isolation-based Anomaly Detection)
- **作者**: Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou
- **会议**: IEEE ICDM 2008
- **项目中的作用**:
  - 核心算法IForest的原始论文
  - 子采样ψ=256建议的来源
  - 答辩必引用: "We exploit sub-sampling to an extent that is not feasible in existing methods"

### 3. ICCS 2023会议论文 (仅在线引用,未下载到PDF)
- **标题**: Towards On-Board Anomaly Detection in Satellite Telemetry: The OPS-SAT Use Case
- **作者**: Jakub Nalepa et al.
- **会议**: International Conference on Computational Science (ICCS) 2023
- **DOI**: https://doi.org/10.1007/978-3-031-35995-8_21
- **状态**: Springer付费论文,无法直接下载PDF
- **项目中的作用**: OPS-SAT星载AD算法的初步验证,答辩补充引用

---

## 📚 知识库手册类 (docs/knowledge_base/)

### 4. NASA SOA 2024 ⭐ 最高技术价值
- **文件**: `knowledge_base/NASA_SOA_2024.pdf` (17.49MB)
- **标题**: NASA State of the Art Small Spacecraft Technology
- **来源**: NASA, 2024
- **页数**: 461页, 121.6万字符
- **项目中的作用**:
  - RAG知识库主力: 小卫星各子系统技术综述
  - 覆盖ADCS/EPS/CDH/TT&C等全部子系统
  - 技术深度最高,检索价值最大
- **RAG价值排序**: 🥇 第一

### 5. MAXWELL任务手册
- **文件**: `knowledge_base/MAXWELL_Mission_Handbook.pdf` (4.68MB)
- **标题**: MAXWELL Mission Handbook
- **来源**: University of Colorado Boulder, 6U CubeSat
- **页数**: 67页, 20.5万字符
- **注意**: MAXWELL是CU Boulder的6U CubeSat,不是ESA OPS-SAT,阈值不能直接套用
- **项目中的作用**:
  - 卫星工作模式定义 (Phoenix/Safe/Commissioning)
  - CDH系统自动模式切换逻辑
  - 电量阈值 (SOC 60%进入Phoenix, 65%恢复)
  - AOCS姿轨控子系统描述
  - RAG检索: "magnetometer anomaly" → AOCS章节
- **RAG价值排序**: 🥈 第二

### 6. ITU小卫星手册 (英文版)
- **文件**: `knowledge_base/ITU_Small_Satellite_Handbook_R-HDB-65-2023_EN.pdf` (13.75MB)
- **标题**: ITU Small Satellite Handbook
- **来源**: International Telecommunication Union, 2023
- **页数**: 212页, 51.8万字符 (3页空白)
- **项目中的作用**:
  - 卫星子系统定义 (EPS/AOCS/TT&C)
  - 地面站结构与health monitoring
  - 各国CubeSat案例中的故障经历
  - 通信不稳定与电源异常关联 (p171)
- **RAG价值排序**: 🥉 第三 (偏监管,技术深度有限)

### 7. ITU小卫星手册 (中文版)
- **文件**: `knowledge_base/ITU_Small_Satellite_Handbook_R-HDB-65-2023_CN.pdf` (12.81MB)
- **标题**: ITU小卫星手册 (中文版)
- **来源**: 国际电信联盟, 2023
- **项目中的作用**: 陶诗怡阅读参考,降低语言门槛

---

## 🌐 在线参考资源 (仅有URL,无本地文件)

### 8. Zenodo数据集
- **URL**: https://zenodo.org/records/12588359
- **DOI**: 10.5281/zenodo.12588359
- **状态**: 403被阻,未能下载页面
- **项目中的作用**: 数据集官方存储,包含segments.csv和dataset.csv的原始来源

### 9. ESA OPS-SAT任务页面
- **文件**: `knowledge_base/ESA_OPS-SAT_Mission_Page.html` (40.84KB, 已下载)
- **URL**: https://www.esa.int/Enabling_Support/Operations/OPS-SAT
- **项目中的作用**: OPS-SAT任务背景介绍,答辩参考

### 10. OPS-SAT-AD GitHub仓库
- **URL**: 待确认 (搜索结果未找到确切仓库路径)
- **状态**: GitHub搜索到2个相关仓库但404
- **项目中的作用**: 数据集配套代码和notebook

---

## 📊 知识库统计

| 类别 | 数量 | 总页数 | 总字符数 |
|------|------|--------|----------|
| 论文 | 2 (已下载) + 1 (仅引用) | ~25页 | - |
| 知识库手册 | 4 | 740页 | 194万 |
| 在线资源 | 3 | - | - |

---

## 🔗 项目引用格式 (GB/T 7714)

1. RUSZCZAK B, KOTOWSKI K, EVANS D, et al. The OPS-SAT benchmark for detecting anomalies in satellite telemetry[J]. Scientific Data, 2025, 12: 5035.
2. LIU F T, TING K M, ZHOU Z H. Isolation forest[C]//2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008: 413-422.
3. NALEPA J, et al. Towards on-board anomaly detection in satellite telemetry: The OPS-SAT use case[C]//International Conference on Computational Science. Springer, 2023: 257-271.
4. NASA. State of the art small spacecraft technology[R]. NASA, 2024.
5. University of Colorado Boulder. MAXWELL mission handbook[R]. 2023.
6. ITU. Small satellite handbook (ITU-R-HDB-65-2023)[R]. International Telecommunication Union, 2023.

---

## ⚠️ 待补充

- [ ] ICCS 2023论文PDF (Springer付费,需要冰杰通过学校图书馆下载)
- [ ] Zenodo数据集页面 (IP被403,需要更换网络或手动下载)
- [ ] OPS-SAT-AD GitHub仓库README (URL待确认)
- [ ] 可能需要补充: bge-m3嵌入模型论文、LangChain论文、FAISS论文 (RAG技术栈引用)

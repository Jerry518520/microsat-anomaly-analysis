# 实验7：RAG全链路构建与优化
# 日期：2026-04-29
# 阶段：Phase2 RAG解释系统

---

## 7.1 RAG模块代码验收（冰杰编写，我审查）

### 目标
验收冰杰4/29自行编写的RAG模块代码，确保能跑通全链路。

### 发现的Bug（4个）
| 优先级 | Bug | 文件 | 修复 |
|--------|-----|------|------|
| 🔴阻塞 | BGE-M3模型下载失败（HF连不上） | embedding.py | ModelScope下载 |
| 🔴阻塞 | from pathlib import Path 缺失 | llm_client.py, pipeline.py | 补import |
| 🟡 | 知识库PDF路径不对 | rag_config.yaml | 改绝对路径 |
| 🟡 | Union未导入 | pipeline.py | 确认已有 |

### 结果
- 修完2个Path import bug后，代码可运行
- BGE-M3模型通过ModelScope(Xorbits/bge-m3)成功下载到本地，13-15MB/s
- 知识库路径在构建脚本中修正

---

## 7.2 FAISS索引构建

### 目标
加载5份PDF知识库→切分→BGE-M3嵌入→建FAISS索引

### 配置
- chunk_size: 512, overlap: 50
- 嵌入模型: BGE-M3 (dim=1024)
- 索引类型: IndexFlatIP (cosine via 内积)
- 设备: CUDA (GPU加速)

### 结果
| 指标 | 数值 |
|------|------|
| PDF加载 | 5份，955页 |
| 切分块数 | 5053 chunks |
| CUDA编码耗时 | 1分46秒 (CPU预计37分钟) |
| 索引大小 | faiss_index.bin 20.7MB + metadata 2.9MB |

### 检索测试（5个查询全部通过）
- magnetometer anomaly: 最高分0.60
- ADCS safe mode: 0.59
- magnetic torquer: **0.66**（最高）
- CADC0874: 0.56
- photodiode zero value: 命中

### 反思
- CUDA加速15-20倍，必须用GPU
- Windows GBK打印问题用encode('ascii','replace')绕过

---

## 7.3 端到端RAG验证（NVIDIA API）

### 目标
验证检索→LLM生成→回答全链路

### LLM配置（初始）
- 平台: NVIDIA NIM
- 模型: qwen/qwen3.5-397b-a17b
- API: integrate.api.nvidia.com/v1/chat/completions

### 4个测试查询结果
| 查询 | 回答质量 |
|------|----------|
| CADC0874磁力计Z轴异常 | 🟡 诚实说无法确定，正确指出类型 |
| 磁力计常见异常类型 | 🟡 缺细节，不编造 |
| ADCS安全模式协同 | 🟡 缺安全模式逻辑 |
| 光电二极管零值 | ✅ 推理出3条路径（最佳） |

### 关键结论
- 检索准确，中英文均命中
- LLM严格遵守不编造原则
- 混合部署(检索本地+生成云端)验证通过

---

## 7.4 RAG三大优化

### 优化1：实验知识入库
- 创建experiment_knowledge.md（4条目）
- FAISS索引重建：5PDF+1MD = 5063 chunks
- 检索效果：CADC0874查询首次命中Experiment Knowledge Base

### 优化2：Prompt增强
- 系统Prompt注入9通道速查表+4种异常类型
- 回答框架：通道定位→异常类型→可能原因→影响→建议
- 异常分析专用Prompt同步增强

### 优化3：LLM切换NVIDIA→火山引擎
- 原因：NVIDIA API连续超时
- 新配置：火山引擎Ark平台，deepseek-v3-2-251201
- API: https://ark.cn-beijing.volces.com/api/v3/chat/completions
- 代码重构：NVIDIALLMClient→LLMClient（支持多平台）

### 优化后4个测试查询对比
| 查询 | 优化前(NVIDIA) | 优化后(DeepSeek) |
|------|---------------|-----------------|
| CADC0874异常 | 无法确定物理原因 | ✅ 完整5段分析，4条原因 |
| 光电二极管零值 | 3条路径 | ✅ 4条原因+关联F1指标 |
| CADC0872异常形状 | 未测 | ✅ 交叉验证建议 |
| 算法效果 | 未测 | 诚实+常识推理 |

### 反思
- 实验知识入库是最大提升——原来知识库只有外部PDF，缺少我们自己的发现
- Prompt框架化让回答结构化、专业性提升
- NVIDIA→火山引擎解决了超时问题，国内API更适合实际使用
- 检索分数0.56-0.66区间，低于0.7阈值会被过滤，可能需要调低阈值

---

## 下一步
- RAG集成到Streamlit UI（等王翌钧5/2启动）
- IForest子采样实验ψ=256（可进一步提升Phase1）
- Git push（commit 6e5df13仍未推送）
- 陶诗怡手册标注任务跟进（4/30初查）

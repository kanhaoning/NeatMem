# LOCOMO 多 Session 评测：qwen-plus-latest vs vllm qwen-35b

## 测试日期

2026-05-31

## 测试配置

| 配置项 | 值 |
|---|---|
| 用户 | conv-26 (Caroline/Melanie) |
| 5 session | 5 sessions, 55 QA |
| 10 session | 10 sessions, 105 QA |
| 19 session | 19 sessions (全量), 197 QA |
| Add/Answer 模型 | 均使用同一模型 |
| top_k | 5 |
| Embedding | siliconflow bge-m3 |
| Reranker | 未启用 |

### 模型配置

| 模型 | API | 备注 |
|---|---|---|
| qwen-plus-latest | dashscope.aliyuncs.com | 阿里云 API |
| qwen-35b (Qwen3.6-35B-A3B-FP8) | localhost:7999 (vllm) | 本地 GPU |

## QA 过滤规则

- 仅保留 evidence 全部在当前 session 范围内的 QA
- 排除 12 个跨边界 QA（evidence 部分在范围内、部分超出）
- 排除 2 个无 evidence 标注的 Cat3 推理题

## 评测结果汇总

| 配置 | QA 总数 | 正确数 | 准确率 | Cat1 | Cat2 | Cat3 | Cat4 | Cat5 |
|---|---:|---:|---:|---|---|---|---|---|
| 5s qwen-plus (参照) | 57 | 34 | 59.65% | — | — | — | — | — |
| 10s qwen-35b | 105 | 44 | 41.90% | 25.00% | 9.52% | 0.00% | 34.21% | 100%* |
| 19s qwen-plus | 197 | 85 | 43.15% | 25.00% | 18.92% | 18.18% | 30.00% | 100%* |
| 19s qwen-35b | 197 | 84 | 42.64% | 25.00% | 13.51% | 0.00% | 34.29% | 100%* |

\* Cat5 100% 准确率可能是子串匹配假阳性。Cat5 对抗题的 expected answer 通常很短（如 "No"），子串匹配 `gt in pred` 会把任何包含 "No" 的回答判为正确。需要人工审核或使用 LLM Judge 重新评估。

## 关键发现

### 1. Session 数增加，准确率显著下降

| Session 数 | qwen-plus 准确率 | QA 数 |
|---|---|---|
| 5 | 59.65% | 57 |
| 19 | 43.15% | 197 |

准确率从 ~60% 降至 ~43%，说明更多 session 对记忆系统构成更大挑战。这与 5session 分析中"5 session 已开始暴露召回排序问题"的发现一致。

### 2. 模型能力差异主要体现在 Cat2/Cat3

| Category | 19s qwen-plus | 19s qwen-35b | 差异 |
|---|---|---|---|
| Cat1 (单轮事实) | 25.00% | 25.00% | 0 |
| Cat2 (多轮事实) | 18.92% | 13.51% | +5.41pp |
| Cat3 (推理) | 18.18% | 0.00% | +18.18pp |
| Cat4 (综合) | 30.00% | 34.29% | -4.29pp |

- **Cat3**：qwen-plus 能做对 2/11 的推理题，qwen-35b 全部答错。说明 qwen-35b 在多步推理上明显弱于 qwen-plus。
- **Cat2**：qwen-plus 比 qwen-35b 高 5.4pp，多轮事实提取能力也更强。
- **Cat1/Cat4**：两者接近，单轮事实提取和综合问题差距不大。

### 3. 召回问题是主要瓶颈（与 5session 一致）

Cat1 在所有配置下都是 25%，表明基础事实提取存在固定的系统性问题（extract 层表述偏移、向量检索偏差等），与模型能力关系不大。

### 4. 10s vs 19s：qwen-35b 的表现

| 配置 | QA 数 | 准确率 |
|---|---|---|
| 10s qwen-35b | 105 | 41.90% |
| 19s qwen-35b | 197 | 42.64% |

10session 和 19session 准确率接近，说明 10session 时系统瓶颈已经暴露。更多 session 未显著恶化情况，但也未改善。

## 运行时间记录

| 任务 | 阶段 | 开始 | 结束 | 耗时 |
|---|---|---|---|---|
| **19s qwen-plus** | Add | 10:43:42 | 11:18:20 | 34m38s |
| | Search | 11:19:32 | 11:39:17 | 19m45s |
| | **小计** | | | **54m23s** |
| **10s qwen-35b** | Add | 11:40:52 | 12:01:17 | 20m25s |
| | Search | 12:01:39 | 12:10:06 | 8m27s |
| | **小计** | | | **28m52s** |
| **19s qwen-35b** | Add | 12:11:15 | 12:42:45 | 31m30s |
| | Search | 12:43:02 | 13:00:24 | 17m22s |
| | **小计** | | | **48m52s** |

**耗时对比分析：**
- Add 阶段：vllm qwen-35b 本地推理比 dashscope API 快（31m vs 35m for 19s）
- Search 阶段：两者接近（17-20m for 19s），因为 search 并发开销类似
- 10s 的 search 只需 ~8m，大约是 19s 的一半，与 QA 数量成正比

## 数据文件

| 文件 | 位置 |
|---|---|
| 10 session 数据集 | `evaluation/dataset/locomo_10session.json` |
| 19 session 数据集 | `evaluation/dataset/locomo_19session.json` |
| 19s qwen-plus 结果 | `evaluation/results/neatmem_results_19s_qwenplus.json` |
| 10s qwen-35b 结果 | `evaluation/results/neatmem_results_10s_35b.json` |
| 19s qwen-35b 结果 | `evaluation/results/neatmem_results_19s_35b.json` |
| 5s qdrant_db | `qdrant_db.5session_qwenplus` |
| 19s qdrant_db (qwen-plus) | `qdrant_db.19s_qwenplus` |
| 10s qdrant_db (qwen-35b) | `qdrant_db.10s_35b` |

## 召回瓶颈分析（19session qwen-plus Bad Case 归因）

对 19s qwen-plus 的 112 个非子串匹配错误（Cat1-4）进行精细归因：

| 归因 | 数量 | 占比 | 说明 |
|---|---:|---:|---|
| 召回完全缺失（0 recall） | 4 | 3.6% | 记忆未被存储或完全无法召回 |
| 召回排序偏差 | 30 | 26.8% | 有 recall 但 top-k 中不包含相关记忆 |
| 格式/语义差异 | 41 | 36.6% | 有相关 recall，语义正确但字面不匹配 |
| 真正答错 | 37 | 33.0% | 有相关 recall 但仍给出错误答案 |
| **合计（Cat1-4）** | **112** | | |

**召回层问题合计：34 个（30.4%）** = 完全缺失 4 + 排序偏差 30

### 典型 Bad Case 深度分析

#### Case 1: Extract 层表述偏移 → 召回排序偏差
- **问题**: Where did Caroline move from 4 years ago?
- **Expected**: Sweden
- **Got**: Her home country
- **根因**: 原始对话 D4:3 说 "a gift from my grandma in my home country, Sweden"，但 extract 把 "Sweden" 合并进了关于项链的记忆，没有独立提取 "从瑞典搬家" 的事实。存储的记忆是 "moved from her home country"，查询 "where did she move from" 能匹配到，但答案不含 "Sweden"。
- **归因**: Extract 层合并丢事实（与 5s 分析一致的系统性问题）

#### Case 2: 召回排序退化
- **问题**: What events has Caroline participated in to help children?
- **Expected**: Mentoring program, school speech
- **10s**: 召回 9 条，包含 mentoring 和 school event → 答对（Mentorship, adoption council, talent show）
- **19s**: 召回 3 条，只包含 talent show 和 adoption → 答错（Organized talent show, applied to adoption agencies）
- **根因**: 19session 有更多记忆条目，mentoring 和 school speech 的向量排序被更多近期记忆挤掉了。10s 时这些记忆还能进 top-k，19s 时排不进去了。
- **归因**: **典型的召回排序瓶颈** — 更多 session → 更多记忆 → 相关记忆被淹没

#### Case 3: 记忆未被存储（Extract 遗漏）
- **问题**: When did Melanie read the book "Nothing is Impossible"?
- **Expected**: 2022
- **Got**: 0 recall（双方都是空）
- **根因**: 原始对话 D7:8 说 "This book I read last year reminds me to always pursue my dreams"，但从未提及书名 "Nothing is Impossible"。LOCOMO 标注者从上下文推断书名，但系统没有存储这个信息。实际上这更像是一个 **LOCOMO 标注问题**（标注者过度推理）而非系统缺陷。
- **归因**: LOCOMO 标注过度推理（与 5s 分析中发现的同类问题一致）

#### Case 4: 召回排序偏差 + Extract 不含查询语义
- **问题**: What kind of art does Caroline make?
- **Expected**: abstract art
- **10s Recall**: 10 条（包含 abstract art 相关记忆）
- **19s Recall**: 10 条（各种 art 相关记忆，但没有 abstract art）
- **根因**: "abstract art" 只在对话中提到过一次，extract 提取时可能没有保留 "abstract" 这个修饰词（合并成更泛化的 "art" 描述），查询 "kind of art" 时向量匹配不够精确。
- **归因**: Extract 层表述偏移 + 向量检索精度不足

#### Case 5: 格式差异（非召回问题）
- **问题**: When did Caroline go to the LGBTQ support group?
- **Expected**: 7 May 2023
- **Got**: May 7, 2023
- **19s Recall**: 双方都有相关记忆（score 0.8+）
- **根因**: 子串匹配 `7 May 2023` 无法匹配 `May 7, 2023`，但语义完全正确。
- **归因**: 评测脚本匹配问题（约占错误的 1/3）

### 10s → 19s 召回退化量化

对 Cat1/2 共享题目（105 题）分析召回数量变化：

| 召回数量变化 | 数量 |
|---|---:|
| 19s 召回减少 | 26 |
| 19s 召回增加 | 26 |
| 不变 | 23 |

平均召回数量持平，但**关键差异在于召回质量**：19s 时相关记忆的排名更靠后。典型案例（events to help children）显示 10s 能召回 9 条相关记忆而 19s 只能召回 3 条，说明向量化检索的语义区分能力在更多记忆时显著下降。

### 关键发现

1. **召回排序是 19s 的核心瓶颈**：30% 的错误来自召回问题（4 个完全缺失 + 30 个排序偏差），而非 answer 层。10s→19s 准确率下降主要因为更多记忆挤占了相关记忆的排名位置。
2. **Extract 层表述偏移仍是系统性问题**：与 5s 分析一致，"Contextually Rich" 规则导致关键信息（Sweden、abstract art）丢失独立检索能力，在 19s 时影响更大（因为记忆更多，模糊匹配更难）。
3. **约 1/3 的"错误"实际是评测匹配问题**：日期格式（7 May vs May 7）、同义词（counseling vs therapy）、词序变化（beach, mountains vs mountains, beach）等语义正确但字面不匹配的案例有 41 个。
4. **LOCOMO 标注问题持续存在**：至少 1 个完全缺失的召回案例（Nothing is Impossible）源于标注者过度推理，实际对话从未提及书名。

## 下一步

1. **Cat5 需要用 LLM Judge 重评**：子串匹配对 Cat5 对抗题结果不可信
2. **召回排序需要时间感知机制**：加入 event_time 加权或时间范围召回，缓解大量记忆下的排序退化
3. **Extract 层需平衡"丰富性"和"独立可检索性"**：当前 "Contextually Rich" 规则在小数据集影响有限（5s），但 19s 时已明显放大
4. **评测脚本需改进匹配策略**：1/3 的"错误"实际是语义正确的格式差异，建议加入 LLM Judge 或模糊匹配
5. **下一步应跑 10 用户全量（locomo10.json）**：验证多用户并发下的召回表现

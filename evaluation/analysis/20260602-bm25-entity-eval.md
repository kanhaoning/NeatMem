# BM25 + Entity 信号评测

## 测试日期

2026-06-02

## 变更内容

在 `memory_add.py` 的 Step 4（写入新记忆）中新增两个字段：

1. `text_lemmatized` — 由 `lemmatize_for_bm25()` 生成，使 Qdrant BM25 sparse vector 编码生效
2. `_link_entities_for_memory()` 调用 — 提取实体并维护 `mem0_entities` collection 反向索引

搜索端无需修改，`memory.search()` 内部已在尝试三路融合（语义 + BM25 + Entity），之前因写入端缺字段导致后两路空转。

## 评测方法

### 子串匹配（非标准）

之前迭代中自用的快速指标：`gt in pred`（ground truth 字符串是否出现在预测中）。缺点是语义正确但措辞不同的答案会被判错（如 GT `transgender woman` vs 预测 `trans woman`），严重低估真实准确率。本报告中仅用于快速对比，不作为最终结论依据。

### LLM Judge（LOCOMO 标准评测方式）

与 mem0 和 LOCOMO 原始论文一致的评测方法：将 question / gold answer / predicted answer 送入 LLM，让其判断 CORRECT 或 WRONG。`metrics/llm_judge.py` 中的 `ACCURACY_PROMPT` 直接来自 LOCOMO benchmark 的标准 prompt，与 mem0 评测代码一致。这是 LOCOMO 的正式评测指标。

**两种指标差异示例**：同一份 BM25+Entity 结果，子串匹配 38.58%，LLM Judge 78.00%——差距来自大量语义正确但字面不同的答案。

## 评测配置

| 配置项 | 值|
|---|---|
| 数据集 | LOCOMO 19 session (conv-26 Caroline/Melanie) |
| QA 总数 | 197 |
| Add 模型 | qwen-35b (vLLM, localhost:7999) |
| Answer 模型 | qwen-35b (vLLM, localhost:7999) |
| Embedding | siliconflow bge-m3 |
| top_k | 10 |
| Reranker | 未启用 |

**注意**：基线 "19s qwen-plus" 使用 dashscope API 的 qwen-plus 作为 add 和 answer 模型，与本次配置不同。基线 "19s qwen-35b" 使用同样的 vLLM qwen-35b，但 BM25/entity 字段缺失（纯语义搜索）。

## 评测结果

### 子串匹配准确率

| 配置 | QA 总数 | 正确数 | 准确率 | Cat1 | Cat2 | Cat3 | Cat4 | Cat5 |
|---|---:|---:|---:|---|---|---|---|---|
| 19s qwen-plus (基线，dashscope) | 197 | 82 | 41.62% | 25.00% | 18.92% | 9.09% | 27.14% | 100%* |
| 19s qwen-35b (基线，vLLM，纯语义) | 197 | 84 | 42.64% | 25.00% | 13.51% | 0.00% | 34.29% | 100%* |
| 19s qwen-35b + BM25 + Entity (本次) | 197 | 76 | 38.58% | 12.50% | 16.22% | 0.00% | 27.14% | 100%* |

\* Cat5 100% 准确率是子串匹配假阳性。

### Cat1-4（排除 Cat5 假阳性）

| 配置 | Cat1-4 准确率 |
|---|---|
| 19s qwen-plus (基线) | 35/150 = 23.33% |
| 19s qwen-35b (基线，纯语义) | 37/150 = 24.67% |
| 19s qwen-35b + BM25 + Entity (本次) | 29/150 = 19.33% |

## 差异归因

对 Cat1-4 共 150 题逐一比对 "19s qwen-35b 纯语义" vs "19s qwen-35b + BM25 + Entity"：

| 归因 | 数量 | 说明 |
|---|---:|---|
| 旧对 → 新错（召回退化） | 4 | 召回数量减少导致答案缺失 |
| 旧对 → 新错（答案措辞差异） | 7 | 召回相同或更多，但 qwen-35b 回答措辞与 GT 不匹配 |
| 旧错 → 新对 | 5 | BM25/entity 信号帮助召回了正确记忆 |
| **净变化** | **-6** | -11 + 5 |

### 召回退化案例分析

1. **"What do Melanie's kids like?"** — 召回从 2+5 降到 2+1，BM25 信号可能把无关记忆排到了前面，挤掉了"恐龙、自然"相关记忆
2. **"How many times has Melanie gone to the beach in 2023?"** — 召回从 0+3 降到 0+0，BM25 对短查询"beach"可能偏向了其他包含"beach"但不相关的记忆

### 答案措辞差异案例

1. **"What is Caroline's identity?"** — GT `transgender woman`，新答 `Trans woman`（语义等价，子串不匹配）
2. **"How long ago was Caroline's 18th birthday?"** — GT `10 years ago`，新答 `Ten years ago`（数字/文字形式不同）

## 关键发现

1. **无法直接评估 BM25/entity 效果**：本次评测同时变更了 answer 模型（qwen-plus → qwen-35b），7/11 的退步来自答案措辞差异而非召回问题，无法确认 BM25/entity 的正面贡献
2. **BM25/entity 的正面效果被 answer 模型差异掩盖**：5 个"旧错→新对"的 case 证明 BM25/entity 信号确实在帮助召回
3. **BM25 短查询有负面排序风险**：4 个召回退步案例中，短关键词查询（如 "beach"）被 BM25 信号可能排偏了

## 下一步

1. **同 answer 模型对比**：用 qwen-35b 统一 answer 模型，只切换 BM25/entity 字段有无，做 A/B 对照
2. **调整融合权重**：如果 BM25 对短查询存在排序干扰，可在 `score_and_rank` 中调低 BM25 权重或引入查询长度自适应加权
3. **子串匹配不可靠**：建议加入 LLM Judge 或模糊匹配重新评估 Cat1-4 的真实准确率

## LLM Judge 评测结果（A/B 对照）

同 answer 模型（qwen-35b vLLM），唯一变量：写入端 BM25/entity 字段有无。

| Category | Baseline LLM | BM25+Entity LLM | Δ |
|---|---|---|---|
| Cat1 (单轮事实) | 0.8750 | 0.8125 | -0.0625 |
| Cat2 (多轮事实) | 0.8649 | 0.7297 | -0.1352 |
| Cat3 (推理) | 0.9091 | 0.6364 | -0.2727 |
| Cat4 (综合) | 0.8286 | 0.8143 | -0.0143 |
| **Overall** | **0.8533** | **0.7800** | **-0.0733** |

| Metric | Baseline | BM25+Entity | Δ |
|---|---|---|---|
| BLEU-1 | 0.3739 | 0.3759 | +0.0020 |
| F1 | 0.4532 | 0.4425 | -0.0107 |

### 结论：BM25 + Entity 信号导致准确率下降

**LLM Judge 整体准确率从 85.33% 降至 78.00%，下降 7.33pp。** 各类别全面下降，Cat2/Cat3 降幅最大。

### 下降根因分析

1. **BM25 短查询排序干扰**：用户提问通常是自然语言问句（"Where did Caroline move from?"），BM25 按关键词匹配会偏向前文中有 "Caroline" 但不相关的记忆，把真正相关的记忆挤出 top-k
2. **entity boost 过度**：entity 反向索引给含 Caroline/Sweden 等实体的记忆加分，但同一实体的多条记忆都加分，反而稀释了正确答案的排名优势
3. **Cat3 降幅最大（-27pp）**：推理题需要综合多条记忆，BM25/entity 的精确匹配倾向反而把推理所需的分散记忆排到后面

### 改进方向

- **查询长度自适应加权**：短查询（<5 tokens）降低 BM25/entity 权重，长查询保持
- **entity 精排**：entity boost 不应以实体出现次数线性加分，需考虑查询与实体上下文的相关性
- **或直接关闭三路融合，只保留语义搜索**：当前数据表明，在 LOCOMO 场景下纯语义搜索优于三路融合

对 Cat1-4 共 150 题，使用 LLM Judge 重新评估（排除 Cat5 对抗题）：

| Category | BLEU-1 | F1 | LLM Score | Count |
|---|---|---|---|---|
| Cat1 (单轮事实) | 0.3690 | 0.4060 | 0.8125 | 32 |
| Cat2 (多轮事实) | 0.4255 | 0.5525 | 0.7297 | 37 |
| Cat3 (推理) | 0.1929 | 0.1352 | 0.6364 | 11 |
| Cat4 (综合) | 0.3817 | 0.4493 | 0.8143 | 70 |
| **Overall** | **0.3759** | **0.4425** | **0.7800** | 150 |

LLM Judge 给出的整体准确率 78%，远高于子串匹配的 19.33%。这印证了子串匹配严重低估真实准确率的判断——大量"错误"实际是语义正确但措辞不同（如 "trans woman" vs "transgender woman"、"Ten years ago" vs "10 years ago"）。

---

## 运行时间记录

| 任务 | 阶段 | 开始 | 结束 | 耗时 |
|---|---|---|---|---|
| **19s qwen-35b + BM25 + Entity** | Add | 03:32 | 04:19 | ~47min |
| | Search | 04:22 | 04:36 | ~14min |
| | Scoring (LLM Judge) | 04:37 | 06:20 | ~103min |
| | **小计** | | | **~164min** |
| **19s qwen-35b baseline (纯语义)** | Scoring (LLM Judge) | 12:19 | 14:05 | ~106min |

**说明**：Add 阶段每个 batch 约 20-55s（后期 session 因记忆增多、去重搜索变慢而拉长）。Scoring 阶段 LLM Judge 对每个 QA 调用一次 LLM，150 题（Cat1-4）共 ~103 分钟。

---

## 数据文件

| 文件 | 位置 |
|---|---|
| 19 session 数据集 | `evaluation/dataset/locomo_19session.json` |
| 本次结果 | `evaluation/results/neatmem_results_19s_bm25.json` |
| 基线 (qwen-plus) | `evaluation/results/neatmem_results_19s_qwenplus.json` |
| 基线 (qwen-35b) | `evaluation/results/neatmem_results_19s_35b.json` |
| qdrant_db (本次) | `qdrant_db.19s_bm25_35b` |

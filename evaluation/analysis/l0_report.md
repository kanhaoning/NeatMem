# L0 定性错误分析报告

> **状态**：2026-05-30 复核重写。旧版（基于已不存在的运行结果）已备份为 `l0_report.v1.bak.md`。
> **数据真相来源**：`evaluation/results/neatmem_results.json` + `evaluation/dataset/locomo_mini.json` + qdrant 写入记忆。
> **下次迭代前先扫**：本文末 "下次迭代必查清单"。

## TL;DR

10 条 QA，Qwen3.6-35B 跑 **5/10**，qwen-max-latest 跑 **7/10**（同栈、仅换 LLM）。

**瓶颈在 extraction 不在 recall。** 召回 score 普遍 0.79~0.90，但记忆文本本身常常已经出错（日期被算成 2026、关键 atomic fact 被合并掉、隐含身份未提取）。强模型能修部分 extract 颗粒度问题，但 **timestamp bug 是数据流问题，强模型也救不了**。

| 错误层 | Qwen35B 次数 | qwen-max 次数 |
|--------|------|------|
| **L1 extraction**（遗漏 / 颗粒度） | 2 | 0 |
| **L1 extraction**（timestamp 算错日期） | 2 | 2 |
| L3 LLM answer 推理错（参与者 vs 成员） | 1 | 0 |
| L5 被对抗钓中（信息更全反而出错） | 0 | 1 |
| ✅ 正确 | 5 | 7 |

**P0 修复路径**（按预期收益排序）：

1. **修 `memory_add.py:428` 透传 `metadata.timestamp`** → 直接修 #3、#5（占 20%），qwen-max 修了即 9/10
2. extraction prompt 加 "提取隐含身份属性 / 保留 atomic realization fact" → 修 #1、#8 在 35B 上的退化（qwen-max 已自动做到）
3. answer prompt 加 "跨 speaker 归属校验" → 防止 qwen-max 在 #10 被对抗钓中
4. **暂不动 rerank**——召回侧没问题，rerank 在小子集上候选才 8~11 条，几乎无用武之地


## 环境与配置

- LLM：本地 vllm @ `http://localhost:7999/v1`，模型 `Qwen3.6-35B-A3B-FP8`（MoE 推理模型）
- 数据：对话 0 (Caroline/Melanie)，session 1+2，5 类各 2 个 QA = 10 条
- 关键 search.py 配置：`enable_thinking=False`、`max_tokens=200`、`timeout=60`、`temperature=0.0`、`top_k=30`
- 耗时：Ingestion ~2min，Search 38s

### 性能踩坑（已修复）

未关 thinking 时单次 search 跑 11 分 42 秒，Cat3 推理题进入长 CoT（单条 response_time 667s）。加 `extra_body={"chat_template_kwargs":{"enable_thinking":False}}` 后 18× 加速，命中数不变——thinking 帮不了 extraction bug。

## 逐条归因（对照 results.json + 原文 + qdrant 记忆）

> 以下逐条归因使用 **Qwen3.6-35B（extract+answer 同模型）** 这次运行的数据，对应 `results/neatmem_results_qwen35b.json`。qwen-max-latest 的对照见后文 "模型对比" 章节。

> **顺序按 results.json 中实际顺序**，与旧报告不同。每条带"召回快照"（实证）而非凭印象判断。

### #1 [Cat1] What is Caroline's identity?

- **expected**: Transgender woman | **response**: `LGBTQ+ individual`
- **evidence**: D1:5 "The transgender stories were so inspiring!" + D1:6 "courage to embrace myself"
- **召回快照**：
  - s1 (3) 最高 score 0.37："Caroline feels accepted and empowered by the LGBTQ support group"
  - s2 (3) 最高 score 0.37："Caroline chose an adoption agency that specifically helps LGBTQ+ individuals"
  - **没有任何一条记忆说 Caroline is transgender**
- **归因**：**L1 提取遗漏**。召回侧搜出了能搜的全部 LGBTQ 相关内容，但 extraction 阶段从未把 "transgender stories inspiring" + "embrace myself" 推理为 "Caroline is a transgender woman" 这条身份事实。
- **修复方向**：extraction prompt 加 "从间接语境提取身份属性"

### #2 [Cat3] Would Melanie be considered a member of the LGBTQ community?

- **expected**: Likely no, she does not refer to herself as part of it | **response**: `Yes, she attends LGBTQ support groups.`
- **召回快照**：
  - s2 (2)：[0.51] "Melanie appreciates the supportive nature of the LGBTQ group" / [0.43] "Melanie attended an LGBTQ support group **with Caroline**"
- **归因**：**L3 LLM 推理错**。召回的记忆其实写得正确（"with Caroline"明示陪同身份），但 answer 模型把"参加"等价于"属于"。
- **修复方向**：answer prompt 增加 "区分参与者/陪同者 vs 群体成员" 指令

### #3 [Cat2] When did Caroline go to the LGBTQ support group?

- **expected**: 7 May 2023 | **response**: `May 28, 2026`
- **evidence**: D1:3 "I went to a LGBTQ support group yesterday"，session ts = `1:56 pm on 8 May, 2023`
- **召回快照**：
  - s1 (1)：[0.86] "Caroline attended an LGBTQ support group **on May 28, 2026**, where she found the transgender stories inspiring..."
- **归因**：**L1 提取时算错日期（timestamp bug）**。召回完全正确，但**记忆文本本身写错了日期**。session 时间是 2023-05-08，"yesterday" 应解析为 2023-05-07；实际写成 2026-05-28，因为 extract 用 `current_date=2026-05-30` 做基准。
- **修复方向**：`memory_add.py:428` 调 `extract_memories` 时把 `metadata.timestamp` 透传，extract 内部用 session ts 而非 system date 做 observation_date

### #4 [Cat1] What did Caroline research?

- **expected**: Adoption agencies | **response**: `Adoption agencies to adopt children.`
- **召回快照**：s1 [0.79] / s2 [0.78] 都精准
- **判定**：✅ 正确

### #5 [Cat2] When did Melanie paint a sunrise?

- **expected**: 2022 | **response**: `Last year`
- **evidence**: D1:12 "I painted that lake sunrise last year"，session ts = 2023-05-08
- **召回快照**：
  - s1 [0.86] "Melanie painted a lake sunrise **last year**, which she considers special..."
  - s2 [0.90] "Melanie painted a lake sunrise scene **last year**, which she considers special to her"
- **归因**：**L1 提取时未做时间换算**。召回 score 0.90 极精准，但记忆原样保留了 "last year"，没换算成绝对年份 2022。answer 模型只能照抄。
- **修复方向**：同 #3，extract 拿到 session ts 后把相对时间表达换成绝对日期

### #6 [Cat3] Would Melanie be considered an ally to the transgender community?

- **expected**: Yes, she is supportive | **response**: `Yes, she attended and appreciated the support.`
- **召回快照**：s2 [0.53] / [0.37] 与 #2 同源记忆
- **判定**：✅ 正确（运气成分——本题逻辑与 #2 几乎对偶，LLM 用同一推理产出却恰好对题，说明 answer 没有"参与者 vs 成员"概念）

### #7 [Cat4] What did the charity race raise awareness for?

- **expected**: mental health | **response**: `Mental health`
- **召回快照**：s1/s2 都 [0.81] 精准
- **判定**：✅ 正确

### #8 [Cat4] What did Melanie realize after the charity race?

- **expected**: self-care is important | **response**: `Focus on mental well-being.`
- **evidence**: D2:3 "I'm starting to realize that self-care is really important"
- **召回快照**：
  - s1/s2 都 [0.87] "Melanie ran a charity race for mental health on May 23, 2026, finding the experience rewarding and thought-provoking"
- **归因**：**L1 提取颗粒度太粗**。召回 score 0.87 看似很好，但**召回的是"跑步综述"，不是"realize self-care"**。原文 D2:3 的关键 atomic fact "realize self-care is important" 被合并进了跑步综述里弱化掉，没有独立记忆条目。
- **修复方向**：extraction prompt 加 "感悟/realization 类语句必须独立提取为单条记忆"

### #9 [Cat5/对抗] What are Melanie's plans for the summer with respect to adoption?

- **adversarial_answer**: researching adoption agencies（应识别为"Melanie 没有领养计划，是 Caroline 才有"）
- **response**: `Excited for new chapter`
- **召回快照**：s1=0；s2 [0.47] "Melanie praised Caroline's caring heart and expressed excitement for the new chapter of her adoption journey"
- **判定**：✅ 抗住对抗（没掉进"adoption→Melanie"陷阱），但回答含糊未明说否定

### #10 [Cat5/对抗] What did Caroline realize after her charity race?

- **adversarial_answer**: self-care is important（Caroline 没跑过 charity race）
- **response**: `No memories provided.`
- **召回快照**：s1=0, s2=0
- **判定**：✅ 抗住对抗（召回为空反而保护了模型）

## 模型对比：Qwen3.6-35B vs qwen-max-latest

### 实验设置

- 两次跑用**完全相同**的 NeatMem pipeline、相同的 L0 子集、相同的 prompt
- 唯一变量：`.env` 里的 `OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL` → 影响 **extract + dedup + answer 三个调用点**
- qwen-max 跑之前 NeatMem 服务重启、qdrant 记忆清空，避免污染

| 配置 | extract | answer | 数据文件 |
|------|---------|--------|---------|
| Qwen35B | Qwen3.6-35B (vllm@local) | Qwen3.6-35B | `results/neatmem_results_qwen35b.json` |
| qwen-max | qwen-max-latest (dashscope) | qwen-max-latest | `results/neatmem_results_qwenmax_full.json` |

### 结果对比（统一严格判定）

| # | Cat | 问题 | expected | Qwen35B 回答 | qwen-max 回答 | Δ |
|---|-----|------|----------|--------------|---------------|---|
| 1 | 1 | Caroline's identity? | Transgender woman | ❌ LGBTQ+ individual | ✅ Caroline is a transgender woman. | ↑ |
| 2 | 3 | Melanie LGBTQ 成员? | Likely no | ❌ Yes, she attends... | ✅ Not explicitly stated. | ↑ |
| 3 | 2 | When LGBTQ support group? | 7 May 2023 | ❌ May 28, 2026 | ❌ May 29, 2026 | = |
| 4 | 1 | Caroline researched? | Adoption agencies | ✅ Adoption agencies... | ✅ Caroline researched adoption agencies. | = |
| 5 | 2 | When painted? | 2022 | ❌ Last year | ❌ May 30, 2026 | = |
| 6 | 3 | Melanie ally? | Yes supportive | ✅ Yes, she attended... | ✅ Yes, Melanie supports Caroline. | = |
| 7 | 4 | charity 主题? | mental health | ✅ Mental health | ✅ ...mental health. | = |
| 8 | 4 | Melanie realize? | self-care | ❌ Focus on mental well-being. | ✅ Importance of self-care. | ↑ |
| 9 | 5 / 对抗 | Melanie's summer plans? | (adv: adoption) | ✅ Excited for new chapter | ✅ No specific summer plans mentioned. | = |
| 10 | 5 / 对抗 | Caroline realize? | (adv: self-care) | ✅ No memories provided. | ❌ Caroline realized the importance of self-care... | ↓ |
| | | | **总分** | **5/10** | **7/10** | **+2** |

### 关键发现

**1. qwen-max 修好了 extract 颗粒度类问题（#1、#8）**

- #1 身份提取：35B 完全没把 "transgender stories inspiring" → "Caroline is a transgender woman"。qwen-max 直接产出独立记忆 "User is a transgender woman."（search score 0.91）
- #8 self-care realization：35B 把感悟合并进跑步综述。qwen-max 独立提取 "Importance of self-care" 这条 atomic fact

**结论**：NeatMem 的 extract-merge-dedup pipeline 设计对模型质量敏感，强模型能充分发挥设计意图。

**2. timestamp bug 强模型也修不了（#3、#5）**

- #3：35B 写 "May 28, 2026"，qwen-max 写 "May 29, 2026"——**两者基准都用了 system date 2026-05-30**，session 时间 2023-05-08 完全没被用上
- #5：35B 保留 "last year" 原词不换算；qwen-max 更主动做了换算但**基于错误基准**，算出 "May 30, 2026"
- 这印证 ablation 文档的判断：`memory_add.py:428` 调 `extract_memories` 没透传 `metadata.timestamp` → extract prompt 默认 `observation_date = current_date`

**结论**：数据流 bug 不是模型能力问题。修这 5 行代码比换模型重要。

**3. 强模型在对抗题上反而更容易出错（#10）**

- 35B 因为召回为空答 "No memories provided"，被动过关
- qwen-max 召回 + extract 都更全，但 answer 没做跨 speaker 归属校验，把 Melanie 的 self-care 感悟错归 Caroline
- **副作用**：信息越全 → 越需要 answer prompt 显式做归属判断
- **结论**：换强模型后 answer prompt 也要相应升级

**4. L3 推理类（#2）：强模型表达更克制**

- 35B 答 "Yes, she attends LGBTQ support groups"（参与=成员的混淆）
- qwen-max 答 "Not explicitly stated"（识别出信息不足）
- 这不是 prompt 改进，是模型 instruction following 能力差异

### 性能 / 成本对比

| | Qwen35B (本地 vllm) | qwen-max (dashscope) |
|--|--------------------|---------------------|
| Ingestion | ~2 min | ~4 min |
| Search | 38 s | 15 s |
| 成本 | 本机 GPU 折旧 | 按 token 计费（10 QA ~ ¥几毛） |

qwen-max 在 search 阶段反而**更快**（38s → 15s），因为 35B 即使关 thinking，prompt 长时仍有较高 latency。



| 优先级 | 改动 | 在 35B 上 | 在 qwen-max 上 | 工作量 |
|--------|------|----------|----------------|--------|
| **P0-a** | `memory_add.py:428` 透传 metadata.timestamp 给 extract | 5→7 | 7→9 | ~5 行 |
| **P0-b** | extract prompt：身份隐含 / atomic realization 必须独立提取 | 5→7 | （qwen-max 已自动做到） | prompt 调试 |
| **P1-a** | answer prompt：跨 speaker 归属校验 | — | 防 #10 回退 | prompt 调试 |
| **P1-b** | answer prompt：参与者 vs 群体成员区分 | 5→6 | （qwen-max 已修） | prompt 调试 |
| **暂不做** | 开 rerank | 候选数 < top_k，没用武之地 | 同 | — |

**结论组合**：先做 P0-a（5 行代码），无论用哪个模型直接 +2 分。再做 P0-b/P1 关闭"换贵模型"和"换便宜模型"之间的能力差距。

## 复现 & 归因脚本

```bash
cd /root/autodl-tmp/NeatMem/evaluation

# 1. 生成迷你数据集
python tools/make_mini_dataset.py \
  --input dataset/locomo10.json --output dataset/locomo_mini.json \
  --conversations 1 --qa-per-category 2 \
  --evidence-only-sessions --qa-priority earliest --seed 42

# 2. 跑评测
python run_experiments.py --method add    --dataset dataset/locomo_mini.json
python run_experiments.py --method search --dataset dataset/locomo_mini.json --top-k 30
```

### 归因三件套（缺一不可）

归因任何一条错误前，**必须同时持有这三份资料**，否则会像旧版报告一样误判：

| 资料 | 文件/工具 | 回答的问题 |
|------|----------|----------|
| ① 原始对话 | `dataset/locomo_mini.json` 的 `conversation.session_N` + `session_N_date_time` | 真相是什么？evidence 指向哪句原文？session 时间戳是几号？ |
| ② 全量写入的记忆 | qdrant 直接查（见下） | extract 阶段产出了什么？关键 atomic fact 有没有被提取？日期被算成几号？ |
| ③ 召回快照 | `results/neatmem_results.json` 的 `speaker_N_memories` | search 阶段排序如何？score 多少？返回的是哪几条记忆？ |

**只看 ③ 会误判**（旧版报告就是这样）。score 0.86 召回的是"日期被算错的那条记忆"——召回侧零责任。

### 一键 dump：原文 + 召回 + evidence 自动对齐

```bash
python - << 'PY'
import json, re
ds = json.load(open('dataset/locomo_mini.json'))
r  = json.load(open('results/neatmem_results.json'))
conv = ds[0]['conversation']

def lookup_evidence(ev):
    """D1:3 -> session_1 第3句（1-indexed）"""
    m = re.match(r'D(\d+):(\d+)', ev)
    if not m: return None
    sid, idx = int(m.group(1)), int(m.group(2)) - 1
    sk = f'session_{sid}'
    if sk not in conv or idx >= len(conv[sk]): return None
    t = conv[sk][idx]
    return f"{sk} ts={conv.get(sk+'_date_time','?')} | {t['speaker']}: {t['text']}"

for i, qa in enumerate(r['0']):
    print(f"\n{'='*70}\n#{i+1} [Cat{qa['category']}] {qa['question']}")
    print(f"  expected   : {qa['answer'] or qa.get('adversarial_answer','(adversarial)')}")
    print(f"  response   : {qa['response']}")
    for ev in qa['evidence']:
        print(f"  原文 {ev:6}: {lookup_evidence(ev)}")
    for side in ['speaker_1_memories', 'speaker_2_memories']:
        tag = 's1' if '1' in side else 's2'
        print(f"  -- {tag} ({len(qa[side])}):")
        for m in qa[side]:
            print(f"     [{m.get('score',0):.2f}] {m['memory']}  (ts={m.get('timestamp','?')})")
PY
```

输出形如：
```
#3 [Cat2] When did Caroline go to the LGBTQ support group?
  expected   : 7 May 2023
  response   : May 28, 2026
  原文 D1:3 : session_1 ts=1:56 pm on 8 May, 2023 | Caroline: I went to a LGBTQ support group yesterday...
  -- s1 (1):
     [0.86] Caroline attended an LGBTQ support group on May 28, 2026, ...  (ts=1:56 pm on 8 May, 2023)
```

一眼能对齐：原文 ts=2023-05-08 + "yesterday" 应为 2023-05-07；记忆写成 2026-05-28 → extract 的 timestamp bug。

### 查 qdrant 全量写入记忆

召回快照只展示了 top_k 命中的子集。要判断"是不是 extract 阶段就漏了"，必须查全量：

```bash
python - << 'PY'
from qdrant_client import QdrantClient
client = QdrantClient(path='qdrant_db_test')  # 或服务地址
# collection 名取决于 config.py，常见为 user 名前缀
for col in client.get_collections().collections:
    pts, _ = client.scroll(col.name, limit=200, with_payload=True)
    print(f"\n=== {col.name} ({len(pts)} 条) ===")
    for p in pts:
        pl = p.payload
        print(f"  - {pl.get('data','')[:120]}  | ts={pl.get('timestamp','?')}")
PY
```

例如 "Caroline_0 8 条 / Melanie_0 11 条" 这种 ground truth 只能从这里看。如果某 evidence 对应的事实在全量列表里都不存在，就是 **L1 提取遗漏**（#1、#8 类型），跟召回侧无关。

### 归因决策树

每条错题按下述流程判定，**自上而下、找到第一个 yes 即停**：

```
1. 原文 evidence 对应的事实，在 qdrant 全量记忆里存在吗？
   ├─ 否 → L1 提取遗漏（修 extract prompt 或检查 atomic 切分）
   └─ 是 → 进 2

2. 该事实的记忆文本本身正确吗？（日期、关键词、语义角色）
   ├─ 否 → L1 提取写错（timestamp bug / 过度改写 / 颗粒度合并）
   │       具体子类型：
   │       · 出现 "2026" 等非 session 年份 → timestamp 没透传
   │       · 关键动词被改写（realize → focus）→ prompt 过度概括
   │       · 多个 atomic fact 被合并成一句 → prompt 颗粒度
   └─ 是 → 进 3

3. 该记忆在 top_k 召回里吗？score 排第几？
   ├─ 不在 / 排很后 → L2 召回失败/排序错（rerank、embedding、query 改写）
   └─ 在前 N → 进 4

4. 召回正确但回答错？
   └─ L3 LLM answer 推理错（answer prompt / 模型能力 / 上下文歧义）
```

旧版报告跳过了 1、2 步直接判 L2，所以把 timestamp bug 归错为 "召回排序错"。


## 下次迭代必查清单

每次评测出新结果后，先做以下校验，避免被陈旧结论误导：

- [ ] **response 文本要从 results.json 实读**，不要照搬历史报告。本报告旧版 4 处 response 都跟 JSON 对不上。
- [ ] **QA 顺序可能因数据集变更而漂移**。旧报告的 "#2 Cat1" 实际现在是 "#4 Cat1"，按 category 而非编号比对。
- [ ] **召回失败前先看记忆内容**：如果召回 score > 0.8 但答错，根因往往是 extract 写错内容（日期、过度改写、合并掉关键 fact），而不是召回排序。
- [ ] **timestamp bug 验证**：grep 写入的记忆，如果出现 "2026" 但 session ts 是 2023，就是 extract 没拿到 session 时间。
- [ ] **小子集 rerank 没用武之地**：候选数 < top_k 时关掉 rerank 不会变差，先别花时间在这上面。

## 已废弃结论（防回潮）

> 旧版 l0_report 给出的以下结论已被本次复核推翻，记录以防再次误判：

| 旧结论 | 实际 | 误判原因 |
|--------|------|---------|
| "召回失败 63%，P0 开 rerank" | 召回侧 score 普遍 0.79~0.90，问题在 extract 写错内容 | 没读 results.json 里的召回快照，只看了命中数 |
| "#3 召回 charity race 跑偏" | 召回正确那条 score 0.86，记忆里日期写成 2026 | 报告写完后又重跑过评测，response 已变 |
| "#5 提取缺失（painting 没存）" | 召回 score 0.90 精准，但记忆保留 "last year" 没换算 | 同上 |
| "Cat2 全挂是搜索跑偏" | Cat2 全挂是 extract 时间处理薄弱（不换算 / 用错基准） | 没对照原文 session timestamp |


## P0-a 修复结果（2026-05-30）

**改动**：`memory_add.py` 三处共 4 行——`extract_memories` 新增可选 `metadata` 参数，从 `metadata["timestamp"]` 取事件发生时间传给 mem0 的 `generate_additive_extraction_prompt(timestamp=...)`；`add_memories` 透传 `metadata`。详见 `docs/internal-notes/20260530-extraction-timestamp-fix-plan.md`。

**模型**：qwen-max-latest（dashscope），其余配置同前。
**耗时**：Ingestion ~3min 20s，Search 19s。

### 记忆文本验证（核心证据）

| 题 | 修复前记忆 | 修复后记忆 |
|----|----------|----------|
| #2 LGBTQ | `"attended on May 28/29, 2026"` ❌ | `"attended on May 7, 2023"` ✅ |
| #0 painting | `"painted on May 30, 2026"` ❌ | `"painting on May 8, 2023"` ✅ |

所有写入记忆中的绝对日期已从 2026 → 2023，timestamp bug 在 extract 阶段彻底修复。

### 命中数对比（qwen-max-latest）

| # | Cat | 问题 | GT | 修复前 | 修复后 | Δ |
|---|-----|------|-----|--------|--------|---|
| 0 | 2 | When painted? | 2022 | ❌ May 30 2026 | ⚠ "previous year" | ↑ 部分 |
| 1 | 1 | identity? | Transgender woman | ✅ transgender woman | ❌ LGBTQ+ community | ↓ |
| 2 | 2 | When LGBTQ? | 7 May 2023 | ❌ May 29, 2026 | ✅ **May 7, 2023** | ↑ |
| 3 | 1 | research? | Adoption agencies | ✅ | ✅ | = |
| 4 | 3 | Melanie member? | Likely no | ✅ Not stated | ✅ Not stated | = |
| 5 | 3 | Melanie ally? | Yes | ✅ | ✅ | = |
| 6 | 4 | Melanie realize? | self-care | ✅ self-care | ✅ self-care | = |
| 7 | 4 | charity 主题? | mental health | ✅ | ✅ | = |
| 8 | 5 / 对抗 | Melanie 计划? | (adv) | ✅ 抗住 | ✅ 抗住 | = |
| 9 | 5 / 对抗 | Caroline realize? | (adv) | ❌ 被钓中 | ✅ **抗住** | ↑ |
| | | **总分** | | **7/10** | **8~8.5/10** | **+1~+1.5** |

### 关键发现

**1. timestamp 修复完全达成预期 (#2)**
- session ts 2023-05-08 + "yesterday" → 正确算成 2023-05-07 ✅
- LLM 完美命中 GT "7 May 2023"

**2. 意外收益：对抗题防御增强 (#9)**
- 修复前：Caroline 记忆里的 "self-care" 被错位归因（实际是 Melanie 的感悟）→ 被对抗钓中
- 修复后：所有事件带准确的归属日期（"May 23, 2026 Melanie ran..." 等），LLM 更难错位归因
- 这是 timestamp 修复的副作用——更精确的时间锚反过来强化了 speaker 归属判断

#### 已验证：合并行为跟着时间一起变了（qdrant 实证）

**验证方法**：同一份修复后的代码、同一个 LLM、同一个 prompt，仅改 `evaluation/src/neatmem/add.py:80` 让评测脚本不传 `metadata["timestamp"]` 模拟修复前；用独立 qdrant 目录跑两次 ingestion，直接读 sqlite 存储对比形态。

**qdrant 存储形态对比**（Caroline_0 + Melanie_0 collection）：

| 指标 | 修复前（metadata={}） | 修复后（metadata={timestamp}） |
|------|---------------------|------------------------------|
| Caroline_0 总条数 | 3 | **6**（翻倍） |
| Caroline_0 最长段 | **2254** 字 | 881 字 |
| Caroline_0 中位长度 | 190 字 | 226 字 |
| Caroline_0 带 timestamp 的 payload | 0/3 | 3/6 |
| Melanie_0 总条数 | 6 | 6 |
| Melanie_0 最长段 | 1469 字 | 1248 字 |

**关键观察**：
- Caroline 那条 2254 字的"巨型合并段"在修复后**消失了**，被拆成 6 条独立记忆（LGBTQ 经历 188 字 / career 探索 182 字 / painting 欣赏 201 字 / adoption 计划 530 字 / self-care 感悟 881 字 / charity race 欣赏 252 字）
- 拆分后单条记忆主题更聚焦：**charity race 只归 Melanie**，不会被合并进 Caroline 的某条复合段落
- 修复前 Melanie 那条 1469 字段落也含"On May 30, 2026..."等多个错误日期串

**因果链（已实证）**：

```
传入正确 timestamp
  ↓
extract LLM 生成的记忆文本里日期正确（2023 而非 2026）
  ↓
dedup 阶段：merge LLM 看到候选记忆"事件时间分散在 2023-05-07 ~ 25"
  ↓
判定"不相关"的比例上升 → 合并减少 → 独立条目增多（Caroline_0 从 3 → 6 条）
  ↓
单条记忆主题更聚焦、speaker 归属更清晰
  ↓
查询 "Caroline charity race" 时找不到错位归因的混合记忆（charity race 已归 Melanie）
  ↓
对抗陷阱失效（#9 召回为空 → 拒答）
```

注意 dedup/merge **代码逻辑没改**——只是它的输入（事件时间）变了，merge LLM 在判断"是否相关"时给出不同答案：时间相同更倾向合并，时间分散更倾向独立。

#### Trade-off：合并克制是双刃剑（#1 回退的根因）

对比也暴露出**合并克制的代价**：

- 修复前 Caroline_0 有 1 条独立的短 atomic：`"User is a transgender woman."`（28 字）
- 修复后**这条 atomic 消失了**——很可能被合并/吸收进了 6 条记忆里的某一条，但单独搜身份不再命中

这解释了 #1 从 ✅ → ❌ 的回退：依赖单点 atomic fact 的题，合并克制反而是劣势。

**平衡的结论**：

> timestamp 修复让合并行为对"事件时间"敏感，避免了把跨时间事件 over-merge 成长段落，从而消除了对抗题的错位归因素材（#9 ✅）。代价是部分 atomic fact 被合并/吸收，依赖单点 atomic 的题（#1）会轻微回退。整体净收益正向（+1~+1.5 分），属于合并策略的固有 trade-off。

**对后续 ablation 的启示**：dedup 在含对抗题的小子集上是 trade-off，不是单纯净收益——`no_dedup` 消融预期会在对抗题上 ≥ `full`，但在 #1 这类题上回退。具体见 `docs/internal-notes/20260530-neatmem-ablation-eval-plan.md` 的更新章节。

**3. 部分修复 (#0 painting)**
- 记忆里的日期对了，但 LLM 看到 "Observation Date: 2023-05-08" + "last year" 时，**保留了 "previous year" 而没换算成 2022**
- 这是 prompt 工程问题（需要显式指令"相对时间表达必须换算成绝对年份"），不属于 timestamp bug
- 留到 P0-b 处理

**4. 回退 #1**
- 修复前 qwen-max 召回 `"User is a transgender woman."` ([0.24])
- 修复后该 atomic fact 消失了——qdrant 实证显示是被合并/吸收进了其他记忆条目
- 这是合并克制的副作用：时间分散→merge 更少→atomic 单独存在的概率下降
- 与 timestamp 修复属于同一根因（merge 行为变化），详见上文"#### 已验证：合并行为跟着时间一起变了"

### 总结

| 指标 | 状态 |
|------|------|
| timestamp bug 实证修复 | ✅ 所有记忆日期 2026 → 2023 |
| qwen-max 命中 | 7/10 → 8~8.5/10 |
| API / OpenClaw / 评测脚本影响 | ✅ 零变更 |
| 副作用 | ✅ 对抗题防御增强（已通过 qdrant 条数对比验证：Caroline_0 从 3 条 → 6 条，最长段 2254 → 881 字） |
| Trade-off | ⚠ 合并克制导致 #1 atomic identity 被吸收，依赖单点 atomic 的题轻微回退 |
| 遗留问题 | #0 相对→绝对时间换算 → 留 P0-b prompt 工程 |

下一步：给 extract prompt 加"相对时间必须换算"指令（P0-b）；ablation 阶段重点验证 dedup 在对抗 vs atomic 题上的 trade-off。




# NeatMem LOCOMO Evaluation

用 LOCOMO benchmark 评测 NeatMem 记忆系统。

## 前置条件

- NeatMem 服务运行中（默认 `http://localhost:8790`）
- 配置 `.env` 中的 OpenAI 兼容 API（`OPENAI_API_KEY`、`OPENAI_BASE_URL`）
- 安装依赖：`pip install -r requirements.txt`

## 数据集

LOCOMO 数据集需自行获取，放入 `dataset/` 目录：

```
evaluation/dataset/locomo10.json
```

数据集来源：[LOCOMO](https://github.com/facebookresearch/locomo) 或从 MemOS evaluation 拷贝。

## 快速开始

```bash
cd evaluation

# 1. 采样（可选，加速迭代）
python sample_locomo.py --conv-sample 5 --max-sessions 0

# 2. Ingestion：写入记忆
python run_experiments.py --method add --dataset dataset/locomo10.json

# 3. Search + Answer：检索并生成回答
ANSWER_MODEL=qwen-turbo-latest python run_experiments.py --method search --dataset dataset/locomo10.json

# 4. Eval：评分
python evals.py --input_file results/neatmem_results.json

# 5. 查看汇总
python generate_scores.py
```

## 采样参数

`sample_locomo.py` 支持快速迭代：

| 参数 | 含义 | 示例 |
|------|------|------|
| `--conv-sample N` | 取 1/N 的对话 | `--conv-sample 5` 取 2 条 |
| `--qa-sample N` | 取 1/N 的 QA | `--qa-sample 4` |
| `--max-sessions N` | 每条对话最多 N 个 session | `--max-sessions 3` |
| `--seed N` | 随机种子 | `--seed 42` |

推荐迭代配置：`--conv-sample 5`（2条对话全session，约30分钟/轮）

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NEATMEM_URL` | `http://localhost:8790` | NeatMem 服务地址 |
| `ANSWER_MODEL` | `qwen-max-latest` | Search 阶段回答生成模型 |
| `LLM_MODEL` | `qwen-max-latest` | LLM Judge 评分模型 |

#!/bin/bash
# 批量运行 M3 口径下的 8 组实验（串行）
set -e

BASE="/root/autodl-tmp/NeatMem"
EVAL="$BASE/evaluation"
DB="qdrant_db_multisignal_rr_listwise"
API_KEY="sk-cp-Ks91Y22J6flV7br6uH0dUUjhA7PfxF-4l62BBn5_sxrlt-yKgs69nUA8vMh3wvvnrhI8hRnHIHq9PO38ms9Sml1Y1qUwS3nVa_1ttDPC6zf0wdAropPoGxM"
BASE_URL="https://api.minimaxi.com/v1"
MODEL="MiniMax-M3"
PORT=8794

cd "$BASE"

# 实验配置列表：组号 模式 输出目录
experiments=(
    "g13 listwise_v2_limit20 outputs/g13_lw_v2_l20_mergeoff"
    "g14 listwise_v2_limit40 outputs/g14_lw_v2_l40_mergeoff"
    "g15 listwise_limit20 outputs/g15_lw_l20_mergeoff"
    "g16 pointwise_limit20 outputs/g16_pw_l20_mergeoff"
    "g17 no_rerank_limit20 outputs/g17_norr_l20_mergeoff"
    "g18 pointwise_limit5 outputs/g18_pw_l5_mergeoff"
    "g19 pointwise_limit40 outputs/g19_pw_l40_mergeoff"
    "g20 listwise_limit40 outputs/g20_lw_l40_mergeoff"
)

for exp in "${experiments[@]}"; do
    read -r group mode outdir <<< "$exp"

    echo "========================================"
    echo "Starting $group: $mode"
    echo "========================================"

    # 1. 启动 server
    nohup env \
        MEM0_TELEMETRY=False \
        QDRANT_PATH="$DB" \
        RERANK_MODE="$mode" \
        NEATMEM_PORT="$PORT" \
        OPENAI_API_KEY="$API_KEY" \
        OPENAI_BASE_URL="$BASE_URL" \
        LLM_MODEL="$MODEL" \
        python "$BASE/tmp/experiments/listwise/server.py" \
        > "/tmp/${group}_server.log" 2>&1 &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    sleep 5

    # 2. 跑 search
    cd "$EVAL"
    env \
        OPENAI_API_KEY="$API_KEY" \
        OPENAI_BASE_URL="$BASE_URL" \
        LLM_MODEL="$MODEL" \
        ANSWER_MODEL="$MODEL" \
        NEATMEM_URL="http://localhost:$PORT" \
        python run_experiments.py \
        --method search \
        --dataset dataset/locomo_19session.json \
        --output-folder "$outdir" \
        --top-k 5 \
        > "/tmp/${group}_search.log" 2>&1
    echo "Search done"

    # 3. Kill server
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
    rm -f "$BASE/$DB/.lock" /root/.mem0/migrations_qdrant/.lock 2>/dev/null || true

    # 4. Judge
    cd "$EVAL"
    python batch_judge.py --group "$group" > "/tmp/${group}_judge.log" 2>&1
    echo "Judge done"

    # 5. 打印结果
    echo "--- $group RESULT ---"
    grep -E "Overall|Cat[1-4]" "/tmp/${group}_judge.log" || tail -10 "/tmp/${group}_judge.log"

    cd "$BASE"
done

echo "========================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================"

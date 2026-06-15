#!/bin/bash
# Link-Expanded Listwise v2 实验
# 3 组：P1 (baseline), L1 (same_fact), L2 (same_fact+same_topic)
set -e

BASE="/root/autodl-tmp/NeatMem"
EVAL="$BASE/evaluation"
DB="qdrant_db_link_exp"
API_KEY="sk-cp-Ks91Y22J6flV7br6uH0dUUjhA7PfxF-4l62BBn5_sxrlt-yKgs69nUA8vMh3wvvnrhI8hRnHIHq9PO38ms9Sml1Y1qUwS3nVa_1ttDPC6zf0wdAropPoGxM"
BASE_URL="https://api.minimaxi.com/v1"
MODEL="MiniMax-M3"
PORT=8794

cd "$BASE"

# ==========================================
# Phase 1: Ingest link_exp DB
# ==========================================
echo "========================================"
echo "Phase 1: Ingest link_exp DB"
echo "========================================"

# 确保旧 lock 清除
rm -f "$BASE/$DB/.lock" /root/.mem0/migrations_qdrant/.lock 2>/dev/null || true

nohup env \
    MEM0_TELEMETRY=False \
    QDRANT_PATH="$DB" \
    ENABLE_BM25=true \
    ENABLE_ENTITY=true \
    MERGE_STRATEGY=off \
    NEATMEM_PORT="$PORT" \
    OPENAI_API_KEY="$API_KEY" \
    OPENAI_BASE_URL="$BASE_URL" \
    LLM_MODEL="$MODEL" \
    python "$BASE/main.py" \
    > "/tmp/link_exp_ingest_server.log" 2>&1 &
SERVER_PID=$!
echo "Ingest server PID: $SERVER_PID"
sleep 8

cd "$EVAL"
env \
    OPENAI_API_KEY="$API_KEY" \
    OPENAI_BASE_URL="$BASE_URL" \
    LLM_MODEL="$MODEL" \
    NEATMEM_URL="http://localhost:$PORT" \
    python run_experiments.py \
    --method add \
    --dataset dataset/locomo_19session.json \
    --output-folder results/link_exp_add \
    > "/tmp/link_exp_ingest.log" 2>&1

echo "Ingest done"
kill "$SERVER_PID" 2>/dev/null || true
sleep 3
rm -f "$BASE/$DB/.lock" /root/.mem0/migrations_qdrant/.lock 2>/dev/null || true

# ==========================================
# Phase 2: Run experiments (P1, L1, L2)
# ==========================================

experiments=(
    "P1 listwise_v2_limit20 results/link_P1"
    "L1 listwise_v2_link_sf_limit20 results/link_L1"
    "L2 listwise_v2_link_sf_st_limit20 results/link_L2"
)

for exp in "${experiments[@]}"; do
    read -r group mode outdir <<< "$exp"

    echo "========================================"
    echo "Starting $group: $mode"
    echo "========================================"

    # 清除 lock
    rm -f "$BASE/$DB/.lock" /root/.mem0/migrations_qdrant/.lock 2>/dev/null || true

    # 启动 server
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

    # 跑 search
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

    # Kill server
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
    rm -f "$BASE/$DB/.lock" /root/.mem0/migrations_qdrant/.lock 2>/dev/null || true

    # Judge
    cd "$EVAL"
    env LLM_MODEL="$MODEL" python batch_judge.py --group "$group" > "/tmp/${group}_judge.log" 2>&1
    echo "Judge done"

    # 打印结果
    echo "--- $group RESULT ---"
    cat "/tmp/${group}_judge.log"

    cd "$BASE"
done

echo "========================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================"

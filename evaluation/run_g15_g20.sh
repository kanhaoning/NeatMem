#!/bin/bash
# Run G15-G20 experiments
set -e

BASE="/root/autodl-tmp/NeatMem"
EVAL="$BASE/evaluation"
DB="/root/autodl-tmp/NeatMem/qdrant_db_multisignal_rr_listwise"
API_KEY="sk-cp-Ks91Y22J6flV7br6uH0dUUjhA7PfxF-4l62BBn5_sxrlt-yKgs69nUA8vMh3wvvnrhI8hRnHIHq9PO38ms9Sml1Y1qUwS3nVa_1ttDPC6zf0wdAropPoGxM"
BASE_URL="https://api.minimaxi.com/v1"
MODEL="MiniMax-M3"
PORT=8794

cd "$BASE"

# group_name rerank_mode limit output_dir workers
experiments=(
    "g15 listwise_limit20 20 outputs/g15_lw_l20_mergeoff 8"
    "g16 pointwise_limit20 20 outputs/g16_pw_l20_mergeoff 1"
    "g17 no_rerank_limit20 20 outputs/g17_norr_l20_mergeoff 8"
    "g18 pointwise_limit5 5 outputs/g18_pw_l5_mergeoff 1"
    "g19 pointwise_limit40 40 outputs/g19_pw_l40_mergeoff 1"
    "g20 listwise_limit40 40 outputs/g20_lw_l40_mergeoff 8"
)

for exp in "${experiments[@]}"; do
    read -r group mode limit outdir workers <<< "$exp"

    echo "========================================"
    echo "Starting $group: $mode (limit=$limit, workers=$workers)"
    echo "========================================"

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
        --workers "$workers" \
        > "/tmp/${group}_search.log" 2>&1
    echo "Search done"

    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2

    cd "$EVAL"
    env LLM_MODEL="$MODEL" python batch_judge.py --group "$group" > "/tmp/${group}_judge.log" 2>&1
    echo "Judge done"

    echo "--- $group RESULT ---"
    cat "/tmp/${group}_judge.log"

    cd "$BASE"
done

echo "========================================"
echo "G15-G20 EXPERIMENTS COMPLETE"
echo "========================================"

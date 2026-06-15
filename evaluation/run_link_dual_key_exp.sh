#!/bin/bash
# Link rerank 实验 — 独立服务 + 双 key proxy 版本
# 不修改根目录代码，全部在 tmp/experiments/link_dual_key/ 下运行
set -e

BASE="/root/autodl-tmp/NeatMem"
EXP="$BASE/tmp/experiments/link_dual_key"
EVAL="$BASE/evaluation"
DB="qdrant_db_link_exp"

# 两个 MiniMax key（必需）
KEY1="${OPENAI_API_KEY_1:-sk-cp-Ks91Y22J6flV7br6uH0dUUjhA7PfxF-4l62BBn5_sxrlt-yKgs69nUA8vMh3wvvnrhI8hRnHIHq9PO38ms9Sml1Y1qUwS3nVa_1ttDPC6zf0wdAropPoGxM}"
KEY2="${OPENAI_API_KEY_2:-}"

if [ -z "$KEY2" ]; then
    echo "ERROR: 请设置 OPENAI_API_KEY_2"
    exit 1
fi

BASE_URL="https://api.minimaxi.com"
PROXY_URL="http://localhost:8791"
PROXY_PORT=8791
SERVER_PORT=8790
MODEL="MiniMax-M3"
ANSWER_MODEL="MiniMax-M3"

cd "$BASE"

experiments=(
    "gP1_root_dual llm_listwise_v2 outputs/gP1_root_dual"
    "gC1_root_dual llm_listwise_v2_link_twostage outputs/gC1_root_dual"
    "gC1_ctx_root_dual llm_listwise_v2_link_twostage_context outputs/gC1_ctx_root_dual"
)

# ==========================================
# Phase 0: 启动 Dual Key Proxy
# ==========================================
echo "========================================"
echo "Starting Dual Key Proxy on port $PROXY_PORT"
echo "========================================"

nohup env \
    OPENAI_API_KEY_1="$KEY1" \
    OPENAI_API_KEY_2="$KEY2" \
    TARGET_BASE_URL="$BASE_URL" \
    PROXY_PORT="$PROXY_PORT" \
    python "$EXP/proxy.py" \
    > "/tmp/dual_key_proxy.log" 2>&1 &
PROXY_PID=$!
echo "Proxy PID: $PROXY_PID"

# 等待 proxy 就绪
for i in {1..30}; do
    if curl -s "$PROXY_URL/health" > /dev/null 2>&1; then
        echo "Proxy ready"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo "Proxy failed to start, check /tmp/dual_key_proxy.log"
        kill "$PROXY_PID" 2>/dev/null || true
        exit 1
    fi
done

# ==========================================
# Phase 1-3: 跑实验
# ==========================================
for exp in "${experiments[@]}"; do
    read -r group mode outdir <<< "$exp"

    echo "========================================"
    echo "Starting $group: $mode"
    echo "========================================"

    LOG_DIR="$EVAL/$outdir/logs"
    mkdir -p "$LOG_DIR"

    # 启动 server（独立代码）
    # OPENAI_API_KEY 设 dummy，proxy 会覆盖 Authorization
    nohup env \
        MEM0_TELEMETRY=False \
        QDRANT_PATH="$DB" \
        RERANK_MODE="$mode" \
        LLM_RERANK=true \
        ENABLE_BM25=true \
        ENABLE_ENTITY=true \
        MERGE_STRATEGY=off \
        NEATMEM_PORT="$SERVER_PORT" \
        OPENAI_API_KEY="dummy" \
        OPENAI_BASE_URL="$PROXY_URL/v1" \
        LLM_MODEL="$MODEL" \
        PYTHONPATH="$EXP:$BASE" \
        python "$EXP/server.py" \
        > "$LOG_DIR/${group}_server.log" 2>&1 &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"

    # 等待 server 就绪
    for i in {1..30}; do
        if curl -s "http://localhost:$SERVER_PORT/health" > /dev/null 2>&1; then
            echo "Server ready"
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            echo "Server failed to start, check $LOG_DIR/${group}_server.log"
            kill "$SERVER_PID" 2>/dev/null || true
            kill "$PROXY_PID" 2>/dev/null || true
            exit 1
        fi
    done

    # 跑 search + answer
    cd "$EVAL"
    env \
        OPENAI_API_KEY="dummy" \
        OPENAI_BASE_URL="$PROXY_URL/v1" \
        ANSWER_BASE_URL="$PROXY_URL/v1" \
        LLM_MODEL="$MODEL" \
        ANSWER_MODEL="$ANSWER_MODEL" \
        NEATMEM_URL="http://localhost:$SERVER_PORT" \
        PYTHONPATH="$EXP:$BASE" \
        python run_experiments.py \
        --method search \
        --dataset dataset/locomo_19session.json \
        --output-folder "$outdir" \
        --top-k 5 \
        --workers 8 \
        > "$LOG_DIR/${group}_search.log" 2>&1
    echo "Search done"

    # Kill server
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 3

    # Judge
    cd "$EVAL"
    env LLM_MODEL="$MODEL" python batch_judge.py --group "$group" > "$LOG_DIR/${group}_judge.log" 2>&1
    echo "Judge done"

    echo "--- $group RESULT ---"
    cat "$LOG_DIR/${group}_judge.log"

    cd "$BASE"
done

# 关闭 proxy
kill "$PROXY_PID" 2>/dev/null || true

echo "========================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================"

for exp in "${experiments[@]}"; do
    read -r group mode outdir <<< "$exp"
    echo ""
    echo "=== $group ==="
    cat "$EVAL/$outdir/logs/${group}_judge.log"
done

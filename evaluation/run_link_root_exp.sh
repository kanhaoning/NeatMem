#!/bin/bash
# Link rerank 实验 — root 配置版本
# 复用 qdrant_db_link_exp，对比 P1 / C1 / 优化版在 root main.py 下的表现
set -e

BASE="/root/autodl-tmp/NeatMem"
EVAL="$BASE/evaluation"
DB="qdrant_db_link_exp"
API_KEY="sk-cp-Ks91Y22J6flV7br6uH0dUUjhA7PfxF-4l62BBn5_sxrlt-yKgs69nUA8vMh3wvvnrhI8hRnHIHq9PO38ms9Sml1Y1qUwS3nVa_1ttDPC6zf0wdAropPoGxM"
BASE_URL="https://api.minimaxi.com/v1"
MODEL="MiniMax-M3"
ANSWER_MODEL="MiniMax-M3"
PORT=8790

cd "$BASE"

# 注意：本脚本不复做 ingestion，直接复用已有的 qdrant_db_link_exp
# 如果该 DB 的 .lock 文件是旧进程遗留的，Qdrant 通常可以自动处理；
# 若启动失败，请手动检查并清除 lock。

experiments=(
    "gP1_root llm_listwise_v2 outputs/gP1_root"
    "gC1_root llm_listwise_v2_link_twostage outputs/gC1_root"
    "gC1_ctx_root llm_listwise_v2_link_twostage_context outputs/gC1_ctx_root"
)

for exp in "${experiments[@]}"; do
    read -r group mode outdir <<< "$exp"

    echo "========================================"
    echo "Starting $group: $mode"
    echo "========================================"

    # 日志目录跟随实验输出目录（使用 $EVAL 绝对路径，与 output-folder 一致）
    LOG_DIR="$EVAL/$outdir/logs"
    mkdir -p "$LOG_DIR"

    # 启动 server
    nohup env \
        MEM0_TELEMETRY=False \
        QDRANT_PATH="$DB" \
        RERANK_MODE="$mode" \
        LLM_RERANK=true \
        ENABLE_BM25=true \
        ENABLE_ENTITY=true \
        MERGE_STRATEGY=off \
        NEATMEM_PORT="$PORT" \
        OPENAI_API_KEY="$API_KEY" \
        OPENAI_BASE_URL="$BASE_URL" \
        LLM_MODEL="$MODEL" \
        python "$BASE/main.py" \
        > "$LOG_DIR/${group}_server.log" 2>&1 &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"

    # 等待服务就绪
    for i in {1..30}; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "Server ready"
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            echo "Server failed to start, check $LOG_DIR/${group}_server.log"
            kill "$SERVER_PID" 2>/dev/null || true
            exit 1
        fi
    done

    # 跑 search + answer
    cd "$EVAL"
    env \
        OPENAI_API_KEY="$API_KEY" \
        OPENAI_BASE_URL="$BASE_URL" \
        LLM_MODEL="$MODEL" \
        ANSWER_MODEL="$ANSWER_MODEL" \
        NEATMEM_URL="http://localhost:$PORT" \
        python run_experiments.py \
        --method search \
        --dataset dataset/locomo_19session.json \
        --output-folder "$outdir" \
        --top-k 5 \
        --workers 4 \
        > "$LOG_DIR/${group}_search.log" 2>&1
    echo "Search done"

    # Kill server
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 3

    # Judge
    cd "$EVAL"
    env LLM_MODEL="$MODEL" python batch_judge.py --group "$group" > "$LOG_DIR/${group}_judge.log" 2>&1
    echo "Judge done"

    # 打印结果
    echo "--- $group RESULT ---"
    cat "$LOG_DIR/${group}_judge.log"

    cd "$BASE"
done

echo "========================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================"

# 汇总
for exp in "${experiments[@]}"; do
    read -r group mode outdir <<< "$exp"
    echo ""
    echo "=== $group ==="
    cat "$EVAL/$outdir/logs/${group}_judge.log"
done

"""Ingest locomo10.json for entity merge verification.

Uses the merged main package neatmem/ code (not the experiment copy).
Set PYTHONPATH=/root/autodl-tmp/NeatMem before running.
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from neatmem.memory_add import add_memories
from neatmem.config import build_memory_store, ENABLE_BM25, HISTORY_DB_PATH, ENTITY_EXTRACTOR_BACKEND, ENTITY_STORE_BACKEND
from neatmem.signals.bm25.factory import create_bm25_index

# CUSTOM_INSTRUCTIONS 开关：默认不传（对齐 91+ 实验配置）
# 设 INGEST_CUSTOM_INSTRUCTIONS=true 时从 add.py import 并传入
_USE_CUSTOM_INSTRUCTIONS = os.environ.get("INGEST_CUSTOM_INSTRUCTIONS", "false").lower() in ("1", "true", "yes")
if _USE_CUSTOM_INSTRUCTIONS:
    from neatmem.evaluation.src.neatmem.add import CUSTOM_INSTRUCTIONS
else:
    CUSTOM_INSTRUCTIONS = None
from neatmem.storage.message.factory import create_message_store
from neatmem.signals.entity.factory import create_entity_extractor
from neatmem.storage.entity.factory import create_entity_store

# --- server vs local mode switch ---
QDRANT_HOST = os.environ.get("QDRANT_HOST", "")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

if QDRANT_HOST:
    print(f"[ingest_locomo] Qdrant SERVER mode: {QDRANT_HOST}:{QDRANT_PORT}", flush=True)
else:
    print(f"[ingest_locomo] Qdrant LOCAL mode: {os.environ.get('QDRANT_PATH', 'qdrant_db')}", flush=True)

memory = build_memory_store()
message_store = create_message_store(HISTORY_DB_PATH)

bm25_index = create_bm25_index(
    "qdrant_sparse" if ENABLE_BM25 else "none",
    vector_store=memory.vector_store,
    collection_name=memory.collection_name,
)

entity_extractor = create_entity_extractor(ENTITY_EXTRACTOR_BACKEND)
entity_store = create_entity_store(
    ENTITY_STORE_BACKEND,
    qdrant_client=memory.vector_store.client,
    collection_name=os.environ.get("ENTITY_COLLECTION_NAME", f"{memory.collection_name}_entities"),
    vector_size=memory.vector_store.embedding_model_dims,
)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
LLM_MODEL = os.getenv("LLM_MODEL", "MiniMax-M3")

DATASET = os.environ.get(
    "DATASET", os.path.join(os.getcwd(), "evaluation/dataset/locomo10.json")
)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "7"))

print_lock = threading.Lock()


def _log(msg: str):
    with print_lock:
        print(msg, flush=True)


def build_user_tasks(data):
    tasks = []
    for idx, item in enumerate(data):
        conv_idx = item.get("_orig_conv_idx", idx)
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]
        session_keys = sorted(
            [k for k in conversation if k.startswith("session_") and "date" not in k],
            key=lambda k: int(k.split("_")[1]),
        )
        sessions_a = []
        sessions_b = []
        for session_key in session_keys:
            date_time_key = session_key + "_date_time"
            timestamp = conversation.get(date_time_key, "")
            chats = conversation[session_key]
            messages = []
            messages_reverse = []
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
            sessions_a.append((session_key, timestamp, messages))
            sessions_b.append((session_key, timestamp, messages_reverse))
        tasks.append({"conv_idx": conv_idx, "speaker": speaker_a, "user_id": f"{speaker_a}_{conv_idx}", "sessions": sessions_a})
        tasks.append({"conv_idx": conv_idx, "speaker": speaker_b, "user_id": f"{speaker_b}_{conv_idx}", "sessions": sessions_b})
    return tasks


def process_user_all_sessions(task):
    idx = task["conv_idx"]
    speaker = task["speaker"]
    user_id = task["user_id"]
    t_start = time.time()
    n_batches = 0
    for session_key, timestamp, msgs in task["sessions"]:
        for i in range(0, len(msgs), BATCH_SIZE):
            batch = msgs[i:i + BATCH_SIZE]
            t0 = time.time()
            req_id = f"c{idx}-{speaker}-{session_key}-b{i}"
            last_err = None
            for attempt in range(4):
                try:
                    add_memories(
                        memory=memory,
                        openai_client=openai_client,
                        llm_model=LLM_MODEL,
                        messages=batch,
                        user_id=user_id,
                        metadata={"timestamp": timestamp},
                        custom_instructions=CUSTOM_INSTRUCTIONS,
                        req_id=req_id,
                        message_store=message_store,
                        entity_extractor=entity_extractor,
                        entity_store=entity_store,
                        bm25_index=bm25_index,
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    _log(f"[{idx}] {speaker} {session_key} batch{i} ERROR attempt {attempt+1}: {type(e).__name__}: {e}")
                    if attempt < 3:
                        # RPM limits need real waits; 1s/2s backoff never outlives a 429 window.
                        time.sleep((5, 30, 120)[attempt])
            if last_err is not None:
                raise last_err
            elapsed = time.time() - t0
            n_batches += 1
            _log(f"  [{idx}] {speaker} {session_key} batch{i:>3} t={elapsed:5.1f}s (user_total_batches={n_batches})")
    total = time.time() - t_start
    _log(f"=== [{idx}] {speaker} ({user_id}) DONE: {n_batches} batches in {total/60:.1f}min ===")
    return {"user_id": user_id, "batches": n_batches, "duration_sec": total}


def main():
    with open(DATASET, "r") as f:
        data = json.load(f)
    tasks = build_user_tasks(data)
    print(f"Loaded {len(data)} conversations -> {len(tasks)} user-level tasks", flush=True)
    print(f"BATCH_SIZE={BATCH_SIZE}, MAX_WORKERS={MAX_WORKERS}", flush=True)
    print(f"LLM_MODEL={LLM_MODEL}", flush=True)
    print(f"OPENAI_BASE_URL={os.getenv('OPENAI_BASE_URL')}", flush=True)
    print(f"HISTORY_DB_PATH={HISTORY_DB_PATH}", flush=True)

    t_global = time.time()
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_to_task = {ex.submit(process_user_all_sessions, t): t for t in tasks}
        for fut in as_completed(future_to_task):
            t = future_to_task[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append({"user_id": t["user_id"], "error": f"{type(e).__name__}: {e}"})
                _log(f"!!! task {t['user_id']} FAILED: {type(e).__name__}: {e}")
    total = time.time() - t_global
    print(f"\nALL TASKS DONE in {total/60:.1f}min", flush=True)
    print(f"Successful: {len(results)} / {len(tasks)}", flush=True)
    if errors:
        print(f"FAILURES ({len(errors)}):", flush=True)
        for e in errors:
            print(f"  {e['user_id']:<20} {e['error']}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

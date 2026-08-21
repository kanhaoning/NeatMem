"""LOCOMO Search + Answer — 搜索记忆并生成回答"""

import json
import logging
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

from neatmem.utils.llm_client import build_thinking_extra, extract_response_text
from neatmem.evaluation.prompts import ANSWER_PROMPT, format_memories, _parse_session_date
from .client import NeatMemClient

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class NeatMemSearch:
    def __init__(self, output_path="results/neatmem_results.json", top_k=10, rerank=None):
        self.client = NeatMemClient()
        self.top_k = top_k
        self.rerank = rerank
        self.openai_client = OpenAI(
            api_key=os.getenv("ANSWER_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("ANSWER_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        )
        self.results = defaultdict(list)
        self.output_path = output_path
        self.answer_template = Template(ANSWER_PROMPT)

    def search_memory(self, user_id, query, max_retries=3):
        start_time = time.time()
        retries = 0
        memories = []
        graph_relations = []
        while retries < max_retries:
            try:
                # search_with_graph 关图时 graph_relations=[]（main.py graph 钩子不执行，client data.get 默认 []）
                memories, graph_relations = self.client.search_with_graph(query, user_id=user_id, top_k=self.top_k, rerank=self.rerank)
                break
            except Exception as e:
                print(f"Search retry {retries+1}: {e}")
                retries += 1
                if retries >= max_retries:
                    logger.warning(f"Search failed after {max_retries} retries: {e}")
                    return [], [], 0
                time.sleep(1)

        search_time = time.time() - start_time
        print(f"[search] user={user_id} query={query[:40]}... time={search_time:.2f}s", flush=True)
        semantic_memories = []
        for m in memories:
            memory_text = m.get("memory", "")
            # timestamp 在 metadata 中
            timestamp = m.get("metadata", {}).get("timestamp", "")
            score = round(m.get("score", 0), 2)
            semantic_memories.append({
                "memory": memory_text,
                "timestamp": timestamp,
                "score": score,
            })
        return semantic_memories, graph_relations, search_time

    def answer_question(self, speaker_a_user_id, speaker_b_user_id, question, answer, category, reference_date="2023"):
        speaker_a_memories, speaker_a_graph, speaker_a_time = self.search_memory(speaker_a_user_id, question)
        speaker_b_memories, speaker_b_graph, speaker_b_time = self.search_memory(speaker_b_user_id, question)

        memories_text = format_memories(speaker_a_memories, speaker_b_memories)

        # --- 图记忆注入（照抄 v2 双 speaker 分段结构）---
        # ablation arm: GRAPH_INJECT_RELATIONS=false（默认），不追加
        # graph arm:    GRAPH_INJECT_RELATIONS=true，追加 graph_relations 段
        # 格式照抄 v2 search_locomo_graph.py:138 "source -- relationship -- destination"
        if os.environ.get("GRAPH_INJECT_RELATIONS", "false").lower() == "true":
            relations_parts = []
            if speaker_a_graph:
                lines = [f"{r.get('source','')} -- {r.get('relationship','')} -- {r.get('destination','')}" for r in speaker_a_graph]
                relations_parts.append(f"Relations for user {speaker_a_user_id.split('_')[0]}:\n\n" + "\n".join(lines))
            if speaker_b_graph:
                lines = [f"{r.get('source','')} -- {r.get('relationship','')} -- {r.get('destination','')}" for r in speaker_b_graph]
                relations_parts.append(f"Relations for user {speaker_b_user_id.split('_')[0]}:\n\n" + "\n".join(lines))
            if relations_parts:
                memories_text = memories_text + "\n\n" + "\n\n".join(relations_parts)

        user_prompt = self.answer_template.render(
            memories=memories_text,
            question=question,
            reference_date=reference_date,
        )

        t1 = time.time()
        model = os.getenv("ANSWER_MODEL", os.getenv("LLM_MODEL", "qwen-max-latest"))
        # 429 指数退避重试：proxy 全 key 限流时等 key 恢复，避免单次 429 崩溃整个 run。
        # 不改变 LLM 输入输出，成功后 response 与一次调用等价，不影响分数。
        # 15 次 / 上限 90s（总 ~16min）：抗 MiniMax 上游波动限流 + 多实验共存抢 key。
        response = None
        last_err = None
        for attempt in range(15):
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.0,
                    max_tokens=2000,
                    timeout=60,
                    extra_body=build_thinking_extra(model, enable=True),
                )
                break
            except Exception as e:
                last_err = e
                if "429" in str(e) or "rate limit" in str(e).lower():
                    wait = min(2 ** attempt * 5, 90)  # 5,10,20,40,80,90,90,...
                    print(f"[answer] 429 retry {attempt+1}/15, wait {wait}s: {e}", flush=True)
                    time.sleep(wait)
                else:
                    raise
        if response is None:
            raise last_err
        response_time = time.time() - t1

        # 剥离 <think> 标签后再给 judge（CLAUDE.md 规则 11，与 0716/0717 实验条件对齐）
        raw_response = extract_response_text(response)

        return (
            raw_response,
            speaker_a_memories,
            speaker_b_memories,
            speaker_a_time,
            speaker_b_time,
            response_time,
            speaker_a_graph,
            speaker_b_graph,
        )

    def process_data_file(self, file_path, max_workers=8):
        with open(file_path, "r") as f:
            data = json.load(f)

        self._results_lock = threading.Lock()

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            session_date = conversation.get("session_1_date_time", "")
            reference_date = _parse_session_date(session_date)

            def process_qa(question_item):
                question = question_item.get("question", "")
                answer = question_item.get("answer", "")
                category = question_item.get("category", -1)
                evidence = question_item.get("evidence", [])
                adversarial_answer = question_item.get("adversarial_answer", "")

                (
                    response,
                    speaker_a_memories,
                    speaker_b_memories,
                    speaker_a_time,
                    speaker_b_time,
                    response_time,
                    speaker_a_graph,
                    speaker_b_graph,
                ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category, reference_date=reference_date)

                return {
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "evidence": evidence,
                    "response": response,
                    "adversarial_answer": adversarial_answer,
                    "speaker_1_memories": speaker_a_memories,
                    "speaker_2_memories": speaker_b_memories,
                    "num_speaker_1_memories": len(speaker_a_memories),
                    "num_speaker_2_memories": len(speaker_b_memories),
                    "speaker_1_memory_time": speaker_a_time,
                    "speaker_2_memory_time": speaker_b_time,
                    "response_time": response_time,
                    "speaker_1_graph_relations": speaker_a_graph,
                    "speaker_2_graph_relations": speaker_b_graph,
                    "num_speaker_1_graph_relations": len(speaker_a_graph),
                    "num_speaker_2_graph_relations": len(speaker_b_graph),
                }

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_qa, q): q for q in qa}
                for future in tqdm(as_completed(futures), total=len(qa), desc=f"Conv {idx}", leave=False):
                    result = future.result()
                    with self._results_lock:
                        self.results[idx].append(result)

            # 每个对话结束后保存
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)

        # 最终保存
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {self.output_path}")

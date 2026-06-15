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

from utils.llm_client import build_thinking_extra, extract_response_text
from .client import NeatMemClient

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.),
   calculate the actual date based on the memory timestamp. For example, if a memory from
   4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example,
   convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
   timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse character
   names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Memories for user {{speaker_1_user_id}}:

{{speaker_1_memories}}

Memories for user {{speaker_2_user_id}}:

{{speaker_2_memories}}

Question: {{question}}

Answer:
"""


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
        while retries < max_retries:
            try:
                memories = self.client.search(query, user_id=user_id, top_k=self.top_k, rerank=self.rerank)
                break
            except Exception as e:
                print(f"Search retry {retries+1}: {e}")
                retries += 1
                if retries >= max_retries:
                    logger.warning(f"Search failed after {max_retries} retries: {e}")
                    return [], 0
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
        return semantic_memories, search_time

    def answer_question(self, speaker_a_user_id, speaker_b_user_id, question, answer, category):
        speaker_a_memories, speaker_a_time = self.search_memory(speaker_a_user_id, question)
        speaker_b_memories, speaker_b_time = self.search_memory(speaker_b_user_id, question)

        search_a = [f"{item['timestamp']}: {item['memory']}" for item in speaker_a_memories]
        search_b = [f"{item['timestamp']}: {item['memory']}" for item in speaker_b_memories]

        t1 = time.time()
        system_prompt = "You are an intelligent memory assistant. Answer the user's question based solely on the provided memories. Keep the answer under 5-6 words."
        user_prompt = (
            f"Memories for {speaker_a_user_id.split('_')[0]}:\n{json.dumps(search_a, indent=4)}\n\n"
            f"Memories for {speaker_b_user_id.split('_')[0]}:\n{json.dumps(search_b, indent=4)}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        response = self.openai_client.chat.completions.create(
            model=os.getenv("ANSWER_MODEL", os.getenv("LLM_MODEL", "qwen-max-latest")),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
            timeout=60,
            extra_body=build_thinking_extra(
                os.getenv("ANSWER_MODEL", os.getenv("LLM_MODEL", "qwen-max-latest")),
                enable=True,
            ),
        )
        response_time = time.time() - t1

        return (
            extract_response_text(response),
            speaker_a_memories,
            speaker_b_memories,
            speaker_a_time,
            speaker_b_time,
            response_time,
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
                ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

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

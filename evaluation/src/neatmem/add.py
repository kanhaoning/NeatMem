"""LOCOMO Ingestion — 将对话写入 NeatMem 记忆系统"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from tqdm import tqdm

from .client import NeatMemClient

load_dotenv()

# 与 mem0 评测一致的 custom_instructions
CUSTOM_INSTRUCTIONS = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""


class NeatMemADD:
    def __init__(self, data_path=None, batch_size=10):
        self.client = NeatMemClient()
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, messages, metadata, retries=3):
        for attempt in range(retries):
            try:
                self.client.add(
                    messages,
                    user_id=user_id,
                    metadata=metadata,
                    custom_instructions=CUSTOM_INSTRUCTIONS,
                )
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc):
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc, leave=False):
            batch_messages = messages[i : i + self.batch_size]
            self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp})

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        # 按 user_id 清空旧记忆
        self.client.delete_all(user_id=speaker_a_user_id)
        self.client.delete_all(user_id=speaker_b_user_id)

        # 获取 session keys（按时间排序）
        session_keys = sorted(
            [k for k in conversation if k.startswith("session_") and "date" not in k],
            key=lambda k: int(k.split("_")[1]),
        )

        for session_key in session_keys:
            date_time_key = session_key + "_date_time"
            timestamp = conversation.get(date_time_key, "")
            chats = conversation[session_key]

            # 构建 messages：speaker_a 视角 a=user, b=assistant
            messages = []
            messages_reverse = []
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")

            # 两个 speaker 各存一份
            thread_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_a_user_id, messages, timestamp, f"[{idx}] Speaker A"),
            )
            thread_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_b_user_id, messages_reverse, timestamp, f"[{idx}] Speaker B"),
            )
            thread_a.start()
            thread_b.start()
            thread_a.join()
            thread_b.join()

        print(f"[{idx}] {speaker_a}/{speaker_b} 写入完成")

    def process_all_conversations(self, max_workers=4):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path first.")
        # 不同对话的 user_id 互不干扰，可并发；对话内部仍串行保证去重正确
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_conversation, item, idx)
                       for idx, item in enumerate(self.data)]
            for future in futures:
                future.result()

"""注入日志（INJECTION_LOG_PATH）的 flag 锁与格式验证。

计划 docs/internal-notes/20261005-citation-feedback-memory-importance-plan.md
§5.6 第 1 步要求的单测锁：flag off 与历史行为逐字节一致（零副作用），
flag on 每题落一行（conversation_idx/question/category/memories[id,text,speaker,score]/answer）。
"""

import json
from collections import defaultdict
from types import SimpleNamespace

import pytest
from jinja2 import Template

from neatmem.evaluation.locomo import search as search_mod
from neatmem.evaluation.prompts import ANSWER_PROMPT

HITS = [
    {"id": "mem-1", "memory": "Caroline adopted a dog named Milo.", "score": 0.91, "metadata": {"timestamp": "2023-01-01"}},
]


class _StubClient:
    def __init__(self, hits=None, fail=False):
        self.hits = hits if hits is not None else HITS
        self.fail = fail

    def search_with_graph(self, query, user_id=None, top_k=10, rerank=None):
        if self.fail:
            raise RuntimeError("boom")
        return self.hits, []


def _stub_openai(text="<think>reasoning</think> final answer"):
    msg = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    resp = SimpleNamespace(choices=[choice])
    chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: resp))
    return SimpleNamespace(chat=chat)


def _make_search(tmp_path, client=None):
    s = search_mod.NeatMemSearch.__new__(search_mod.NeatMemSearch)
    s.top_k = 10
    s.rerank = None
    s.client = client or _StubClient()
    s.openai_client = _stub_openai()
    s.results = defaultdict(list)
    s.output_path = str(tmp_path / "results.json")
    s.answer_template = Template(ANSWER_PROMPT)
    return s


@pytest.fixture(autouse=True)
def _clear_flag(monkeypatch):
    monkeypatch.delenv("INJECTION_LOG_PATH", raising=False)
    monkeypatch.delenv("EXCLUDE_MEMORY_IDS_FILE", raising=False)
    search_mod._exclude_ids_cache.clear()


def test_flag_off_path_is_none():
    assert search_mod._injection_log_path() is None


def test_search_memory_default_unchanged(tmp_path):
    s = _make_search(tmp_path)
    semantic, graph, elapsed = s.search_memory("Caroline_0", "what pet?")
    assert [m["memory"] for m in semantic] == [HITS[0]["memory"]]
    assert graph == []


def test_raw_sink_success_and_failure(tmp_path):
    s = _make_search(tmp_path)
    sink = []
    s.search_memory("Caroline_0", "q", raw_sink=sink)
    assert sink == [HITS]

    failing = _make_search(tmp_path, client=_StubClient(fail=True))
    sink2 = []
    semantic, graph, elapsed = failing.search_memory("Caroline_0", "q", max_retries=2, raw_sink=sink2)
    assert semantic == [] and sink2 == [[]]


def test_answer_question_sink_filled(tmp_path):
    s = _make_search(tmp_path)
    sink = []
    out = s.answer_question("Caroline_0", "Melanie_0", "what pet?", "gold", 2, injection_sink=sink)
    assert out[0] == "final answer"  # think 已剥离
    assert {m["speaker"] for m in sink} == {"a", "b"}
    for m in sink:
        assert m["id"] == "mem-1"
        assert m["text"] == HITS[0]["memory"]
        assert m["score"] == 0.91


def test_answer_question_sink_none_no_side_effect(tmp_path):
    s = _make_search(tmp_path)
    out = s.answer_question("Caroline_0", "Melanie_0", "what pet?", "gold", 2)
    assert out[0] == "final answer"


def test_exclude_ids_filter(tmp_path, monkeypatch):
    # flag off：命中原样通过
    s = _make_search(tmp_path)
    semantic, _, _ = s.search_memory("Caroline_0", "q")
    assert len(semantic) == 1

    # flag on：排除指定 id，且 raw_sink 记录的是过滤后的实际注入集
    ids_file = tmp_path / "evicted.txt"
    ids_file.write_text("mem-1\n")
    monkeypatch.setenv("EXCLUDE_MEMORY_IDS_FILE", str(ids_file))
    sink = []
    semantic, _, _ = s.search_memory("Caroline_0", "q", raw_sink=sink)
    assert semantic == [] and sink == [[]]


def test_process_data_file_writes_log_only_when_flag_on(tmp_path, monkeypatch):
    data = [{
        "qa": [{"question": "what pet?", "answer": "a dog", "category": 2, "evidence": []}],
        "conversation": {"speaker_a": "Caroline", "speaker_b": "Melanie", "session_1_date_time": "2023-01-01"},
    }]
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps(data))

    # flag off：不产生日志文件
    log_path = tmp_path / "injections.jsonl"
    s = _make_search(tmp_path)
    s.process_data_file(str(data_file), max_workers=1)
    assert not log_path.exists()

    # flag on：每题一行，字段齐全，answer 已剥离 think
    monkeypatch.setenv("INJECTION_LOG_PATH", str(log_path))
    s2 = _make_search(tmp_path)
    s2.process_data_file(str(data_file), max_workers=1)
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["conversation_idx"] == 0
    assert rec["question"] == "what pet?"
    assert rec["category"] == 2
    assert rec["answer"] == "final answer"
    assert len(rec["memories"]) == 2  # 双 speaker 各搜一次
    assert all(set(m) == {"id", "text", "speaker", "score"} for m in rec["memories"])

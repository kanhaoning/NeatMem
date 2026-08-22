"""示例 prompt 文件与内置常量的逐字节相等性测试。

防止双份内容漂移：谁改了常量忘了重新生成示例（或反之），这里直接红。
dedup_en 不在此列（它是 0724 实验的 v7_en，主包无对应常量）。

2026-08-23: edit_en.txt / rewrite_en.txt 是运行时默认 prompt 的唯一出处
（代码零拷贝），不再是任何常量的镜像，故只做存在性检查；
editv3_en.txt 同理（实验变体，仅路径可达）。

运行:  cd <repo root> && python -m pytest tests/test_prompt_examples.py -q
"""

from importlib.resources import files

import pytest

from neatmem.prompts.extraction import ADDITIVE_EXTRACTION_PROMPT
from neatmem.memory_add import ACTION_DEDUP_PROMPT
from neatmem.rerank import _LISTWISE_PROMPT

MIRRORS = {
    "extraction_en.example.txt": ADDITIVE_EXTRACTION_PROMPT,
    "dedup_zh.example.txt": ACTION_DEDUP_PROMPT,
    "rerank_en.example.txt": _LISTWISE_PROMPT,
}

ALL_EXAMPLES = sorted(MIRRORS) + [
    "dedup_en.example.txt",
    "edit_en.txt",
    "editv3_en.txt",
    "rewrite_en.txt",
]


@pytest.mark.parametrize("fname,expected", sorted(MIRRORS.items()))
def test_example_mirrors_constant(fname, expected):
    actual = (files("neatmem.prompts") / "examples" / fname).read_text(encoding="utf-8")
    assert actual == expected, (
        f"{fname} 与内置常量不一致——改了常量请重新生成示例，改了示例请确认是否应同步常量"
    )


def test_all_examples_present():
    for fname in ALL_EXAMPLES:
        assert (files("neatmem.prompts") / "examples" / fname).is_file(), f"{fname} missing"

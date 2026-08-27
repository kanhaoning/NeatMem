"""Packaged prompt txt tests: existence, migration hashes, pairing table.

1. MIRRORS: example files that mirror an in-code constant (drift guard —
   change one without the other and this goes red).
2. MIGRATED_SHA256: the 2026-08-26 dedup prompt migration moved in-code
   constants and experiment prompts into packaged txt files. The pins below
   lock the migrated bytes; changing a prompt means deliberately changing
   its hash here.
     - dedup_listwise_en.txt            = v7en (ex dedup_en.example.txt)
     - dedup_listwise_strict_en.txt     = strict_update (0802 experiment)
     - dedup_listwise_multitarget_en.txt = mt_strict_v2 (0825 experiment)
     - dedup_pointwise_en.txt           = ex PAIRWISE_DETECTOR_PROMPT constant
3. Pairing table: every (detector, resolver) combo must resolve to an
   existing packaged file; a miss must fail, never fall back silently.

Run:  cd <repo root> && python -m pytest tests/test_prompt_examples.py -q
"""

import hashlib
from importlib.resources import files

import pytest

from neatmem.prompts.extraction import ADDITIVE_EXTRACTION_PROMPT
from neatmem.prompts.loader import dedup_prompt_default_file
from neatmem.rerank import _LISTWISE_PROMPT

MIRRORS = {
    "extraction_en.example.txt": ADDITIVE_EXTRACTION_PROMPT,
    "rerank_en.example.txt": _LISTWISE_PROMPT,
}

MIGRATED_SHA256 = {
    "dedup_listwise_en.txt":
        "9f9bc4fc9adc4017050ea9fdd3f23504cc3020544f1a205bd6ca011b74bc9d71",
    "dedup_listwise_strict_en.txt":
        "3659c207995010b57e1fa0ef02590967971387f448d09cf655700d2ffc6f992e",
    "dedup_listwise_multitarget_en.txt":
        "1b7d45b2d824d10ca22bae0e8e1f0d8ffcc79e22dcfcb83cf8cf43907b51a481",
    "dedup_pointwise_en.txt":
        "93f0c8fddbb4d4139ae600a248195181c2634457a677c1df3176ddfd469415dd",
}

ALL_EXAMPLES = sorted(MIRRORS) + sorted(MIGRATED_SHA256) + [
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


@pytest.mark.parametrize("fname,pinned", sorted(MIGRATED_SHA256.items()))
def test_migrated_prompt_sha256(fname, pinned):
    data = (files("neatmem.prompts") / "examples" / fname).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    assert actual == pinned, (
        f"{fname} hash changed ({actual[:12]}... != {pinned[:12]}...). "
        f"Changing a migrated prompt is a deliberate act: update the pin in "
        f"tests/test_prompt_examples.py and note it in the change log."
    )


def test_all_examples_present():
    for fname in ALL_EXAMPLES:
        assert (files("neatmem.prompts") / "examples" / fname).is_file(), f"{fname} missing"


@pytest.mark.parametrize("detector,resolver,expected", [
    ("listwise", "skip", "dedup_listwise_en.txt"),
    ("listwise", "replace", "dedup_listwise_strict_en.txt"),
    ("listwise", "rewrite", "dedup_listwise_strict_en.txt"),
    ("listwise", "edit", "dedup_listwise_strict_en.txt"),
    ("listwise_multitarget", "skip", "dedup_listwise_multitarget_en.txt"),
    ("listwise_multitarget", "rewrite", "dedup_listwise_multitarget_en.txt"),
    ("pointwise", "skip", "dedup_pointwise_en.txt"),
    ("pointwise", "rewrite", "dedup_pointwise_en.txt"),
])
def test_dedup_prompt_pairing(detector, resolver, expected):
    fname = dedup_prompt_default_file(detector, resolver)
    assert fname == expected
    assert (files("neatmem.prompts") / "examples" / fname).is_file()


def test_dedup_prompt_pairing_miss_fails():
    with pytest.raises(SystemExit):
        dedup_prompt_default_file("listwise", "nonexistent_resolver")

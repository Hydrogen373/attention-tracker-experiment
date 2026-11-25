"""Integration check against the real microsoft/BIPIA builder and data.

Skipped unless external/BIPIA (or $BIPIA_REPO) exists and its package imports.
Needs no GPU: only prompt construction is exercised.
"""
import json
import os
import sys
from pathlib import Path

import pytest

from tracker.datasets import BIPIADataset

REPO = Path(os.environ.get("BIPIA_REPO", "external/BIPIA"))


@pytest.fixture(scope="module")
def official_builder():
    if not (REPO / "benchmark").exists():
        pytest.skip(f"BIPIA repo not found at {REPO}")
    sys.path.insert(0, str(REPO.resolve()))
    try:
        from bipia.data import AutoPIABuilder
    except ImportError as exc:
        pytest.skip(f"BIPIA package not importable: {exc}")
    finally:
        sys.path.remove(str(REPO.resolve()))
    return AutoPIABuilder


def strip_attack(poisoned, attack, position):
    """Undo the official insert_end / insert_start / insert_middle."""
    if position == "end":
        assert poisoned.endswith("\n" + attack)
        return poisoned[: -len(attack) - 1]
    if position == "start":
        assert poisoned.startswith(attack + "\n")
        return poisoned[len(attack) + 1:]
    marker = "\n" + attack + "\n"
    assert marker in poisoned
    return poisoned.replace(marker, "", 1)


def source_contexts(task):
    with (REPO / "benchmark" / task / "test.jsonl").open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    return [("\n".join(r["context"]) if task == "code" else r["context"]) for r in rows]


@pytest.mark.parametrize("task", ["email", "table", "code"])
def test_official_pairs_match_source_contexts(official_builder, task):
    cases = BIPIADataset(REPO, max_contexts=0, max_variants_per_attack=1, max_pairs=60).load(task)
    sources = source_contexts(task)
    builder = official_builder.from_name(task)(seed=2023)
    assert len(cases) == 120
    for i in range(0, len(cases), 2):
        clean, attack = cases[i], cases[i + 1]
        assert clean.metadata["pair_id"] == attack.metadata["pair_id"]
        # Clean side is exactly the source context the attack was inserted into.
        assert clean.data_parts[0] in sources
        assert strip_attack(attack.data_parts[0], attack.metadata["attack_str"],
                            attack.metadata["position"]) == clean.data_parts[0]
        assert attack.metadata["attack_str"] not in clean.system
        # Same trusted instruction and user turn on both sides.
        assert clean.instruction == attack.instruction
        assert clean.user == attack.user
        assert attack.metadata["position"] in {"end", "start", "middle"}
        # Clean prompt equals the official template filled with the source context.
        system, _ = builder.construct_prompt(
            {**{k: v for k, v in attack.metadata.items()}, "context": clean.data_parts[0],
             "question": clean.metadata["user_question"], "code": "", "error": ""},
            require_system_prompt=True,
        )
        assert system == clean.system


def test_official_row_count_matches_full_benchmark(official_builder):
    """email: 50 contexts x 15 attack families x 5 variants x 3 positions."""
    cases = BIPIADataset(REPO, max_contexts=0, max_variants_per_attack=0, max_pairs=0).load("email")
    assert len(cases) == 2 * 50 * 75 * 3

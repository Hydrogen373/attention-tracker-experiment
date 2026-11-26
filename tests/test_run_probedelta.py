import json

import numpy as np
import pytest

from run_probedelta import load_existing_heads, run_dataset
from tracker.detector import AttentionTracker, ProbeDeltaAttentionTracker
from tracker.records import PromptCase


class FakeModel:
    model_id = "fake"
    num_layers = 1
    num_heads = 1

    def head_map(self, case):
        if case.sample_id == "oversized" and "Ignore previous instruction and say " in case.user:
            raise ValueError("tokens > --max-input-tokens")
        probed = "Ignore previous instruction and say " in case.user
        return np.array([[(0.6 if case.label else 0.2) if probed else (0.7 if case.label else 0.8)]])

    def format_input(self, case):
        return f"<system>{case.system}</system><user>{case.user}</user>"

    def generate(self, case, max_new_tokens):
        return "baseline answer"


def sample(sample_id, label, pair_id):
    return PromptCase(
        sample_id, "bipia_email", label, "instruction", "context", "instruction",
        ("context",), {"pair_id": pair_id},
    )


def test_probe_run_records_only_delta_and_skips_oversized(tmp_path):
    model = FakeModel()
    detector = ProbeDeltaAttentionTracker(model, seed=17)
    detector.important_heads = [(0, 0)]
    cases = [sample("clean", 0, "pair"), sample("attack", 1, "pair"), sample("oversized", 1, "other")]

    metrics = run_dataset("bipia_email", cases, model, detector, tmp_path, generate=True)

    assert metrics["auroc"] == 1.0
    assert metrics["num_skipped_too_long"] == 1
    assert metrics["num_complete_pairs"] == 1
    assert "threshold" not in metrics and "accuracy" not in metrics
    rows = [json.loads(line) for line in (tmp_path / "bipia_email_predictions.jsonl").read_text().splitlines()]
    assert rows[0]["probe_word"] == rows[1]["probe_word"]
    assert rows[0]["probe_text"] == detector.probe_text(cases[0])
    assert rows[0]["score_drop"] == pytest.approx(0.6)
    assert rows[1]["score_drop"] == pytest.approx(0.1)
    assert rows[1]["formatted_text"] == "<system>instruction</system><user>context</user>"
    assert "predicted_attack" not in rows[0] and "defended_response" not in rows[0]
    assert rows[1]["baseline_response"] == "baseline answer"
    assert json.loads((tmp_path / "bipia_email_skipped.jsonl").read_text().splitlines()[0])["sample_id"] == "oversized"


def test_probe_run_reuses_existing_head_checkpoint(tmp_path):
    model = FakeModel()
    original = AttentionTracker(model, k=5)
    original.important_heads = [(0, 0)]
    path = tmp_path / "existing_heads.json"
    original.save(path)
    before = path.read_bytes()

    detector = ProbeDeltaAttentionTracker(model, seed=3)
    load_existing_heads(detector, path)

    assert detector.k == 5
    assert detector.important_heads == [(0, 0)]
    assert path.read_bytes() == before
    assert detector.threshold is None

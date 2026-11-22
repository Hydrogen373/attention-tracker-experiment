import json

import numpy as np
import pytest

from run_probedelta import run_dataset
from tracker.detector import ProbeDeltaAttentionTracker
from tracker.records import PromptCase


class FakeModel:
    model_id = "fake"
    num_layers = 1
    num_heads = 1

    def head_map(self, case):
        if case.sample_id == "oversized" and case.user.endswith(ProbeDeltaAttentionTracker.PROBE_TEXT):
            raise ValueError("tokens > --max-input-tokens")
        probed = case.user.endswith(ProbeDeltaAttentionTracker.PROBE_TEXT)
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


def test_probe_run_records_delta_decision_and_skips_oversized(tmp_path):
    model = FakeModel()
    detector = ProbeDeltaAttentionTracker(model, threshold=0.3)
    detector.important_heads = [(0, 0)]
    cases = [sample("clean", 0, "pair"), sample("attack", 1, "pair"), sample("oversized", 1, "other")]

    metrics = run_dataset("bipia_email", cases, model, detector, tmp_path, generate=True)

    assert (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]) == (1, 0, 1, 0)
    assert metrics["num_skipped_too_long"] == 1
    assert metrics["paired_clean_pass_attack_block_rate"] == 1.0
    rows = [json.loads(line) for line in (tmp_path / "bipia_email_predictions.jsonl").read_text().splitlines()]
    assert [row["predicted_attack"] for row in rows] == [False, True]
    assert rows[0]["score_drop"] == pytest.approx(0.6)
    assert rows[1]["score_drop"] == pytest.approx(0.1)
    assert rows[1]["formatted_text"] == "<system>instruction</system><user>context</user>"
    assert rows[0]["defended_response"] == "baseline answer"
    assert rows[1]["defended_response"] == "[BLOCKED BY PROBE DELTA TRACKER]"
    assert rows[1]["baseline_response"] == "baseline answer"
    assert json.loads((tmp_path / "bipia_email_skipped.jsonl").read_text().splitlines()[0])["sample_id"] == "oversized"

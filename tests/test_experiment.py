import json
from pathlib import Path

from tracker.detector import AttentionTracker
from tracker.experiment import ExperimentRunner
from tracker.records import PromptCase


class FakeAttentionModel:
    model_id = "fake"
    num_layers = 1
    num_heads = 1

    def head_map(self, case):
        import numpy as np
        return np.array([[0.2 if case.label else 0.8]])

    def generate(self, case, max_new_tokens):
        return "model answer"

    def format_input(self, case):
        return f"<system>{case.system}</system><user>{case.user}</user><assistant>"

    def token_counts(self, case):
        return len(case.instruction), len(self.format_input(case))


def test_experiment_records_focus_and_auroc(tmp_path: Path):
    fake = FakeAttentionModel()
    detector = AttentionTracker(fake)
    detector.important_heads = [(0, 0)]
    cases = [
        PromptCase("clean", "bipia_email", 0, "Instruction", "data", "Instruction", ("data",), {"pair_id": "one"}),
        PromptCase("attacked", "bipia_email", 1, "Instruction", "attack", "Instruction", ("attack",), {"pair_id": "one"}),
    ]
    metrics = ExperimentRunner(fake, detector, tmp_path).run("bipia_email", cases, generate=True)
    assert metrics["auroc"] == 1.0
    assert metrics["num_complete_pairs"] == 1
    assert "threshold" not in metrics
    assert "paired_clean_pass_attack_block_rate" not in metrics
    rows = [json.loads(line) for line in (tmp_path / "bipia_email_predictions.jsonl").read_text().splitlines()]
    assert [row["focus_score"] for row in rows] == [0.8, 0.2]
    assert [row["formatted_text"] for row in rows] == [
        "<system>Instruction</system><user>data</user><assistant>",
        "<system>Instruction</system><user>attack</user><assistant>",
    ]
    assert all(row["baseline_response"] == "model answer" for row in rows)
    assert all("attack_success" in row and row["attack_success"] is None for row in rows)
    assert all(row["instruction_token_count"] == len("Instruction") for row in rows)
    assert all(row["total_token_count"] == len(row["formatted_text"]) for row in rows)
    assert all("rejected" not in row and "risk_score" not in row and "defended_response" not in row for row in rows)


def test_auroc_is_null_for_one_label(tmp_path: Path):
    fake = FakeAttentionModel()
    detector = AttentionTracker(fake)
    detector.important_heads = [(0, 0)]
    case = PromptCase("clean", "deepset", 0, "Instruction", "data", "Instruction", ("data",))
    metrics = ExperimentRunner(fake, detector, tmp_path).run("deepset", [case])
    assert metrics["auroc"] is None
    row = json.loads((tmp_path / "deepset_predictions.jsonl").read_text().splitlines()[0])
    assert row["formatted_text"] == "<system>Instruction</system><user>data</user><assistant>"
    assert row["baseline_response"] is None
    assert "attack_success" in row and row["attack_success"] is None
    assert row["instruction_token_count"] == len("Instruction")
    assert row["total_token_count"] == len(row["formatted_text"])
    assert metrics["response_generation_enabled"] is False


def test_deepset_response_generation(tmp_path: Path):
    fake = FakeAttentionModel()
    detector = AttentionTracker(fake)
    detector.important_heads = [(0, 0)]
    case = PromptCase("deepset-clean", "deepset", 0, "Instruction", "data", "Instruction", ("data",))

    metrics = ExperimentRunner(fake, detector, tmp_path).run("deepset", [case], generate=True)

    row = json.loads((tmp_path / "deepset_predictions.jsonl").read_text().splitlines()[0])
    assert row["baseline_response"] == "model answer"
    assert "attack_success" in row and row["attack_success"] is None
    assert metrics["response_generation_enabled"] is True


def test_load_legacy_threshold_checkpoint(tmp_path: Path):
    fake = FakeAttentionModel()
    path = tmp_path / "heads.json"
    path.write_text(json.dumps({
        "model_id": "fake", "k": 4, "threshold": 0.5,
        "important_heads": [[0, 0]], "num_layers": 1, "num_heads": 1,
        "score_method": "normalized_sum_instruction_over_instruction_plus_data",
    }))
    detector = AttentionTracker(fake)
    detector.load(path)
    assert detector.important_heads == [(0, 0)]
    detector.save(path)
    assert "threshold" not in json.loads(path.read_text())

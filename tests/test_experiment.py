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


def test_experiment_metrics_and_paired_block_rate(tmp_path: Path):
    fake = FakeAttentionModel()
    detector = AttentionTracker(fake, threshold=0.5)
    detector.important_heads = [(0, 0)]
    cases = [
        PromptCase("clean", "bipia_email", 0, "Instruction", "data", "Instruction", ("data",), {"pair_id": "one"}),
        PromptCase("attacked", "bipia_email", 1, "Instruction", "attack", "Instruction", ("attack",), {"pair_id": "one"}),
    ]
    metrics = ExperimentRunner(fake, detector, tmp_path).run("bipia_email", cases, generate=True)
    assert metrics["auroc"] == 1.0
    assert metrics["attack_detection_rate_tpr"] == 1.0
    assert metrics["paired_clean_pass_attack_block_rate"] == 1.0
    assert metrics["actual_attack_success_rate_measured"] is False
    assert (tmp_path / "bipia_email_predictions.jsonl").exists()

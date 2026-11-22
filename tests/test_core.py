import numpy as np
import pytest

from tracker.detector import AttentionTracker, ProbeDeltaAttentionTracker
from tracker.datasets import SyntheticHeadDataset
from tracker.records import PromptCase


def test_head_selection_matches_paper_equation():
    normal = np.array([[[0.75, 0.6]], [[0.77, 0.1]], [[0.76, 0.4]]])
    attack = np.array([[[0.03, 0.1]], [[0.04, 0.8]], [[0.02, 0.4]]])
    heads, candidate = AttentionTracker.select_heads(normal, attack, k=4)
    assert heads == [(0, 0)]
    assert candidate[0, 0] > 0
    assert candidate[0, 1] < 0


def test_synthetic_does_not_use_benchmarks():
    cases = SyntheticHeadDataset().load_pairs(30)
    assert len(cases) == 30
    for clean, attacked in cases:
        assert clean.label == 0 and attacked.label == 1
        assert clean.dataset == attacked.dataset == "synthetic"
        assert "Ignore previous instruction" not in clean.user
        assert "Ignore previous instruction" in attacked.user


def test_head_selection_failure_on_wrong_shape():
    with pytest.raises(ValueError):
        AttentionTracker.select_heads(np.zeros((2, 3)), np.zeros((2, 3)), 4)


class ProbeModel:
    model_id = "probe-fake"
    num_layers = 1
    num_heads = 1

    def __init__(self):
        self.seen = []

    def head_map(self, case):
        self.seen.append(case)
        probed = case.user.endswith(ProbeDeltaAttentionTracker.PROBE_TEXT)
        value = (0.6 if case.label else 0.2) if probed else (0.7 if case.label else 0.8)
        return np.array([[value]])


def probe_case(label):
    return PromptCase(
        f"case-{label}", "sample", label, "System instruction", "User data",
        "System instruction", ("User data",), {"source": "test"},
    )


def test_probe_drop_appends_only_user_text_and_preserves_case():
    model = ProbeModel()
    tracker = ProbeDeltaAttentionTracker(model, threshold=0.3)
    tracker.important_heads = [(0, 0)]
    case = probe_case(1)

    drop, attacked = tracker.predict(case)

    assert drop == pytest.approx(0.1)
    assert attacked is True
    assert case.user == "User data"
    assert case.data_parts == ("User data",)
    assert model.seen[0] is case
    probed = model.seen[1]
    assert probed.system == case.system
    assert probed.instruction == case.instruction
    assert probed.user == f"User data\n\n{tracker.PROBE_TEXT}"
    assert probed.data_parts == ("User data", tracker.PROBE_TEXT)
    assert probed.metadata == case.metadata


def test_probe_calibration_and_strict_boundary():
    tracker = ProbeDeltaAttentionTracker(ProbeModel())
    tracker.important_heads = [(0, 0)]
    with pytest.raises(RuntimeError, match="threshold"):
        tracker.predict(probe_case(0))
    assert tracker.calibrate_threshold([probe_case(0), probe_case(1)]) == pytest.approx(0.35)
    assert tracker.predict(probe_case(0))[1] is False
    assert tracker.predict(probe_case(1))[1] is True
    tracker.threshold = tracker.score_drop(probe_case(1))
    assert tracker.predict(probe_case(1))[1] is False


def test_probe_calibration_tie_prefers_no_false_positives():
    tracker = ProbeDeltaAttentionTracker(ProbeModel())
    tracker.important_heads = [(0, 0)]
    tracker.score_drop = lambda case: 0.0
    threshold = tracker.calibrate_threshold([probe_case(0), probe_case(1)])
    assert threshold < 0.0
    with pytest.raises(ValueError, match="both clean"):
        tracker.calibrate_threshold([probe_case(0)])


def test_probe_checkpoint_round_trip_and_rejects_base_checkpoint(tmp_path):
    tracker = ProbeDeltaAttentionTracker(ProbeModel(), threshold=0.25)
    tracker.important_heads = [(0, 0)]
    path = tmp_path / "probe.json"
    tracker.save(path)

    restored = ProbeDeltaAttentionTracker(ProbeModel())
    restored.load(path)
    assert restored.important_heads == [(0, 0)]
    assert restored.threshold == 0.25
    assert restored.predict(probe_case(1))[1] is True

    base_path = tmp_path / "base.json"
    AttentionTracker(ProbeModel()).save(base_path)
    with pytest.raises(ValueError, match="incompatible probe"):
        restored.load(base_path)


def test_probe_propagates_model_input_length_error():
    class LimitedModel(ProbeModel):
        def head_map(self, case):
            if case.user.endswith(ProbeDeltaAttentionTracker.PROBE_TEXT):
                raise ValueError("tokens > --max-input-tokens")
            return super().head_map(case)

    tracker = ProbeDeltaAttentionTracker(LimitedModel(), threshold=0.3)
    tracker.important_heads = [(0, 0)]
    with pytest.raises(ValueError, match="tokens >"):
        tracker.predict(probe_case(0))

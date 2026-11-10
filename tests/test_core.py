import numpy as np

from tracker.detector import AttentionTracker
from tracker.datasets import SyntheticHeadDataset


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
    import pytest
    with pytest.raises(ValueError):
        AttentionTracker.select_heads(np.zeros((2, 3)), np.zeros((2, 3)), 4)

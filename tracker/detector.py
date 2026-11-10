"""Head selection and Focus Score, following Attention Tracker (NAACL 2025)."""
import json
from pathlib import Path

import numpy as np

from .records import PromptCase


class AttentionTracker:
    """One set of selected heads and one decision threshold for every benchmark."""

    def __init__(self, model, k: float = 4.0, threshold: float | None = None):
        self.model = model
        self.k = k
        self.threshold = threshold
        self.important_heads: list[tuple[int, int]] = []

    @staticmethod
    def select_heads(normal_maps: np.ndarray, attack_maps: np.ndarray, k: float = 4.0):
        """Both inputs: (samples, layers, heads), computed on *independent* data."""
        if normal_maps.ndim != 3 or attack_maps.ndim != 3:
            raise ValueError("Head maps must have shape (N, layers, heads)")
        if normal_maps.shape[1:] != attack_maps.shape[1:]:
            raise ValueError("Normal and attack head-map shapes differ")
        if min(len(normal_maps), len(attack_maps)) < 2:
            raise ValueError("At least 2 normal and 2 attack calibration examples are required")

        normal_mean = normal_maps.mean(axis=0)
        attack_mean = attack_maps.mean(axis=0)
        normal_std = normal_maps.std(axis=0, ddof=0)
        attack_std = attack_maps.std(axis=0, ddof=0)

        # Equation 1 and 2: select heads with conservatively separated distributions.
        candidates = normal_mean - attack_mean - k * (normal_std + attack_std)
        selected = [tuple(map(int, location)) for location in np.argwhere(candidates > 0)]
        return selected, candidates

    def fit_heads(self, pairs: list[tuple[PromptCase, PromptCase]]):
        if not pairs:
            raise ValueError("No synthetic head selection pairs supplied")
        normal, attacked = [], []
        for clean_case, attack_case in pairs:
            normal.append(self.model.head_map(clean_case))
            attacked.append(self.model.head_map(attack_case))

        self.important_heads, candidate_scores = self.select_heads(
            np.stack(normal), np.stack(attacked), self.k
        )
        if not self.important_heads:
            raise RuntimeError(
                f"k={self.k} selected zero heads. Inspect synthetic data and try --k 3 "
                "or --k 2 explicitly; no silent fallback is performed."
            )
        return candidate_scores

    def focus_score(self, case: PromptCase) -> float:
        if not self.important_heads:
            raise RuntimeError("Head selection must be completed or loaded first")
        per_head = self.model.head_map(case)
        return float(np.mean([per_head[layer, head] for layer, head in self.important_heads]))

    def calibrate_benign(self, benign_cases: list[PromptCase], target_fpr: float = 0.05):
        """An empirical FPR operating point; NOT trained on test/attack labels."""
        if not 0 < target_fpr < 1:
            raise ValueError("target_fpr must be between 0 and 1")
        if len(benign_cases) < 2:
            raise ValueError("At least two benign calibration cases are required")
        scores = []
        for case in benign_cases:
            try:
                scores.append(self.focus_score(case))
            except ValueError as exc:
                if "tokens >" not in str(exc):
                    raise
                print(f"[calibration] Skipped oversized example {case.sample_id}")
        if len(scores) < 2:
            raise RuntimeError("Insufficient scoreable benign examples to calibrate threshold")
        print(f"[calibration] Scored {len(scores)}/{len(benign_cases)} benign examples")
        self.threshold = float(np.quantile(scores, target_fpr))
        return self.threshold

    def predict(self, case: PromptCase):
        if self.threshold is None:
            raise RuntimeError("No threshold defined; calibrate or supply --threshold")
        score = self.focus_score(case)
        return score, score < self.threshold  # lower FS -> higher injection risk

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "model_id": self.model.model_id,
            "k": self.k,
            "threshold": self.threshold,
            "important_heads": self.important_heads,
            "num_layers": self.model.num_layers,
            "num_heads": self.model.num_heads,
            "score_method": "normalized_sum_instruction_over_instruction_plus_data",
            "head_selection_source": "synthetic_30_pairs",
        }, indent=2), encoding="utf-8")

    def load(self, path: Path):
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["model_id"] != self.model.model_id:
            raise ValueError("Detector checkpoint was created for a different model")
        if state["num_layers"] != self.model.num_layers or state["num_heads"] != self.model.num_heads:
            raise ValueError("Attention dimensions do not match the saved head selection")
        if state.get("score_method") != "normalized_sum_instruction_over_instruction_plus_data":
            raise ValueError("Saved detector uses an incompatible Focus Score definition")
        self.k = float(state["k"])
        self.threshold = state["threshold"]
        self.important_heads = [tuple(pair) for pair in state["important_heads"]]
        if not self.important_heads:
            raise ValueError("Saved detector contains no heads")

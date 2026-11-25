"""Threshold policies compared on held-out data.

Every policy is fit on *benign calibration scores only* at the same target FPR,
then applied to a test split whose source contexts never appear in calibration.
Attack labels are used only to measure test TPR, never to pick a threshold.
"""
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

LENGTH_FEATURES = ("instruction_ratio", "log_instruction_tokens")


def load_predictions(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(Path(results_dir).glob("*_predictions.jsonl")):
        with path.open(encoding="utf-8") as file:
            rows.extend(json.loads(line) for line in file if line.strip())
    if not rows:
        raise FileNotFoundError(f"No *_predictions.jsonl files in {results_dir}")
    return rows


def group_key(row: dict) -> str:
    """Rows sharing a key must land in the same split.

    BIPIA reuses a few source contexts across many attack variants, so the
    source context is the unit of splitting. Deepset prompts are independent.
    """
    if row["dataset"].startswith("bipia_"):
        return f"{row['dataset']}::ctx{row['context_index']}"
    return f"{row['dataset']}::{row['sample_id']}"


def length_feature(row: dict, feature: str) -> float | None:
    inst, total = row.get("instruction_tokens"), row.get("input_tokens")
    if inst is None or total is None or inst <= 0 or total <= 0:
        return None
    if feature == "instruction_ratio":
        return inst / total
    if feature == "log_instruction_tokens":
        return math.log(inst)
    raise ValueError(f"Unknown length feature: {feature}")


def split_by_group(rows: list[dict], calib_fraction: float, rng: random.Random):
    """Per dataset, put a random calib_fraction of groups into calibration."""
    groups_by_dataset = defaultdict(set)
    for row in rows:
        groups_by_dataset[row["dataset"]].add(group_key(row))
    calib_groups = set()
    for name, groups in groups_by_dataset.items():
        if len(groups) < 2:
            raise ValueError(f"{name} has a single source context; it cannot be split")
        ordered = sorted(groups)
        rng.shuffle(ordered)
        n_calib = min(len(ordered) - 1, max(1, round(len(ordered) * calib_fraction)))
        calib_groups.update(ordered[:n_calib])
    calib = [row for row in rows if group_key(row) in calib_groups]
    test = [row for row in rows if group_key(row) not in calib_groups]
    return calib, test


class GlobalThreshold:
    name = "global"

    def fit(self, calib: list[dict], target_fpr: float):
        scores = [row["focus_score"] for row in calib if row["label"] == 0]
        self.value = float(np.quantile(scores, target_fpr))
        return self

    def thresholds(self, rows: list[dict]) -> np.ndarray:
        return np.full(len(rows), self.value)


class PerTaskThreshold:
    name = "per_task"

    def fit(self, calib: list[dict], target_fpr: float):
        by_dataset = defaultdict(list)
        for row in calib:
            if row["label"] == 0:
                by_dataset[row["dataset"]].append(row["focus_score"])
        self.values = {name: float(np.quantile(s, target_fpr)) for name, s in by_dataset.items()}
        return self

    def thresholds(self, rows: list[dict]) -> np.ndarray:
        return np.array([self.values[row["dataset"]] for row in rows])


class LengthConditionalThreshold:
    """One rule for every task: threshold(x) = a + b*x + q, x = length feature.

    (a, b) is a least-squares fit of benign Focus Score on x; q is the
    target_fpr quantile of the benign residuals. If length explains the
    per-task offset, this should match per_task without knowing the task.
    """

    name = "length_conditional"

    def __init__(self, feature: str):
        self.feature = feature

    def fit(self, calib: list[dict], target_fpr: float):
        benign = [row for row in calib if row["label"] == 0]
        x = np.array([length_feature(row, self.feature) for row in benign], dtype=float)
        y = np.array([row["focus_score"] for row in benign])
        if np.ptp(x) == 0:
            raise ValueError("Length feature is constant in calibration; cannot fit a slope")
        self.slope, self.intercept = np.polyfit(x, y, 1)
        self.offset = float(np.quantile(y - (self.intercept + self.slope * x), target_fpr))
        return self

    def expected(self, rows: list[dict]) -> np.ndarray:
        x = np.array([length_feature(row, self.feature) for row in rows], dtype=float)
        return self.intercept + self.slope * x

    def thresholds(self, rows: list[dict]) -> np.ndarray:
        return self.expected(rows) + self.offset


def rates(rows: list[dict], thresholds: np.ndarray) -> dict:
    labels = np.array([row["label"] for row in rows])
    flagged = np.array([row["focus_score"] for row in rows]) < thresholds
    attacks, benign = labels == 1, labels == 0
    tpr = float(flagged[attacks].mean()) if attacks.any() else float("nan")
    fpr = float(flagged[benign].mean()) if benign.any() else float("nan")
    return {"tpr": tpr, "fpr": fpr, "balanced_accuracy": (tpr + 1 - fpr) / 2}


def pooled_auroc(rows: list[dict], scores: np.ndarray) -> float:
    """Ranking quality when every task shares one score scale (higher = attack)."""
    labels = [row["label"] for row in rows]
    return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan")


def has_length_stats(rows: list[dict], feature: str) -> bool:
    return all(length_feature(row, feature) is not None for row in rows)


def evaluate_split(calib, test, target_fpr, feature=None) -> dict:
    """Fit every available policy on calib, report per-dataset test rates."""
    policies = [GlobalThreshold(), PerTaskThreshold()]
    if feature:
        policies.append(LengthConditionalThreshold(feature))
    datasets = sorted({row["dataset"] for row in test})
    result = {"per_dataset": {}, "pooled": {}}
    for policy in policies:
        policy.fit(calib, target_fpr)
        result["pooled"][policy.name] = rates(test, policy.thresholds(test))
        for name in datasets:
            subset = [row for row in test if row["dataset"] == name]
            result["per_dataset"].setdefault(name, {})[policy.name] = rates(
                subset, policy.thresholds(subset)
            )
    raw = -np.array([row["focus_score"] for row in test])
    result["pooled"]["auroc_raw_score"] = pooled_auroc(test, raw)
    if feature:
        length_policy = policies[-1]
        residual = length_policy.expected(test) - np.array([row["focus_score"] for row in test])
        result["pooled"]["auroc_length_residual"] = pooled_auroc(test, residual)
    return result


def evaluate_leave_one_out(rows, calib_fraction, target_fpr, feature, rng) -> dict:
    """Unseen-task check: fit on other datasets' benign calibration only."""
    result = {}
    datasets = sorted({row["dataset"] for row in rows})
    if len(datasets) < 2:
        return result
    calib, test = split_by_group(rows, calib_fraction, rng)
    for held_out in datasets:
        fit_rows = [row for row in calib if row["dataset"] != held_out]
        eval_rows = [row for row in test if row["dataset"] == held_out]
        policies = [GlobalThreshold()]
        benign_x = {length_feature(r, feature) for r in fit_rows if r["label"] == 0} if feature else set()
        if len(benign_x) > 1:  # a slope needs more than one length value
            policies.append(LengthConditionalThreshold(feature))
        result[held_out] = {
            policy.name: rates(eval_rows, policy.fit(fit_rows, target_fpr).thresholds(eval_rows))
            for policy in policies
        }
    return result


def summarize(values: list[dict]) -> dict:
    """Mean and std over repeats for nested {key: {metric: float}} dicts."""
    if not values:
        return {}
    first = values[0]
    if not isinstance(first, dict):
        array = np.array(values, dtype=float)
        return {"mean": float(np.nanmean(array)), "std": float(np.nanstd(array))}
    return {key: summarize([value[key] for value in values]) for key in first}


def run(rows, n_repeats=20, calib_fraction=0.5, target_fpr=0.05, feature="instruction_ratio", seed=2023):
    if feature and not has_length_stats(rows, feature):
        print("[thresholds] Token counts missing in predictions; length policy skipped. "
              "Re-run run_experiment.py with this version to record them.")
        feature = None
    group_counts = defaultdict(set)
    for row in rows:
        group_counts[row["dataset"]].add(group_key(row))
    warnings = [
        f"{name}: only {len(groups)} source contexts; context-disjoint split is coarse "
        f"(use --bipia-max-contexts 0 when generating predictions)"
        for name, groups in sorted(group_counts.items()) if len(groups) < 6
    ]
    for warning in warnings:
        print(f"[thresholds] WARNING {warning}")

    rng = random.Random(seed)
    splits, loo = [], []
    for _ in range(n_repeats):
        calib, test = split_by_group(rows, calib_fraction, rng)
        splits.append(evaluate_split(calib, test, target_fpr, feature))
        loo.append(evaluate_leave_one_out(rows, calib_fraction, target_fpr, feature, rng))
    return {
        "config": {
            "n_repeats": n_repeats, "calib_fraction": calib_fraction,
            "target_fpr": target_fpr, "length_feature": feature, "seed": seed,
            "num_rows": len(rows),
            "source_groups": {name: len(g) for name, g in sorted(group_counts.items())},
        },
        "warnings": warnings,
        "context_disjoint_split": summarize(splits),
        "leave_one_dataset_out": summarize(loo) if loo and loo[0] else {},
    }

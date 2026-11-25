import json
import random

import numpy as np

from tracker.thresholds import group_key, load_predictions, run, split_by_group


def synthetic_rows(seed=0):
    """Two tasks whose benign baseline differs only through instruction ratio.

    Scores follow 0.3 + 0.8 * ratio for benign and drop by 0.25 for attacks,
    so short-instruction prompts sit lower overall.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for name, ratio in [("deepset", 0.05), ("bipia_code", 0.6)]:
        for context in range(20):
            for variant in range(3):
                inst = 10 if name == "deepset" else 300
                total = int(inst / ratio)
                for label in (0, 1):
                    score = 0.3 + 0.8 * ratio - 0.25 * label + rng.normal(0, 0.03)
                    rows.append({
                        "sample_id": f"{name}-{context}-{variant}-{label}",
                        "dataset": name, "label": label, "focus_score": float(score),
                        "context_index": context, "pair_id": f"{context}-{variant}",
                        "instruction_tokens": inst, "input_tokens": total,
                    })
    return rows


def test_split_never_shares_a_bipia_source_context():
    rows = synthetic_rows()
    for row in rows:
        row["dataset"] = "bipia_" + row["dataset"].removeprefix("bipia_")
    calib, test = split_by_group(rows, 0.5, random.Random(1))
    assert calib and test
    assert not {group_key(r) for r in calib} & {group_key(r) for r in test}
    assert {r["dataset"] for r in calib} == {r["dataset"] for r in test}


def test_length_rule_recovers_per_task_offsets_without_task_labels():
    summary = run(synthetic_rows(), n_repeats=5, target_fpr=0.05)
    per_dataset = summary["context_disjoint_split"]["per_dataset"]

    # A single pooled threshold sits between the two baselines: it misses
    # every deepset-style prompt or flags none of the code-style attacks.
    assert per_dataset["bipia_code"]["global"]["tpr"]["mean"] < 0.2
    for name in per_dataset:
        assert per_dataset[name]["per_task"]["tpr"]["mean"] > 0.95
        assert per_dataset[name]["length_conditional"]["tpr"]["mean"] > 0.95
        assert per_dataset[name]["length_conditional"]["fpr"]["mean"] < 0.2

    pooled = summary["context_disjoint_split"]["pooled"]
    assert pooled["auroc_length_residual"]["mean"] > pooled["auroc_raw_score"]["mean"]

    # With only two tasks, x is constant inside the remaining one, so the
    # leave-one-out length fit is skipped by design; global still reports.
    assert set(summary["leave_one_dataset_out"]) in ({"deepset", "bipia_code"}, set())


def test_missing_token_counts_skip_length_policy(capsys):
    rows = synthetic_rows()
    for row in rows:
        row.pop("instruction_tokens")
    summary = run(rows, n_repeats=2)
    assert summary["config"]["length_feature"] is None
    assert "length_conditional" not in summary["context_disjoint_split"]["pooled"]
    assert "Token counts missing" in capsys.readouterr().out


def test_load_predictions_reads_all_jsonl(tmp_path):
    rows = synthetic_rows()[:4]
    (tmp_path / "deepset_predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    assert load_predictions(tmp_path) == rows

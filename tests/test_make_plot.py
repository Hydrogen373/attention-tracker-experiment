import json

import pytest

import make_plot
from make_plot import DATASETS, load_points, save_plots


def prediction(dataset, label):
    instruction_tokens = 11 if dataset == "deepset" else 35
    total_tokens = 100 if dataset == "deepset" else 230
    return {
        "sample_id": f"{dataset}-{label}", "dataset": dataset,
        "label": label, "focus_score": 0.7 if label == 0 else 0.3,
        "score_drop": 0.4 if label == 0 else -0.2,
        "instruction_token_count": instruction_tokens,
        "total_token_count": total_tokens,
    }


def test_load_points_and_save_both_plots(tmp_path):
    for dataset in DATASETS:
        path = tmp_path / f"{dataset}_predictions.jsonl"
        path.write_text(
            "\n".join(json.dumps(prediction(dataset, label)) for label in (0, 1)) + "\n",
            encoding="utf-8",
        )
    points = load_points(tmp_path)
    assert len(points) == 8
    for dataset, label, instruction_tokens, ratio, score in points:
        row = prediction(dataset, label)
        assert instruction_tokens == row["instruction_token_count"]
        assert ratio == row["instruction_token_count"] / row["total_token_count"]
        assert score == row["focus_score"]

    save_plots(points, tmp_path)
    for filename in (
        "instruction_token_count_vs_focus.png",
        "instruction_token_ratio_vs_focus.png",
    ):
        path = tmp_path / filename
        assert path.is_file() and path.stat().st_size > 0


@pytest.mark.parametrize("with_probe_results", [False, True])
def test_main_saves_available_probe_plots(tmp_path, monkeypatch, with_probe_results):
    for dataset in DATASETS:
        (tmp_path / f"{dataset}_predictions.jsonl").write_text(
            json.dumps(prediction(dataset, 0)) + "\n", encoding="utf-8"
        )

    probe_results = tmp_path / "probedelta"
    if with_probe_results:
        probe_results.mkdir()
        (probe_results / "deepset_predictions.jsonl").write_text(
            "\n".join(json.dumps(prediction("deepset", label)) for label in (0, 1)) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(make_plot, "RESULTS", tmp_path)

    make_plot.main()

    for stem in ("instruction_token_count", "instruction_token_ratio"):
        assert (tmp_path / f"{stem}_vs_focus.png").is_file()
        assert (probe_results / f"{stem}_vs_score_drop.png").is_file() == with_probe_results
    if with_probe_results:
        points = load_points(
            probe_results, score_field="score_drop", datasets=("deepset",)
        )
        assert [point[4] for point in points] == [0.4, -0.2]


@pytest.mark.parametrize("missing_field", ["instruction_token_count", "total_token_count"])
def test_load_points_rejects_legacy_rows(tmp_path, missing_field):
    row = prediction("deepset", 0)
    del row[missing_field]
    path = tmp_path / "deepset_predictions.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=f"Missing {missing_field}.*deepset_predictions.jsonl:1"):
        load_points(tmp_path, datasets=("deepset",))


@pytest.mark.parametrize("instruction,total", [(0, 100), (11, 0), (101, 100), (True, 100), (11.0, 100)])
def test_load_points_rejects_invalid_token_counts(tmp_path, instruction, total):
    row = prediction("deepset", 0)
    row["instruction_token_count"] = instruction
    row["total_token_count"] = total
    path = tmp_path / "deepset_predictions.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="deepset_predictions.jsonl:1"):
        load_points(tmp_path, datasets=("deepset",))

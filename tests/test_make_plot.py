import json
import sys
from types import SimpleNamespace

import pytest
import torch

import make_plot
from make_plot import DATASETS, SavedPromptEncoder, case_from_prediction, load_points, save_plots


class CharacterTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.arange(len(text)).unsqueeze(0),
            "offset_mapping": torch.tensor(
                [(i, i + 1) for i in range(len(text))]
            ).unsqueeze(0),
        }


def prediction(dataset, label):
    if dataset == "deepset":
        system, user = "Say xxxxxx", "Some user data"
        extra = {"text": user}
    else:
        system, user = "Follow task.\nContext:\nexternal text", "Find answer"
        extra = {"context": "external text", "user_question": user}
    rendered = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return {
        "sample_id": f"{dataset}-{label}", "dataset": dataset,
        "label": label, "focus_score": 0.7 if label == 0 else 0.3,
        "score_drop": 0.4 if label == 0 else -0.2,
        "formatted_text": rendered, **extra,
    }


def test_saved_prompt_uses_experiment_instruction_spans():
    encoder = SavedPromptEncoder(CharacterTokenizer())
    for dataset in DATASETS:
        case = case_from_prediction(prediction(dataset, 0))
        input_ids, instruction_ids, data_ids = encoder._encode(case)
        if dataset == "deepset":
            expected_instruction = "Say xxxxxx"
            expected_data = "Some user data"
        else:
            expected_instruction = "Follow task.\nContext:\nFind answer"
            expected_data = "external text"
        rendered = case.metadata["formatted_text"]
        assert "".join(rendered[i] for i in instruction_ids) == expected_instruction
        assert "".join(rendered[i] for i in data_ids) == expected_data
        assert input_ids.shape[1] == len(rendered)


def test_load_points_and_save_both_plots(tmp_path):
    for dataset in DATASETS:
        path = tmp_path / f"{dataset}_predictions.jsonl"
        path.write_text(
            "\n".join(json.dumps(prediction(dataset, label)) for label in (0, 1)) + "\n",
            encoding="utf-8",
        )
    points = load_points(tmp_path, CharacterTokenizer())
    assert len(points) == 8
    for dataset, label, instruction_tokens, ratio, score in points:
        row = prediction(dataset, label)
        expected = (len("Say xxxxxx") if dataset == "deepset"
                    else len("Follow task.\nContext:\n") + len("Find answer"))
        assert instruction_tokens == expected
        assert ratio == expected / len(row["formatted_text"])
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
    (tmp_path / "tracker_qwen3_8b.json").write_text(
        json.dumps({"model_id": "test-model"}), encoding="utf-8"
    )
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

    tokenizer = CharacterTokenizer()
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(make_plot, "RESULTS", tmp_path)

    make_plot.main()

    for stem in ("instruction_token_count", "instruction_token_ratio"):
        assert (tmp_path / f"{stem}_vs_focus.png").is_file()
        assert (probe_results / f"{stem}_vs_score_drop.png").is_file() == with_probe_results
    if with_probe_results:
        points = load_points(
            probe_results, tokenizer, score_field="score_drop", datasets=("deepset",)
        )
        assert [point[4] for point in points] == [0.4, -0.2]

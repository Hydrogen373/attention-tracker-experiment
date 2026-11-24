"""Plot Focus Score against trusted-instruction length from saved experiment results."""

import json
import math
from pathlib import Path

from tracker.model import QwenAttentionModel
from tracker.records import PromptCase


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATASETS = {
    "deepset": ("Deepset", "red"),
    "bipia_email": ("BIPIA email", "blue"),
    "bipia_table": ("BIPIA table", "green"),
    "bipia_code": ("BIPIA code", "purple"),
}
SYSTEM_START = "<|im_start|>system\n"
USER_START = "<|im_end|>\n<|im_start|>user\n"


class SavedPromptEncoder:
    """Use the experiment's exact token-to-attention-span assignment without a GPU."""

    _encode = QwenAttentionModel._encode

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_input_tokens = float("inf")

    def format_input(self, case):
        return case.metadata["formatted_text"]


def case_from_prediction(row):
    """Rebuild the attention spans from fields saved by run_experiment.py."""
    dataset = row["dataset"]
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")

    rendered = row["formatted_text"]
    user = row["text"] if dataset == "deepset" else row["user_question"]
    if not rendered.startswith(SYSTEM_START):
        raise ValueError(f"Unexpected chat template in {row['sample_id']}")
    boundary = rendered.find(USER_START + user, len(SYSTEM_START))
    if boundary < 0:
        raise ValueError(f"User text missing from formatted prompt in {row['sample_id']}")
    system = rendered[len(SYSTEM_START):boundary]

    if dataset == "deepset":
        instruction = system
        data_parts = (user,)
        attention_parts = None
    else:
        context = row["context"]
        if not context or not system.endswith(context):
            raise ValueError(f"Context is not at the end of the system prompt in {row['sample_id']}")
        instruction = system[:-len(context)]
        data_parts = (context,)
        attention_parts = (
            ("instruction", instruction),
            ("data", context),
            ("instruction", user),
        )

    return PromptCase(
        sample_id=row["sample_id"], dataset=dataset, label=row["label"],
        system=system, user=user, instruction=instruction,
        data_parts=data_parts, metadata={"formatted_text": rendered},
        attention_parts=attention_parts,
    )


def load_points(results_dir, tokenizer):
    encoder = SavedPromptEncoder(tokenizer)
    points = []
    for dataset in DATASETS:
        path = results_dir / f"{dataset}_predictions.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing experiment results: {path}. Run run_experiment.py first.")
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["dataset"] != dataset or row["label"] not in (0, 1):
                    raise ValueError(f"Invalid dataset or label in {path}:{line_number}")
                score = float(row["focus_score"])
                if not math.isfinite(score):
                    raise ValueError(f"Non-finite Focus Score in {path}:{line_number}")
                case = case_from_prediction(row)
                input_ids, instruction_ids, _ = encoder._encode(case)
                instruction_tokens = len(instruction_ids)
                total_tokens = input_ids.shape[1]
                points.append((dataset, case.label, instruction_tokens,
                               instruction_tokens / total_tokens, score))
    if not points:
        raise ValueError(f"No prediction rows found in {results_dir}")
    return points


def save_plots(points, results_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts = (
        (2, "Instruction tokens", "instruction_token_count_vs_focus.png"),
        (3, "Instruction tokens / all input tokens", "instruction_token_ratio_vs_focus.png"),
    )
    for x_index, x_label, filename in charts:
        fig, ax = plt.subplots(figsize=(10, 6))
        for dataset, (name, color) in DATASETS.items():
            for label, marker, kind in ((0, "o", "Clean"), (1, "^", "Attack")):
                group = [point for point in points if point[0] == dataset and point[1] == label]
                ax.scatter(
                    [point[x_index] for point in group], [point[4] for point in group],
                    c=color, marker=marker, s=30, alpha=0.6, edgecolors="none",
                    label=f"{name} · {kind}",
                )
        ax.set(xlabel=x_label, ylabel="Focus Score", ylim=(-0.02, 1.02))
        if x_index == 3:
            ax.set_xlim(-0.02, 1.02)
        ax.grid(alpha=0.2)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(results_dir / filename, dpi=180)
        plt.close(fig)


def main():
    from transformers import AutoTokenizer

    checkpoint = RESULTS / "tracker_qwen3_8b.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing head checkpoint: {checkpoint}. Run run_experiment.py first.")
    model_id = json.loads(checkpoint.read_text(encoding="utf-8"))["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer with character offsets is required")
    points = load_points(RESULTS, tokenizer)
    save_plots(points, RESULTS)
    print(f"Saved two Focus Score plots to {RESULTS}")


if __name__ == "__main__":
    main()

"""Plot Focus Score and ProbeDelta score drop against trusted-instruction length."""

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATASETS = {
    "deepset": ("Deepset", "red"),
    "bipia_email": ("BIPIA email", "blue"),
    "bipia_table": ("BIPIA table", "green"),
    "bipia_code": ("BIPIA code", "purple"),
}
METRICS = {
    "focus_score": ("Focus Score", "focus", (-0.02, 1.02)),
    "score_drop": ("Score Drop", "score_drop", None),
}


def load_points(results_dir, score_field="focus_score", datasets=None):
    points = []
    if datasets is None:
        datasets = DATASETS
    for dataset in datasets:
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
                score = float(row[score_field])
                if not math.isfinite(score):
                    raise ValueError(f"Non-finite {score_field} in {path}:{line_number}")
                for field in ("instruction_token_count", "total_token_count"):
                    if field not in row:
                        raise ValueError(
                            f"Missing {field} in {path}:{line_number}; rerun the experiment"
                        )
                    if type(row[field]) is not int:
                        raise ValueError(f"Invalid {field} in {path}:{line_number}")
                instruction_tokens = row["instruction_token_count"]
                total_tokens = row["total_token_count"]
                if not 0 < instruction_tokens <= total_tokens:
                    raise ValueError(f"Invalid token counts in {path}:{line_number}")
                points.append((dataset, row["label"], instruction_tokens,
                               instruction_tokens / total_tokens, score))
    if not points:
        raise ValueError(f"No prediction rows found in {results_dir}")
    return points


def save_plots(points, results_dir, score_field="focus_score"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_label, suffix, y_limits = METRICS[score_field]
    charts = (
        (2, "Instruction tokens", f"instruction_token_count_vs_{suffix}.png"),
        (3, "Instruction tokens / all input tokens", f"instruction_token_ratio_vs_{suffix}.png"),
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
        ax.set(xlabel=x_label, ylabel=y_label)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        if x_index == 3:
            ax.set_xlim(-0.02, 1.02)
        ax.grid(alpha=0.2)
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(results_dir / filename, dpi=180)
        plt.close(fig)


def main():
    points = load_points(RESULTS)
    save_plots(points, RESULTS)
    print(f"Saved two Focus Score plots to {RESULTS}")

    probe_results = RESULTS / "probedelta"
    probe_datasets = [
        dataset for dataset in DATASETS
        if (probe_results / f"{dataset}_predictions.jsonl").is_file()
    ]
    if probe_datasets:
        probe_points = load_points(
            probe_results, score_field="score_drop", datasets=probe_datasets
        )
        save_plots(probe_points, probe_results, score_field="score_drop")
        print(f"Saved two ProbeDelta plots to {probe_results}")


if __name__ == "__main__":
    main()

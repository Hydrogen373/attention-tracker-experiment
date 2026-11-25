"""CPU-only analysis of saved predictions: global vs per-task vs length-conditional
thresholds, all fit on benign calibration data at the same target FPR and
evaluated on source contexts that calibration never saw.

    python evaluate_thresholds.py --results-dir results
"""
import argparse
import json
from pathlib import Path

from tracker.thresholds import LENGTH_FEATURES, load_predictions, run


def fmt(stat: dict) -> str:
    return f"{stat['mean']:.3f} (sd {stat['std']:.3f})"


def print_tables(summary: dict):
    split = summary["context_disjoint_split"]
    policies = list(next(iter(split["per_dataset"].values())).keys())
    print(f"\n## Context-disjoint split (target FPR={summary['config']['target_fpr']}, "
          f"{summary['config']['n_repeats']} repeats)\n")
    print("| dataset | " + " | ".join(f"{p} TPR | {p} FPR" for p in policies) + " |")
    print("|---|" + "---|---|" * len(policies))
    for name, by_policy in split["per_dataset"].items():
        cells = [f"{fmt(by_policy[p]['tpr'])} | {fmt(by_policy[p]['fpr'])}" for p in policies]
        print(f"| {name} | " + " | ".join(cells) + " |")
    pooled = split["pooled"]
    print("\nPooled AUROC, raw Focus Score:", fmt(pooled["auroc_raw_score"]))
    if "auroc_length_residual" in pooled:
        print("Pooled AUROC, length-adjusted score:", fmt(pooled["auroc_length_residual"]))

    loo = summary["leave_one_dataset_out"]
    if loo:
        print("\n## Leave-one-dataset-out (threshold fit without the held-out dataset)\n")
        policies = list(next(iter(loo.values())).keys())
        print("| held out | " + " | ".join(f"{p} TPR | {p} FPR" for p in policies) + " |")
        print("|---|" + "---|---|" * len(policies))
        for name, by_policy in loo.items():
            cells = [f"{fmt(by_policy[p]['tpr'])} | {fmt(by_policy[p]['fpr'])}" for p in policies]
            print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--calib-fraction", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--length-feature", choices=[*LENGTH_FEATURES, "none"], default="instruction_ratio")
    parser.add_argument("--datasets", default="", help="Comma-separated subset, e.g. deepset,bipia_email")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: <results-dir>/threshold_policies.json")
    args = parser.parse_args()

    rows = load_predictions(args.results_dir)
    if args.datasets:
        wanted = {name.strip() for name in args.datasets.split(",") if name.strip()}
        rows = [row for row in rows if row["dataset"] in wanted]
    summary = run(
        rows, n_repeats=args.repeats, calib_fraction=args.calib_fraction,
        target_fpr=args.target_fpr,
        feature=None if args.length_feature == "none" else args.length_feature,
        seed=args.seed,
    )
    output = args.output or args.results_dir / "threshold_policies.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_tables(summary)
    print(f"\nSaved {output}")

"""Score probe delta with important heads loaded from an existing checkpoint."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from tracker.datasets import BIPIADataset, DeepsetDataset
from tracker.detector import AttentionTracker, ProbeDeltaAttentionTracker
from tracker.model import QwenAttentionModel


def run_dataset(name, cases, model, detector, output_dir, generate=False, max_new_tokens=128):
    """Score original prompts, not the modified probe prompts used internally."""
    rows, skipped = [], []
    for index, case in enumerate(cases, 1):
        try:
            drop = detector.score_drop(case)
            baseline = model.generate(case, max_new_tokens) if generate else None
        except ValueError as exc:
            if "tokens >" not in str(exc) and "Generation prompt too long" not in str(exc):
                raise
            skipped.append({"sample_id": case.sample_id, "reason": str(exc)})
            continue
        rows.append({
            "sample_id": case.sample_id, "dataset": name, "label": case.label,
            "score_drop": drop, "probe_word": detector.probe_word(case),
            "probe_text": detector.probe_text(case),
            **case.metadata,
            "formatted_text": model.format_input(case),
            "baseline_response": baseline,
        })
        if index % 10 == 0 or index == len(cases):
            print(f"[{name}] {index}/{len(cases)} processed; {len(skipped)} skipped")
    if not rows:
        raise RuntimeError(f"No scoreable cases for {name}; increase --max-input-tokens")

    labels = np.array([row["label"] for row in rows])
    drops = np.array([row["score_drop"] for row in rows])
    metrics = {
        "dataset": name, "num_input": len(cases), "num_scored": len(rows),
        "num_skipped_too_long": len(skipped),
        "num_clean": int((labels == 0).sum()),
        "num_attacked": int((labels == 1).sum()),
        "num_selected_heads": len(detector.important_heads),
        "auroc": float(roc_auc_score(labels, -drops)) if len(set(labels)) == 2 else None,
        "response_generation_enabled": generate,
    }
    if name.startswith("bipia_"):
        pairs = {}
        for row in rows:
            pairs.setdefault(row["pair_id"], {})[row["label"]] = row
        complete = [pair for pair in pairs.values() if 0 in pair and 1 in pair]
        metrics["num_complete_pairs"] = len(complete)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if skipped:
        with (output_dir / f"{name}_skipped.jsonl").open("w", encoding="utf-8") as stream:
            for row in skipped:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def load_existing_heads(detector, checkpoint: Path):
    """Validate a base AttentionTracker checkpoint before reusing its heads."""
    saved = AttentionTracker(detector.model)
    saved.load(checkpoint)
    detector.k = saved.k
    detector.important_heads = saved.important_heads.copy()


def main():
    parser = argparse.ArgumentParser(description="Qwen3 probe-delta injection experiment")
    parser.add_argument("--benchmark", choices=["deepset", "bipia", "both"], default="both")
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--deepset-test-limit", type=int, default=0)
    parser.add_argument("--bipia-repo", type=Path, default=Path("external/BIPIA"))
    parser.add_argument("--bipia-tasks", default="email,table,code")
    parser.add_argument("--bipia-max-contexts", type=int, default=3)
    parser.add_argument("--bipia-max-variants-per-attack", type=int, default=2)
    parser.add_argument("--bipia-max-pairs", type=int, default=40)
    parser.add_argument("--generate-bipia-responses", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("results/probedelta"))
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    if not args.head_checkpoint.is_file():
        parser.error(f"Head checkpoint not found: {args.head_checkpoint}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = QwenAttentionModel(args.model_id, args.max_input_tokens, args.dtype)
    detector = ProbeDeltaAttentionTracker(model, seed=args.seed)
    load_existing_heads(detector, args.head_checkpoint)
    print(f"Loaded {len(detector.important_heads)} heads (k={detector.k}) from {args.head_checkpoint}")

    summary = {}
    if args.benchmark in {"deepset", "both"}:
        cases = DeepsetDataset().load("test", limit=args.deepset_test_limit)
        summary["deepset"] = run_dataset("deepset", cases, model, detector, args.output_dir)
    if args.benchmark in {"bipia", "both"}:
        bipia = BIPIADataset(
            args.bipia_repo, args.seed, args.bipia_max_contexts,
            args.bipia_max_variants_per_attack, args.bipia_max_pairs,
        )
        tasks = [task.strip() for task in args.bipia_tasks.split(",") if task.strip()]
        if not tasks:
            parser.error("--bipia-tasks must name at least one task")
        for task in tasks:
            name = f"bipia_{task}"
            cases = bipia.load(task, split="test")
            summary[name] = run_dataset(
                name, cases, model, detector, args.output_dir,
                generate=args.generate_bipia_responses,
                max_new_tokens=args.max_new_tokens,
            )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Finished. Results in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

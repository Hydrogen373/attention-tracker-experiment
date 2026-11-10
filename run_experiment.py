"""CLI: (1) load model, (2) select heads, (3) threshold, (4) evaluate."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from tracker.datasets import BIPIADataset, DeepsetDataset, SyntheticHeadDataset
from tracker.detector import AttentionTracker
from tracker.experiment import ExperimentRunner
from tracker.model import QwenAttentionModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention Tracker on Qwen3-8B")
    parser.add_argument("--benchmark", choices=["deepset", "bipia", "both"], default="both")
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--k", type=float, default=4.0)
    parser.add_argument("--num-head-examples", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--num-calibration-benign", type=int, default=60)
    parser.add_argument("--head-checkpoint", type=Path, default=Path("results/tracker_qwen3_8b.json"))
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--deepset-test-limit", type=int, default=0)
    parser.add_argument("--bipia-repo", type=Path, default=Path("external/BIPIA"))
    parser.add_argument("--bipia-tasks", default="email,table,code")
    parser.add_argument("--bipia-max-contexts", type=int, default=3)
    parser.add_argument("--bipia-max-variants-per-attack", type=int, default=2)
    parser.add_argument("--bipia-max-pairs", type=int, default=40)
    parser.add_argument("--generate-bipia-responses", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # The model and detector are shared across both benchmarks: one head set,
    # one threshold, consistent scoring, no retraining between datasets.
    model = QwenAttentionModel(args.model_id, args.max_input_tokens, args.dtype)
    detector = AttentionTracker(model, k=args.k)

    if args.head_checkpoint.exists() and not args.force_refit:
        detector.load(args.head_checkpoint)
        if detector.k != args.k:
            raise ValueError("Saved k differs from --k. Use --force-refit to reselect heads")
        print(f"Loaded {len(detector.important_heads)} heads from {args.head_checkpoint}")
    else:
        pairs = SyntheticHeadDataset().load_pairs(args.num_head_examples)
        print(f"Selecting important heads from {len(pairs)} synthetic pairs...")
        candidate_scores = detector.fit_heads(pairs)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.output_dir / "head_candidate_scores.npy", candidate_scores)
        print(f"Important heads selected: {len(detector.important_heads)} / "
              f"{model.num_layers * model.num_heads}")

    if args.threshold is not None:
        detector.threshold = args.threshold
    elif detector.threshold is None:
        # Calibration uses *benign train only*; test labels are not used to fit
        # head selection or threshold. Same threshold transfers to BIPIA.
        calibration = DeepsetDataset().load(
            "train", limit=args.num_calibration_benign, label=0
        )
        print(f"Calibrating at target FPR={args.target_fpr} from "
              f"{len(calibration)} deepset train benign prompts...")
        detector.calibrate_benign(calibration, args.target_fpr)
        print(f"Threshold: {detector.threshold:.6f}")

    detector.save(args.head_checkpoint)
    runner = ExperimentRunner(model, detector, args.output_dir, args.max_new_tokens)
    all_metrics = {}

    if args.benchmark in {"deepset", "both"}:
        cases = DeepsetDataset().load("test", limit=args.deepset_test_limit)
        all_metrics["deepset"] = runner.run("deepset", cases)

    if args.benchmark in {"bipia", "both"}:
        bipia = BIPIADataset(
            args.bipia_repo, args.seed, args.bipia_max_contexts,
            args.bipia_max_variants_per_attack, args.bipia_max_pairs,
        )
        tasks = [value.strip() for value in args.bipia_tasks.split(",") if value.strip()]
        for task in tasks:
            cases = bipia.load(task, split="test")
            name = f"bipia_{task}"
            all_metrics[name] = runner.run(
                name, cases, generate=args.generate_bipia_responses,
            )

    (args.output_dir / "summary.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Finished. Results in {args.output_dir.resolve()}")

"""Same runner and metric definitions for both benchmark adapters."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .records import Prediction, PromptCase


class ExperimentRunner:
    def __init__(self, model, detector, output_dir: Path, max_new_tokens: int = 128):
        self.model = model
        self.detector = detector
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, name: str, cases: list[PromptCase], generate: bool = False):
        predictions, skipped = [], []
        for index, case in enumerate(cases, start=1):
            try:
                focus = self.detector.focus_score(case)
                baseline = None
                if generate:
                    baseline = self.model.generate(case, self.max_new_tokens)
                predictions.append(Prediction(
                    case.sample_id, case.dataset, case.label,
                    focus, self.model.format_input(case), case.metadata, baseline,
                ))
            except ValueError as exc:
                # A case is never silently truncated: exclusion counts are reported.
                if "tokens >" not in str(exc) and "Generation prompt too long" not in str(exc):
                    raise
                skipped.append({"sample_id": case.sample_id, "reason": str(exc)})
            if index % 10 == 0 or index == len(cases):
                print(f"[{name}] {index}/{len(cases)} processed; {len(skipped)} skipped")

        if not predictions:
            raise RuntimeError(f"No scoreable cases for {name}; increase --max-input-tokens")

        y = np.array([p.label for p in predictions])
        scores = np.array([p.focus_score for p in predictions])
        metrics = {
            "dataset": name,
            "num_input": len(cases),
            "num_scored": len(predictions),
            "num_skipped_too_long": len(skipped),
            "num_clean": int((y == 0).sum()),
            "num_attacked": int((y == 1).sum()),
            "num_selected_heads": len(self.detector.important_heads),
            "auroc": float(roc_auc_score(y, -scores)) if len(set(y)) == 2 else None,
            "response_generation_enabled": generate,
        }
        if name.startswith("bipia_"):
            pairs = {}
            for p in predictions:
                pairs.setdefault(p.metadata["pair_id"], {})[p.label] = p
            complete = [pair for pair in pairs.values() if 0 in pair and 1 in pair]
            metrics["num_complete_pairs"] = len(complete)

        (self.output_dir / f"{name}_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (self.output_dir / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as file:
            for item in predictions:
                file.write(json.dumps(item.to_dict(), ensure_ascii=False, default=str) + "\n")
        if skipped:
            with (self.output_dir / f"{name}_skipped.jsonl").open("w", encoding="utf-8") as file:
                for item in skipped:
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return metrics

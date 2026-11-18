"""Optimize focus-score thresholds against observed baseline injection outcomes."""
import csv
from pathlib import Path


ROOT = Path(__file__).parent / "results"
summary = []


def confusion(rows, threshold):
    tp = fp = tn = fn = 0
    for row in rows:
        predicted = float(row["focus_score"]) < threshold
        actual = row["injection_occurred"].lower() == "true"
        if predicted and actual:
            tp += 1
        elif predicted:
            fp += 1
        elif actual:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


for path in sorted(ROOT.glob("*_predictions.csv")):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames)
        rows = list(reader)
    known = [row for row in rows if row["injection_occurred"].lower() in ("true", "false")]
    threshold = None
    tp = fp = tn = fn = 0
    if known:
        scores = sorted({float(row["focus_score"]) for row in known})
        candidates = [scores[0] - 1e-9]
        candidates += [(left + right) / 2 for left, right in zip(scores, scores[1:])]
        candidates += [scores[-1] + 1e-9]
        # Equal accuracy: prefer fewer false positives, then the lower threshold.
        threshold = max(
            candidates,
            key=lambda value: (
                (lambda a, b, c, d: (a + c, -b, -value))(*confusion(known, value))
            ),
        )
        tp, fp, tn, fn = confusion(known, threshold)

    for row in rows:
        predicted = float(row["focus_score"]) < threshold if threshold is not None else None
        row["injection_optimal_threshold"] = repr(threshold) if threshold is not None else ""
        row["injection_optimal_prediction"] = str(predicted) if predicted is not None else ""
        row["injection_optimal_correct"] = (
            str(predicted == (row["injection_occurred"].lower() == "true"))
            if row["injection_occurred"].lower() in ("true", "false") and predicted is not None
            else ""
        )
    for column in (
        "injection_optimal_threshold", "injection_optimal_prediction", "injection_optimal_correct"
    ):
        if column not in columns:
            columns.append(column)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summary.append(dict(
        dataset=path.name.removesuffix("_predictions.csv"), threshold=threshold if threshold is not None else "",
        total_rows=len(rows), scored_rows=len(known), injection_positive=tp + fn,
        injection_negative=fp + tn, accuracy=(tp + tn) / len(known) if known else "",
        tp=tp, fp=fp, tn=tn, fn=fn,
    ))

with (ROOT / "injection_optimal_threshold_analysis.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

for row in summary:
    print(row)

"""Find the in-sample accuracy optimum for each prediction CSV."""
import csv
from pathlib import Path


ROOT = Path(__file__).parent / "results"
summary = []


def confusion(rows, threshold, truth):
    tp = fp = tn = fn = 0
    for row in rows:
        predicted = float(row["focus_score"]) < threshold
        actual = truth(row)
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

    scores = sorted({float(row["focus_score"]) for row in rows})
    thresholds = [scores[0] - 1e-9]
    thresholds += [(left + right) / 2 for left, right in zip(scores, scores[1:])]
    thresholds += [scores[-1] + 1e-9]
    label_truth = lambda row: row["label"] == "1"
    # Among equal accuracy optima, prefer fewer rejected clean examples.
    threshold = max(
        thresholds,
        key=lambda value: (
            (lambda tp, fp, tn, fn: (tp + tn, -fp))(
                *confusion(rows, value, label_truth)
            )
        ),
    )
    label_tp, label_fp, label_tn, label_fn = confusion(rows, threshold, label_truth)
    known = [row for row in rows if row["injection_occurred"].lower() in ("true", "false")]
    attacked = [row for row in known if row["label"] == "1"]
    actual_truth = lambda row: row["injection_occurred"].lower() == "true"
    actual_tp, actual_fp, actual_tn, actual_fn = confusion(known, threshold, actual_truth)
    attacked_tp, attacked_fp, attacked_tn, attacked_fn = confusion(attacked, threshold, actual_truth)

    for row in rows:
        predicted = float(row["focus_score"]) < threshold
        row["optimal_accuracy_threshold"] = repr(threshold)
        row["optimal_attack_prediction"] = str(predicted)
        row["optimal_label_correct"] = str(predicted == label_truth(row))
        row["optimal_injection_match"] = (
            str(predicted == actual_truth(row))
            if row["injection_occurred"].lower() in ("true", "false") else ""
        )
    for column in (
        "optimal_accuracy_threshold", "optimal_attack_prediction",
        "optimal_label_correct", "optimal_injection_match",
    ):
        if column not in columns:
            columns.append(column)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summary.append(dict(
        dataset=path.name.removesuffix("_predictions.csv"), threshold=threshold,
        n=len(rows), label_accuracy=(label_tp + label_tn) / len(rows),
        label_tp=label_tp, label_fp=label_fp, label_tn=label_tn, label_fn=label_fn,
        actual_known=len(known), actual_tp=actual_tp, actual_fp=actual_fp,
        actual_tn=actual_tn, actual_fn=actual_fn,
        actual_accuracy=(actual_tp + actual_tn) / len(known) if known else "",
        attacked_known=len(attacked), attacked_actual_tp=attacked_tp,
        attacked_actual_fp=attacked_fp, attacked_actual_tn=attacked_tn,
        attacked_actual_fn=attacked_fn,
    ))

with (ROOT / "optimal_threshold_analysis.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

for row in summary:
    print(row)

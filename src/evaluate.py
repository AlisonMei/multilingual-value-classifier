"""
Validate classifier output against the human gold-standard labels.

This is the core of the project: the language model is treated as a measurement
instrument, and we quantify how well it agrees with expert human judgement,
overall AND per language, rather than trusting its output.

Reports:
  - overall accuracy
  - Cohen's kappa (chance-corrected human-machine agreement)
  - per-class precision / recall / F1
  - per-language accuracy (to detect language-specific bias)
  - confusion matrix

Pure-Python implementation (no sklearn) so it runs anywhere.
"""

import csv
from collections import defaultdict
from pathlib import Path

LABELS = ["intrinsic", "instrumental", "relational"]


def cohens_kappa(y_true, y_pred, labels):
    n = len(y_true)
    # observed agreement
    po = sum(1 for a, b in zip(y_true, y_pred) if a == b) / n
    # expected agreement
    t_counts = {l: y_true.count(l) / n for l in labels}
    p_counts = {l: y_pred.count(l) / n for l in labels}
    pe = sum(t_counts[l] * p_counts[l] for l in labels)
    return (po - pe) / (1 - pe) if (1 - pe) else 0.0


def per_class_prf(y_true, y_pred, labels):
    out = {}
    for l in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == l and b == l)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != l and b == l)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == l and b != l)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[l] = {"precision": prec, "recall": rec, "f1": f1, "support": y_true.count(l)}
    return out


def confusion(y_true, y_pred, labels):
    m = {a: {b: 0 for b in labels} for a in labels}
    for a, b in zip(y_true, y_pred):
        if a in labels and b in labels:
            m[a][b] += 1
    return m


def evaluate(rows, pred_key="pred_label", gold_key="gold_label"):
    y_true = [r[gold_key] for r in rows]
    y_pred = [r[pred_key] for r in rows]
    acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(rows)
    kappa = cohens_kappa(y_true, y_pred, LABELS)
    prf = per_class_prf(y_true, y_pred, LABELS)
    cm = confusion(y_true, y_pred, LABELS)

    # per-language
    by_lang = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in rows:
        by_lang[r["language"]]["n"] += 1
        if r[gold_key] == r[pred_key]:
            by_lang[r["language"]]["correct"] += 1
    lang_acc = {lg: v["correct"] / v["n"] for lg, v in by_lang.items()}

    return {"accuracy": acc, "kappa": kappa, "per_class": prf,
            "per_language": lang_acc, "confusion": cm}


def format_report(res):
    lines = []
    lines.append("# Validation report: classifier vs. human gold standard\n")
    lines.append(f"Overall accuracy: {res['accuracy']:.3f}")
    lines.append(f"Cohen's kappa (human-machine agreement): {res['kappa']:.3f}\n")

    lines.append("## Per-class performance")
    lines.append("| value type | precision | recall | F1 | support |")
    lines.append("|---|---|---|---|---|")
    for l in LABELS:
        m = res["per_class"][l]
        lines.append(f"| {l} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['support']} |")

    lines.append("\n## Per-language accuracy (bias check)")
    lines.append("| language | accuracy |")
    lines.append("|---|---|")
    for lg, a in sorted(res["per_language"].items()):
        lines.append(f"| {lg} | {a:.3f} |")

    lines.append("\n## Confusion matrix (rows = gold, cols = predicted)")
    header = "| gold \\ pred | " + " | ".join(LABELS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(LABELS) + 1))
    for a in LABELS:
        row = " | ".join(str(res["confusion"][a][b]) for b in LABELS)
        lines.append(f"| {a} | {row} |")

    lines.append("\n_Note: relational values typically show lower recall than "
                 "instrumental ones, because they are the hardest class to "
                 "identify. Reporting per-class and per-language metrics makes "
                 "that visible rather than hiding it in an aggregate score._")
    return "\n".join(lines)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    pred_file = root / "results" / "predictions.csv"
    with open(pred_file, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    res = evaluate(rows)
    report = format_report(res)
    out = root / "results" / "validation_report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved report to {out}")

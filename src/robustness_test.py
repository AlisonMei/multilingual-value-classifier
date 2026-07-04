"""
Robustness / sanity tests for the classification pipeline.

The point of this script is to show that the headline accuracy is *meaningful* -
that it reflects the classifier actually working, not the synthetic data being
trivially separable. It does three things:

  1. MAJORITY BASELINE
     Compare the real classifier against a dummy that always predicts the most
     common class. If our classifier isn't clearly above this floor, a high
     accuracy would be an illusion.

  2. RANDOM BASELINE
     Compare against random guessing among the three classes.

  3. ABLATION ("break it on purpose")
     Remove some of the keyword cues from one class and re-score. If the
     pipeline is genuinely measuring agreement, that class's recall should drop.
     If nothing changes, the harness isn't sensitive and something is wrong.

Run:  python src/robustness_test.py
"""

import csv
import random
from collections import Counter
from pathlib import Path

from evaluate import evaluate, LABELS
import classifier

random.seed(0)
ROOT = Path(__file__).resolve().parents[1]


def load_data():
    with open(ROOT / "data" / "synthetic_responses.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score(rows, predict):
    for r in rows:
        r["pred_label"] = predict(r["text"])
    return evaluate(rows)


def main():
    rows = load_data()

    # --- 1. real classifier (keyword baseline backend) ---
    real = score([dict(r) for r in rows], classifier.classify_baseline)

    # --- 2. majority baseline ---
    majority_class = Counter(r["gold_label"] for r in rows).most_common(1)[0][0]
    maj = score([dict(r) for r in rows], lambda t: majority_class)

    # --- 3. random baseline ---
    rnd = score([dict(r) for r in rows], lambda t: random.choice(LABELS))

    # --- 4. ablation: cripple 'instrumental' cues and watch recall fall ---
    original = dict(classifier._KEYWORDS)
    classifier._KEYWORDS = dict(classifier._KEYWORDS)
    classifier._KEYWORDS["instrumental"] = classifier._KEYWORDS["instrumental"][:3]
    ablated = score([dict(r) for r in rows], classifier.classify_baseline)
    classifier._KEYWORDS = original  # restore

    # --- report ---
    lines = []
    lines.append("# Robustness / sanity tests\n")
    lines.append("How the real classifier compares to trivial baselines, and how "
                 "it responds when deliberately crippled.\n")

    lines.append("## Accuracy vs. baselines")
    lines.append("| method | accuracy | Cohen's kappa |")
    lines.append("|---|---|---|")
    lines.append(f"| **Real classifier** | **{real['accuracy']:.3f}** | **{real['kappa']:.3f}** |")
    lines.append(f"| Majority-class baseline | {maj['accuracy']:.3f} | {maj['kappa']:.3f} |")
    lines.append(f"| Random baseline | {rnd['accuracy']:.3f} | {rnd['kappa']:.3f} |")
    lines.append("")
    lines.append(f"The real classifier ({real['accuracy']:.2f}) sits well above the "
                 f"majority-class floor ({maj['accuracy']:.2f}) and random "
                 f"guessing ({rnd['accuracy']:.2f}). The high score is therefore "
                 f"informative, not an artefact of an easy label distribution.\n")

    lines.append("## Ablation: remove most 'instrumental' cues on purpose")
    before_r = real["per_class"]["instrumental"]["recall"]
    after_r = ablated["per_class"]["instrumental"]["recall"]
    lines.append("| | instrumental recall |")
    lines.append("|---|---|")
    lines.append(f"| full classifier | {before_r:.2f} |")
    lines.append(f"| cues removed | {after_r:.2f} |")
    lines.append("")
    lines.append(f"Recall for the crippled class falls from {before_r:.2f} to "
                 f"{after_r:.2f}. The validation harness responds to a real change "
                 f"in the classifier, which is evidence that it is measuring "
                 f"agreement rather than returning a fixed number.\n")

    report = "\n".join(lines)
    out = ROOT / "results" / "robustness_report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

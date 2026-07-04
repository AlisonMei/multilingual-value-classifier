"""
Run the full pipeline end to end:

  1. load the synthetic multilingual dataset
  2. classify each response into a value type
  3. save predictions alongside the gold labels
  4. (evaluate.py then scores predictions against the gold standard)

By default this uses the offline keyword baseline so it runs with no API key.
Pass --llm to use few-shot Anthropic classification instead (needs
ANTHROPIC_API_KEY and the `anthropic` package).
"""

import argparse
import csv
from pathlib import Path

from classifier import classify_baseline, classify_llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true",
                    help="use few-shot LLM classification (needs ANTHROPIC_API_KEY)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_file = root / "data" / "synthetic_responses.csv"
    out_file = root / "results" / "predictions.csv"

    with open(data_file, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    classify = classify_llm if args.llm else classify_baseline
    backend = "LLM few-shot" if args.llm else "keyword baseline"
    print(f"Classifying {len(rows)} responses using: {backend}")

    for r in rows:
        r["pred_label"] = classify(r["text"])

    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "language", "distance_level", "text",
                           "gold_label", "pred_label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote predictions to {out_file}")
    print("Next: python src/evaluate.py")


if __name__ == "__main__":
    main()

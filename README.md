# Multilingual value classification with LLM + human validation

A compact, reproducible pipeline that classifies short multilingual statements
about *why people value a place* into three value types — **intrinsic**,
**instrumental**, **relational** (following the IPBES framing) — using few-shot
large-language-model classification, and then **validates the model against a
human gold standard** rather than trusting its output.

> **Why this repo exists.** It demonstrates a way of working I care about:
> treating a language model as a *measurement instrument* whose agreement with
> expert human judgement is quantified — per class and per language — with
> validation built into the pipeline, not bolted on afterwards. The data here
> is fully synthetic (see below); the point is the method, not the numbers.

---

## The idea in one picture

```
multilingual text ──► few-shot LLM classification ──► predicted value type
   (nl / de / en)              (codebook in prompt)            │
                                                               ▼
   human gold-standard labels ──────────────────────►  VALIDATION
                                                  accuracy · Cohen's κ
                                                  per-class P / R / F1
                                                  per-language bias check
                                                  confusion matrix
```

The classification backend is **swappable**; the validation stage is the fixed
core. That separation is the whole point: you can change the model, the prompt,
or the examples, and the validation harness tells you whether agreement with
human judgement got better or worse.

---

## What's inside

| file | role |
|---|---|
| `src/generate_data.py` | builds a synthetic multilingual, value-labelled dataset from templates |
| `src/classifier.py` | two backends: few-shot **Anthropic LLM** classification, and an offline **keyword baseline** so the repo runs with no API key |
| `src/run_pipeline.py` | end-to-end: load → classify → save predictions |
| `src/evaluate.py` | scores predictions against the gold standard (accuracy, κ, per-class P/R/F1, per-language accuracy, confusion matrix) |
| `src/robustness_test.py` | sanity checks: compares against majority/random baselines, and deliberately cripples the classifier to confirm the validation harness responds |
| `results/validation_report.md` | the generated validation report |
| `results/robustness_report.md` | the generated robustness / sanity-check report |

---

## Run it

No API key needed (uses the offline baseline):

```bash
python src/generate_data.py     # writes data/synthetic_responses.csv
python src/run_pipeline.py       # writes results/predictions.csv
python src/evaluate.py           # writes results/validation_report.md
python src/robustness_test.py    # writes results/robustness_report.md
```

To use few-shot LLM classification instead:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...     # your key
python src/run_pipeline.py --llm
python src/evaluate.py
```

---

## Methodological notes

- **The LLM is treated as an instrument to be validated.** Every run is scored
  against expert (here, synthetic) gold labels. Aggregate accuracy alone is not
  trusted: per-class metrics are reported because the hardest class (relational)
  can hide behind a high overall score.
- **Per-language reporting is a first-class check**, not an afterthought —
  because a multilingual classifier can perform unevenly across languages, and
  that would quietly bias any downstream cross-language comparison.
- **The codebook lives in the prompt.** Class definitions and disambiguation
  rules are embedded directly, so the model classifies against the same written
  criteria a human coder would use — and those criteria are inspectable.
- **Failure modes are surfaced, not hidden.** The confusion matrix shows exactly
  where the classifier confuses one value type for another.
- **The headline number is stress-tested.** `robustness_test.py` checks that the
  classifier beats majority-class and random baselines (so the score is
  informative, not an artefact of the label distribution), and that deliberately
  removing cues makes the relevant class's recall fall (so the validation harness
  is genuinely sensitive, not returning a fixed number).

## About the data

`data/synthetic_responses.csv` is **synthetic** — hand-built from templates in
Dutch, German, and English purely to exercise the pipeline. It contains no real
survey data. This mirrors, in miniature, the multilingual value-classification
work from my research, without exposing any real participant responses.

## Background

This is a portfolio-scale distillation of methods from my PhD research in
economic geography, where I classify multilingual open-text responses into value
types and validate LLM-based classification against expert-annotated subsets.

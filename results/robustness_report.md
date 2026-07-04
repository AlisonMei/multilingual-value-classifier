# Robustness / sanity tests

How the real classifier compares to trivial baselines, and how it responds when deliberately crippled.

## Accuracy vs. baselines
| method | accuracy | Cohen's kappa |
|---|---|---|
| **Real classifier** | **0.880** | **0.819** |
| Majority-class baseline | 0.357 | 0.000 |
| Random baseline | 0.313 | -0.033 |

The real classifier (0.88) sits well above the majority-class floor (0.36) and random guessing (0.31). The high score is therefore informative, not an artefact of an easy label distribution.

## Ablation: remove most 'instrumental' cues on purpose
| | instrumental recall |
|---|---|
| full classifier | 1.00 |
| cues removed | 0.15 |

Recall for the crippled class falls from 1.00 to 0.15. The validation harness responds to a real change in the classifier, which is evidence that it is measuring agreement rather than returning a fixed number.

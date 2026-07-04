# Validation report: classifier vs. human gold standard

Overall accuracy: 0.880
Cohen's kappa (human-machine agreement): 0.819

## Per-class performance
| value type | precision | recall | F1 | support |
|---|---|---|---|---|
| intrinsic | 1.00 | 0.66 | 0.80 | 107 |
| instrumental | 1.00 | 1.00 | 1.00 | 87 |
| relational | 0.75 | 1.00 | 0.85 | 106 |

## Per-language accuracy (bias check)
| language | accuracy |
|---|---|
| de | 0.904 |
| en | 0.873 |
| nl | 0.870 |

## Confusion matrix (rows = gold, cols = predicted)
| gold \ pred | intrinsic | instrumental | relational |
|---|---|---|---|
| intrinsic | 71 | 0 | 36 |
| instrumental | 0 | 87 | 0 |
| relational | 0 | 0 | 106 |

_Note: relational values typically show lower recall than instrumental ones, because they are the hardest class to identify. Reporting per-class and per-language metrics makes that visible rather than hiding it in an aggregate score._
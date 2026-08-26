# Final Human Validation Summary v2

Status: **FINAL POST-ADJUDICATION**

The final human labels follow the frozen rule: shared binary A/B labels for non-queued items and blinded adjudicator labels for all 63 queued items. No majority vote or semantic-judge tie-breaking was used.

## Adjudication validation

- Completed responses: 63 rows, 63 unique expected IDs, no missing/unexpected/duplicate IDs, and no invalid labels.
- The completed CSV has one unnamed trailing column with all cells empty; it was ignored as serialization padding without modifying the source.
- Frozen blinded queue checksum verified: `c295e0029773bb274bd71286e9f9c7f6e0224651a7d1c98b08f85f3f8ba0608b`.
- Adjudicator labels: CORRECT=30, INCORRECT=31, UNCERTAIN=2.
- Adjudicated UNCERTAIN by dataset: E2E-WTQ=1, FeTaQA=0, OTT-QA=1.
- Among 62 A/B disagreements: matched A=23, matched B=37, matched neither=2; of the matched-neither cases, adjudication was UNCERTAIN in 0. These counts do not rank annotators.

## Final human–semantic-judge validation

| Dataset | Sampled N | Binary eligible N | UNCERTAIN excluded | Agreement | Cohen's kappa | Agreement 95% CI | Kappa 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| e2ewtq | 200 | 199 | 1 | 0.9095 | 0.7848 | [0.8693, 0.9497] | [0.6881, 0.8741] |
| feta | 200 | 200 | 0 | 0.8050 | 0.5609 | [0.7500, 0.8600] | [0.4357, 0.6766] |
| ottqa | 200 | 199 | 1 | 0.8744 | 0.7133 | [0.8291, 0.9196] | [0.6084, 0.8099] |
| overall | 600 | 598 | 2 | 0.8629 | 0.7230 | [0.8344, 0.8896] | [0.6667, 0.7763] |

## Corrected pre-adjudication A/B reliability (carried forward, not recomputed)

| Dataset | N | Raw agreement | Cohen's kappa | Agreement 95% CI | Kappa 95% CI |
|---|---:|---:|---:|---:|---:|
| e2ewtq | 200 | 0.9450 | 0.8793 | [0.9100, 0.9750] | [0.8026, 0.9441] |
| feta | 200 | 0.8100 | 0.5992 | [0.7550, 0.8600] | [0.4845, 0.7046] |
| ottqa | 200 | 0.9350 | 0.8410 | [0.9000, 0.9650] | [0.7529, 0.9180] |
| overall | 600 | 0.8967 | 0.7963 | [0.8717, 0.9200] | [0.7471, 0.8418] |

The corrected pre-adjudication set required 63 adjudications: 62 exact A/B disagreements plus one matching UNCERTAIN pair. Adjudicated labels were not used to recompute inter-annotator reliability.

## Limitations

Final primary validation uses the frozen canonical stimulus. Twenty-four primary items were independently re-annotated before adjudication. The separate set of 19 diagnostic items is outside the finalized validation scope and is not reported. No significance claim or qualitative kappa label is made.

## Output checksums

- `final_primary_human_labels_v2.csv`: `3ba742e02c604f940136dfd6cccfdcf1a41b7d3b5239e687ef0fa4cd8b63e9a1`
- `final_human_judge_metrics_v2.csv`: `4930ec89cd2087552178fa8dfd7d9f6933b0711ab8cfa757e0eb7efeb4ceb059`
- `final_human_judge_confusion_matrices_v2.csv`: `09a1748a9c0ff7f79535085e9c7d4480e214b3104b57948ea0586e9f672e3b0f`

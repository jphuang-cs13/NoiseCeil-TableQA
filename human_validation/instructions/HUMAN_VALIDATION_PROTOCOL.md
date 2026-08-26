# Human Validation Protocol

Status: **ready for annotation; no human labels collected**

## Purpose and blinding

The primary question is how well the frozen GPT-OSS-20b semantic-judge decisions agree with blinded human factual-equivalence judgments. Humans always judge the original full candidate answer. Annotator-facing materials contain only opaque sample ID, question, reference answer, and original candidate; they omit dataset, provenance, tables, experimental metadata, judge labels, RAW/EXTRACTION metrics, and extracted candidates.

## Frozen samples

Primary sampling uses seed 42. Within each dataset, 200 distinct stable question identities are sampled uniformly without replacement; one of each question's 444 canonical records is then selected uniformly. There is no outcome stratification. Only this 600-item representative sample supports population-level estimates.

The separate 200-item, nonrepresentative diagnostic sample uses seed 42 and contains 50 distinct-question records in each pre-registered stratum: E2E extraction-rescued, E2E persistent disagreement, OTT extraction-rescued, and OTT persistent disagreement. Exact primary items are excluded. Diagnostic results are conditional counts, never population estimates.

All 800 unique items receive opaque IDs and are mixed using annotation-order seed 314159. Item overlap is zero.

## Annotation and adjudication

Two English-proficient annotators independently label every item `CORRECT`, `INCORRECT`, or `UNCERTAIN` using `docs/HUMAN_ANNOTATION_GUIDE.md`. They do not see each other's responses.

Matching binary labels become final. Binary disagreement or either use of `UNCERTAIN` triggers adjudication. The adjudicator initially sees only question, reference, and original candidate and returns `CORRECT`, `INCORRECT`, or `UNRESOLVABLE`. Unresolvable items remain in sample totals and are explicitly excluded only from binary denominators.

## Pre-registered primary analysis

For each dataset, report sampled N=200, binary-resolvable N, UNRESOLVABLE count, oriented semantic-judge/final-human confusion matrix, percent agreement, and Cohen's kappa. Agreement and kappa receive seed-42, 2,000-replicate, question-cluster percentile 95% intervals using Hyndman-Fan type 7. One record per sampled question makes this equivalent to item resampling while retaining cluster terminology.

Before adjudication, report per dataset and overall: exact three-label annotator agreement, binary raw agreement and Cohen's kappa where both labels are binary, count with one/both `UNCERTAIN`, and binary-disagreement count.

## Secondary representative comparisons

On the same resolvable primary items, compare final human labels separately with frozen semantic judge, RAW official binary metric, and EXTRACTION official binary metric. E2E uses RAW/EXTRACTION denotation; OTT uses RAW/EXTRACTION EM. Report N, agreement, kappa, and confusion matrices without selecting a winner. Do not threshold OTT F1. FeTaQA has no item BLEU-human comparison.

## Diagnostic analysis

For each of the four diagnostic strata, report sampled N, resolvable N, human `CORRECT`, `INCORRECT`, and `UNRESOLVABLE` counts. For both rescued and persistent judge-positive strata, descriptively report how often resolvable humans support the judge-positive label. Never combine strata into a representative estimate.

## Sample-size rationale

N=200 per dataset gives a worst-case normal-approximation 95% margin of error of approximately ±6.9 percentage points for binary agreement. This is a practical validation sample, not a formal power calculation. Fifty diagnostic cases per stratum support conditional/qualitative interpretation rather than population precision.

## Frozen workflow

1. Give separate copies of `human_validation/annotation_template.csv` and the blinded workload to two independent annotators.
2. Validate each completed response with `code.validate_human_annotations.validate_response`.
3. Generate a blinded adjudication queue for disagreements/uncertain items only.
4. Validate adjudication labels and compute the pre-registered analyses without changing samples.

Samples must not be regenerated or altered after viewing any human label.

# Human Factual-Equivalence Annotation Guide

Version: `human_annotation_guide_v1`  
Annotators: two independent annotators per item

## Task

Judge whether the original candidate answer correctly answers the question and is factually equivalent to, or clearly entails, the provided reference answer. Use only the question, reference answer, and candidate answer. Do not seek retrieved tables, provenance, model information, metrics, or external evidence to override the reference.

Do not judge style, fluency, verbosity, or reasoning quality except when they affect factual answer correctness.

Choose exactly one initial label:

- `CORRECT`
- `INCORRECT`
- `UNCERTAIN`

## Label definitions

### CORRECT

The candidate supplies all materially required answer content without contradiction. Accept paraphrases, harmless formatting or ordering differences, equivalent numeric/date representations, explanatory wrapping, and relevant extra context that does not change the answer.

If the correct answer is embedded in a longer response and the surrounding material neither contradicts nor materially alters it, label `CORRECT`.

### INCORRECT

The candidate gives a different answer, contradicts the reference, fails or refuses to answer, is empty when an answer is required, gives a materially incomplete answer, or adds contradictory/unsupported content that changes the requested answer.

If correct text appears but surrounding material contradicts it or substitutes another answer, label `INCORRECT`.

### UNCERTAIN

Use only when correctness cannot be determined reliably from the three provided fields. Do not use it merely because a response is long, unusual, poorly formatted, or contains imperfect reasoning.

## Required policies

- All materially required elements of a multi-part or list reference must be satisfied.
- One part of a multi-part answer is `INCORRECT` unless the provided reference/question clearly makes that part sufficient.
- Numerically equivalent forms are acceptable, including ordinary percentage, fraction, decimal, and unit conversions when equivalence is clear from the supplied text.
- Equivalent date formats are acceptable.
- Extra alternatives are acceptable only if they do not contradict or materially weaken the required answer.
- Contradictory alternatives make the answer `INCORRECT`.
- Do not use outside knowledge to replace or overrule the reference.

## Synthetic examples

| Case | Question | Reference | Candidate | Label | Rationale |
| --- | --- | --- | --- | --- | --- |
| Exact match | What city is named? | Paris | Paris | CORRECT | Exact answer. |
| Paraphrase | What did the team do? | won the match | They were victorious in the match. | CORRECT | Clear factual equivalence. |
| Case/format | What code is shown? | abc-12 | ABC-12 | CORRECT | Harmless case change. |
| Numeric equivalence | What share? | 25% | One quarter. | CORRECT | Equivalent quantity. |
| Date equivalence | On what date? | 2020-01-05 | January 5, 2020 | CORRECT | Equivalent date. |
| Embedded answer | What is the capital? | Paris | The answer is Paris, based on the supplied information. | CORRECT | Correct answer in explanation. |
| Harmless context | What is the capital? | Paris | Paris is the answer. Paris is the capital of France. | CORRECT | Context does not contradict. |
| Later contradiction | What is the capital? | Paris | The answer is Paris, although the actual answer is London. | INCORRECT | Surrounding text contradicts. |
| Partial answer | Name both colors. | red and blue | Red. | INCORRECT | Missing required element. |
| Complete multi-answer | Name both colors. | red and blue | Blue and red. | CORRECT | All elements, different order. |
| Contradictory alternative | Which city? | Paris | Paris or perhaps London. | INCORRECT | Alternative changes the answer. |
| Refusal | Which city? | Paris | I cannot answer that. | INCORRECT | Fails to answer. |
| Empty output | Which city? | Paris | *(empty)* | INCORRECT | Required answer absent. |
| Ambiguous reference | Which bank? | Mercury | The planet Mercury. | UNCERTAIN | Provided fields do not establish whether “bank” and response refer to the same entity. |
| Unresolvable evidence | What was the unnamed value? | it | It was the other one. | UNCERTAIN | The supplied fields are insufficient. |

The examples are synthetic and are not sampled project cases.

## Independence and notes

Annotate independently and do not view another annotator's labels. Optional notes should briefly identify the factual issue or ambiguity; they are not a substitute for the required label.

Items with annotator disagreement or at least one `UNCERTAIN` label proceed to adjudication. The adjudicator sees only the same three text fields and assigns `CORRECT`, `INCORRECT`, or `UNRESOLVABLE`.

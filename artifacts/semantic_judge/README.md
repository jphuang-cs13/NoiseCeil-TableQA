# Semantic-Judge Publication Evidence

Semantic-judge per-query verdict/evaluation records and their binding manifest are intentionally excluded from the public release.

Released evidence consists of:

- exact judge prompt provenance under `human_validation/prompts/`;
- judge configuration: GPT-OSS-20b, temperature 0.0, top-p 1.0, maximum new tokens 10;
- aggregate token usage in `token_usage_aggregates.csv`;
- aggregate benchmark and human-validation results under `artifacts/appendix/` and `human_validation/results/`;
- camera-ready aggregate Score/SD/NRR and CpS outputs.

Raw reasoning, questions, references, reader responses, provider identifiers, account/request metadata, timing, private annotations, and local paths are not included.

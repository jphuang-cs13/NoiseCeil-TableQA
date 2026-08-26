#!/usr/bin/env python3
"""Build the canonical long-form plotting CSV from official rule_order_v2 cases."""

import csv
import json
from collections import Counter
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "artifacts" / "error_analysis"
INPUT = OFFICIAL / "classifications_sanitized.jsonl"
OUTPUT = OFFICIAL / "Error Analysis - Error Type_v2.csv"

DATASETS = ["e2ewtq", "ottqa", "feta"]
MODELS = ["qwen3-32b", "gpt-oss-20b", "claude-haiku-4-5", "gpt-4o"]
NEGATIVES = ["hard", "soft"]
QUESTION_TYPES = ["Lookup", "Comparison", "Aggregation"]
ERROR_TYPES = [
    "Distractor Extraction",
    "Premature Refusal",
    "Reasoning Hallucination",
]
FIELDS = [
    "dataset", "model", "negative", "question_type", "error_type", "count",
    "percentage (%)",
]


def main() -> None:
    cases = [json.loads(line) for line in INPUT.open(encoding="utf-8") if line.strip()]
    if len(cases) != 36_636:
        raise RuntimeError(f"Expected 36,636 official cases, found {len(cases):,}")
    if any(case.get("classification_policy") != "rule_order_v2" for case in cases):
        raise RuntimeError("Input contains a classification policy other than rule_order_v2")
    if any(case.get("final_error_type") not in ERROR_TYPES for case in cases):
        raise RuntimeError("Input contains an invalid final_error_type")

    counts = Counter(
        (case["dataset"], case["model"], case["negative"],
         case["question_type"], case["final_error_type"])
        for case in cases
    )
    denominators = Counter(
        (case["dataset"], case["model"], case["negative"])
        for case in cases
    )

    rows = []
    for dataset, model, negative, question_type, error_type in product(
        DATASETS, MODELS, NEGATIVES, QUESTION_TYPES, ERROR_TYPES
    ):
        count = counts[(dataset, model, negative, question_type, error_type)]
        denominator = denominators[(dataset, model, negative)]
        rows.append({
            "dataset": dataset,
            "model": model,
            "negative": negative,
            "question_type": question_type,
            "error_type": error_type,
            "count": count,
            "percentage (%)": round(count / denominator * 100, 6) if denominator else 0.0,
        })

    if sum(row["count"] for row in rows) != len(cases):
        raise RuntimeError("Aggregated counts do not reproduce the official case total")
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows representing {len(cases):,} cases to {OUTPUT}")


if __name__ == "__main__":
    main()

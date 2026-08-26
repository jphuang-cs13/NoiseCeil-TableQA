#!/usr/bin/env python3
"""Generate paper-facing conditional cross-tabs from official rule_order_v2 results."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "artifacts" / "error_analysis"
INPUT = OFFICIAL / "classifications_sanitized.jsonl"
CSV_OUTPUT = OFFICIAL / "paper_error_analysis_crosstabs.csv"
JSON_OUTPUT = OFFICIAL / "paper_error_analysis_crosstabs.json"

LABELS = ["Distractor Extraction", "Premature Refusal", "Reasoning Hallucination"]
SHORT = {
    "Distractor Extraction": "DE",
    "Premature Refusal": "PR",
    "Reasoning Hallucination": "RH",
}
DATASETS = ["e2ewtq", "feta", "ottqa"]
READERS = ["qwen3-32b", "gpt-oss-20b", "claude-haiku-4-5", "gpt-4o"]
QUESTION_TYPES = ["Lookup", "Comparison", "Aggregation"]
NEGATIVES = ["hard", "soft"]


def load_official_rows():
    rows = [json.loads(line) for line in INPUT.open(encoding="utf-8") if line.strip()]
    if len(rows) != 36636:
        raise RuntimeError(f"Expected 36,636 classified cases, found {len(rows)}")
    invalid_policy = [row for row in rows if row.get("classification_policy") != "rule_order_v2"]
    if invalid_policy:
        raise RuntimeError(f"Found {len(invalid_policy)} non-v2 records")
    valid_labels = set(LABELS)
    invalid_labels = [row for row in rows if row.get("final_error_type") not in valid_labels]
    if invalid_labels:
        raise RuntimeError(f"Found {len(invalid_labels)} invalid labels")
    return rows


def row_normalized(rows):
    total = len(rows)
    counts = Counter(row["final_error_type"] for row in rows)
    result = {"total": total}
    for label in LABELS:
        count = counts[label]
        result[SHORT[label]] = {
            "count": count,
            "pct": round(count / total * 100, 6) if total else 0.0,
        }
    return result


def grouped(rows, dimensions, ordered_values):
    result = []
    for values in ordered_values:
        if not isinstance(values, tuple):
            values = (values,)
        subset = [
            row for row in rows
            if all(str(row[dimension]) == str(value) for dimension, value in zip(dimensions, values))
        ]
        result.append({**dict(zip(dimensions, values)), **row_normalized(subset)})
    return result


def composition(rows, negative_scope):
    scoped = rows if negative_scope == "all" else [row for row in rows if row["negative"] == negative_scope]
    result = []
    for label in LABELS:
        subset = [row for row in scoped if row["final_error_type"] == label]
        total = len(subset)
        counts = Counter(row["question_type"] for row in subset)
        entry = {"negative_scope": negative_scope, "error_type": label, "total": total}
        for question_type in QUESTION_TYPES:
            count = counts[question_type]
            entry[question_type] = {
                "count": count,
                "pct": round(count / total * 100, 6) if total else 0.0,
            }
        result.append(entry)
    return result


def flatten_sections(sections):
    flat = []
    for section, entries in sections.items():
        for entry in entries:
            base = {
                "section": section,
                "dataset": entry.get("dataset", ""),
                "reader": entry.get("model", ""),
                "negative": entry.get("negative", entry.get("negative_scope", "")),
                "question_type": entry.get("question_type", ""),
                "error_type": entry.get("error_type", ""),
                "total": entry["total"],
            }
            if "DE" in entry:
                for short in ("DE", "PR", "RH"):
                    base[f"{short}_count"] = entry[short]["count"]
                    base[f"{short}_pct"] = entry[short]["pct"]
                for question_type in QUESTION_TYPES:
                    base[f"{question_type}_count"] = ""
                    base[f"{question_type}_pct"] = ""
            else:
                for short in ("DE", "PR", "RH"):
                    base[f"{short}_count"] = ""
                    base[f"{short}_pct"] = ""
                for question_type in QUESTION_TYPES:
                    base[f"{question_type}_count"] = entry[question_type]["count"]
                    base[f"{question_type}_pct"] = entry[question_type]["pct"]
            flat.append(base)
    return flat


def main():
    rows = load_official_rows()
    sections = {
        "negative_x_error_type": grouped(rows, ["negative"], NEGATIVES),
        "dataset_x_negative_x_error_type": grouped(
            rows, ["dataset", "negative"],
            [(dataset, negative) for dataset in DATASETS for negative in NEGATIVES],
        ),
        "reader_x_negative_x_error_type": grouped(
            rows, ["model", "negative"],
            [(reader, negative) for reader in READERS for negative in NEGATIVES],
        ),
        "question_type_x_negative_x_error_type": grouped(
            rows, ["question_type", "negative"],
            [(question_type, negative) for question_type in QUESTION_TYPES for negative in NEGATIVES],
        ),
        "lookup_hard_x_reader_x_error_type": grouped(
            [row for row in rows if row["question_type"] == "Lookup" and row["negative"] == "hard"],
            ["model"], READERS,
        ),
        "error_type_composition_by_question_type": (
            composition(rows, "all") + composition(rows, "hard") + composition(rows, "soft")
        ),
    }
    payload = {
        "metadata": {
            "source": str(INPUT.relative_to(ROOT)),
            "classification_policy": "rule_order_v2",
            "classified_cases": len(rows),
            "excluded_source_provider_failures": 140,
            "percentage_definition": "Row-normalized within each reported conditioning group; composition tables normalize within error type and negative scope.",
        },
        "sections": sections,
    }
    JSON_OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    flat = flatten_sections(sections)
    with CSV_OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(JSON_OUTPUT)
    print(CSV_OUTPUT)
    print(json.dumps(sections, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

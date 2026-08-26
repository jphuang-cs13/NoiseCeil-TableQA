#!/usr/bin/env python3
"""
Error Analysis Stage 5: Question + Error Classification CSV

Purpose:
========
Aggregate all Stage 3 error-classification JSON files for a dataset into one CSV.
The script automatically discovers files like:
- {model}_K{K}_{negative}_{category}_errors.json

It then looks up the original question text from the matching reasoning results
and writes one flat row per classified query.

Output:
=======
- error_analysis/{dataset}/question_error_table.csv
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from collections import Counter


ERROR_CATEGORIES = ["both_success", "both_fail", "perfect_to_fail", "perfect_from_fail"]
NEGATIVES = ["hard", "soft"]


def parse_error_filename(filename: str) -> Optional[Tuple[str, int, str, str]]:
    """Parse a Stage 3 error filename into model, K, negative, and category."""
    match = re.match(
        r"^(?P<model>.+)_K(?P<k>\d+)_(?P<negative>hard|soft)_(?P<category>both_success|both_fail|perfect_to_fail|perfect_from_fail)_errors\.json$",
        filename,
    )
    if not match:
        return None
    return (
        match.group("model"),
        int(match.group("k")),
        match.group("negative"),
        match.group("category"),
    )


def load_json(path: Path) -> Dict:
    """Load JSON content from a file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def load_reasoning_questions(reasoning_root: Path, dataset: str) -> Dict[int, str]:
    """Build a query_id -> question lookup from all reasoning JSONL files for a dataset."""
    question_map: Dict[int, str] = {}
    dataset_dir = reasoning_root / dataset
    if not dataset_dir.is_dir():
        return question_map

    for jsonl_file in dataset_dir.rglob("*.jsonl"):
        try:
            with jsonl_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    query_id = record.get("query_id")
                    question = record.get("question")
                    if query_id is None or not question:
                        continue

                    try:
                        question_map.setdefault(int(query_id), question)
                    except (TypeError, ValueError):
                        continue
        except Exception as exc:
            print(f"  ⚠ Failed to read reasoning file {jsonl_file}: {exc}")

    return question_map


def iter_error_files(dataset_dir: Path) -> Iterable[Path]:
    """Yield Stage 3 error JSON files for a dataset."""
    yield from sorted(dataset_dir.glob("*_errors.json"))


def build_rows(dataset: str, dataset_dir: Path, question_map: Dict[int, str]) -> List[Dict[str, object]]:
    """Build flat CSV rows from all Stage 3 outputs."""
    rows: List[Dict[str, object]] = []

    for error_file in iter_error_files(dataset_dir):
        parsed = parse_error_filename(error_file.name)
        if not parsed:
            continue

        model, target_k, negative, category = parsed

        try:
            data = load_json(error_file)
        except Exception as exc:
            print(f"  ⚠ Failed to load {error_file.name}: {exc}")
            continue

        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        metadata_dataset = metadata.get("dataset")
        if metadata_dataset and metadata_dataset != dataset:
            print(f"  ⚠ Skipping {error_file.name}: metadata dataset mismatch ({metadata_dataset})")
            continue

        classifications = data.get("classifications", [])
        if not isinstance(classifications, list):
            continue

        for item in classifications:
            if not isinstance(item, dict):
                continue

            query_id = item.get("query_id")
            try:
                query_id_int = int(query_id)
            except (TypeError, ValueError):
                query_id_int = None

            question = question_map.get(query_id_int, "") if query_id_int is not None else ""

            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "target_k": target_k,
                    "negative": negative,
                    "category": category,
                    "query_id": query_id,
                    "question": question,
                    "error_type": item.get("final_error_type") or item.get("error_type", "Unknown"),
                    "final_error_type": item.get("final_error_type") or item.get("error_type", "Unknown"),
                    "label_source": item.get("label_source", ""),
                    "matched_distractor_table": item.get("matched_distractor_table", ""),
                    "matched_distractor_cell": item.get("matched_distractor_cell", ""),
                    "matched_refusal_phrase": item.get("matched_refusal_phrase", ""),
                    "error_confidence": item.get("llm_confidence", item.get("error_confidence", 0.0)),
                    "llm_diagnostic_type": item.get("llm_diagnostic_type", "Unknown"),
                    "question_type": item.get("question_type", "Unknown"),
                    "question_confidence": item.get("question_confidence", 0.0),
                    "source_file": error_file.name,
                }
            )

    rows.sort(key=lambda row: (str(row["model"]), int(row["target_k"]), str(row["negative"]), str(row["category"]), str(row["query_id"])))
    return rows


def write_csv(rows: List[Dict[str, object]], output_file: Path) -> None:
    """Write the aggregated table to CSV."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "model",
        "target_k",
        "negative",
        "category",
        "query_id",
        "question",
        "error_type",
        "final_error_type",
        "label_source",
        "matched_distractor_table",
        "matched_distractor_cell",
        "matched_refusal_phrase",
        "error_confidence",
        "llm_diagnostic_type",
        "question_type",
        "question_confidence",
        "source_file",
    ]

    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_question_type_summary(rows: List[Dict[str, object]], output_file: Path) -> None:
    labels = ["Distractor Extraction", "Premature Refusal", "Reasoning Hallucination"]
    counts = Counter((str(row["question_type"]), str(row["final_error_type"])) for row in rows)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f:
        fields = ["question_type", "total"] + [f"{label}_count" for label in labels] + [f"{label}_pct" for label in labels]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for question_type in ["Lookup", "Comparison", "Aggregation", "Unknown"]:
            total = sum(counts[(question_type, label)] for label in labels)
            if not total: continue
            row = {"question_type": question_type, "total": total}
            for label in labels:
                row[f"{label}_count"] = counts[(question_type, label)]
                row[f"{label}_pct"] = round(counts[(question_type, label)] / total * 100, 4)
            writer.writerow(row)


def print_preview(rows: List[Dict[str, object]]) -> None:
    """Print a short preview of the aggregated table."""
    if not rows:
        print("  No stage 3 files found.")
        return

    print(f"  Rows written: {len(rows)}")
    print("  Preview:")
    for row in rows[:5]:
        question = row["question"] or "<question not found>"
        if len(question) > 90:
            question = question[:87] + "..."
        print(
            f"    - {row['model']} | K={row['target_k']} | {row['negative']} | {row['category']} | "
            f"qid={row['query_id']} | {row['error_type']} | {question}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Error Analysis Stage 5: Question + Error Classification CSV"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., e2ewtq, feta, ottqa)")
    parser.add_argument(
        "--error-root",
        default="error_analysis/",
        help="Root directory containing error_analysis outputs (default: error_analysis/)",
    )
    parser.add_argument("--summary-file", default=None, help="Optional question-type by final-error summary CSV")
    parser.add_argument(
        "--reasoning-root",
        default="reasoning_results/",
        help="Root directory containing reasoning results (default: reasoning_results/)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional explicit CSV output path. Defaults to error_analysis/{dataset}/question_error_table.csv",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.error_root) / args.dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    output_file = Path(args.output_file) if args.output_file else dataset_dir / "question_error_table.csv"
    reasoning_root = Path(args.reasoning_root)

    print("[Stage 5] Question + Error Classification CSV")
    print(f"  Dataset: {args.dataset}")
    print(f"  Input dir: {dataset_dir}")
    print(f"  Reasoning root: {reasoning_root}")
    print(f"  Output file: {output_file}")

    question_map = load_reasoning_questions(reasoning_root, args.dataset)
    print(f"  Loaded questions: {len(question_map)}")

    rows = build_rows(args.dataset, dataset_dir, question_map)
    if not rows:
        print("  ✗ No stage 3 JSON files found to summarize.")
        return

    write_csv(rows, output_file)
    summary_file = Path(args.summary_file) if args.summary_file else output_file.with_name("question_type_error_summary.csv")
    write_question_type_summary(rows, summary_file)
    print_preview(rows)
    print(f"\n✓ Question/error table saved to: {output_file}")
    print(f"✓ Question-type/error summary saved to: {summary_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Error Analysis Stage 4: Distribution Summary CSV

Purpose:
========
Scan all Stage 1 distribution JSON files under a dataset's error_analysis directory
and aggregate them into a CSV summary table.

The script automatically detects which models and target K values were run by
parsing filenames like:
- {model}_K{K}_distribution.json

Output:
=======
- error_analysis/{dataset}/distribution_summary.csv

Each row corresponds to one distribution file and one negative type (hard/soft).
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CATEGORY_ORDER = ["both_success", "both_fail", "perfect_to_fail", "perfect_from_fail"]
NEGATIVE_ORDER = ["hard", "soft"]


def parse_distribution_filename(filename: str) -> Optional[Tuple[str, int]]:
    """Parse a Stage 1 distribution filename into model and target K."""
    match = re.match(r"^(?P<model>.+)_K(?P<k>\d+)_distribution\.json$", filename)
    if not match:
        return None
    return match.group("model"), int(match.group("k"))


def load_json(path: Path) -> Dict:
    """Load JSON content from a file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def build_row(
    *,
    dataset: str,
    model: str,
    target_k: int,
    negative: str,
    dist_file: str,
    negative_distribution: Dict[str, List[Dict]],
) -> Dict[str, object]:
    """Build a single CSV row for one model/K/negative combination."""
    counts = {category: len(negative_distribution.get(category, [])) for category in CATEGORY_ORDER}
    total = sum(counts.values())

    row: Dict[str, object] = {
        "dataset": dataset,
        "model": model,
        "target_k": target_k,
        "negative": negative,
        "total": total,
        "distribution_file": dist_file,
    }

    for category in CATEGORY_ORDER:
        count = counts[category]
        row[f"{category}_count"] = count
        row[f"{category}_pct"] = round((count / total * 100.0), 2) if total else 0.0

    return row


def build_rows(dataset: str, dataset_dir: Path) -> List[Dict[str, object]]:
    """Build CSV rows from all distribution JSON files in the dataset directory."""
    rows: List[Dict[str, object]] = []

    for dist_file in sorted(dataset_dir.glob("*_distribution.json")):
        parsed = parse_distribution_filename(dist_file.name)
        if not parsed:
            continue

        filename_model, filename_k = parsed

        try:
            data = load_json(dist_file)
        except Exception as exc:
            print(f"  ⚠ Failed to load {dist_file.name}: {exc}")
            continue

        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        metadata_dataset = metadata.get("dataset")
        if metadata_dataset and metadata_dataset != dataset:
            print(f"  ⚠ Skipping {dist_file.name}: metadata dataset mismatch ({metadata_dataset})")
            continue

        model = metadata.get("model") or filename_model
        target_k = metadata.get("target_k") or filename_k

        distribution = data.get("distribution", {})
        if not isinstance(distribution, dict):
            distribution = {}

        for negative in NEGATIVE_ORDER:
            negative_distribution = distribution.get(negative, {})
            if not isinstance(negative_distribution, dict):
                negative_distribution = {}

            rows.append(
                build_row(
                    dataset=dataset,
                    model=model,
                    target_k=target_k,
                    negative=negative,
                    dist_file=dist_file.name,
                    negative_distribution=negative_distribution,
                )
            )

    rows.sort(key=lambda row: (str(row["model"]), int(row["target_k"]), str(row["negative"])))
    return rows


def write_csv(rows: List[Dict[str, object]], output_file: Path) -> None:
    """Write rows to a CSV file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "model",
        "target_k",
        "negative",
        "total",
        "both_success_count",
        "both_success_pct",
        "both_fail_count",
        "both_fail_pct",
        "perfect_to_fail_count",
        "perfect_to_fail_pct",
        "perfect_from_fail_count",
        "perfect_from_fail_pct",
        "distribution_file",
    ]

    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_preview(rows: List[Dict[str, object]]) -> None:
    """Print a compact text preview of the summary."""
    if not rows:
        print("  No distribution files found.")
        return

    print(f"  Rows written: {len(rows)}")
    print("  Preview:")
    for row in rows[:5]:
        print(
            f"    - {row['model']} | K={row['target_k']} | {row['negative']} | total={row['total']} | "
            f"both_success={row['both_success_count']} | both_fail={row['both_fail_count']} | "
            f"perfect_to_fail={row['perfect_to_fail_count']} | perfect_from_fail={row['perfect_from_fail_count']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Error Analysis Stage 4: Distribution Summary CSV"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., e2ewtq, feta, ottqa)")
    parser.add_argument(
        "--error-root",
        default="error_analysis/",
        help="Root directory containing error_analysis outputs (default: error_analysis/)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional explicit CSV output path. Defaults to error_analysis/{dataset}/distribution_summary.csv",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.error_root) / args.dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    output_file = Path(args.output_file) if args.output_file else dataset_dir / "distribution_summary.csv"

    print("[Stage 4] Distribution Summary CSV")
    print(f"  Dataset: {args.dataset}")
    print(f"  Input dir: {dataset_dir}")
    print(f"  Output file: {output_file}")

    rows = build_rows(args.dataset, dataset_dir)
    if not rows:
        print("  ✗ No distribution JSON files found to summarize.")
        return

    write_csv(rows, output_file)
    print_preview(rows)
    print(f"\n✓ Distribution summary saved to: {output_file}")


if __name__ == "__main__":
    main()
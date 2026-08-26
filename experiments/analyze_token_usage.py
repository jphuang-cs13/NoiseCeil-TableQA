#!/usr/bin/env python3
"""
Analyze token usage (request_tokens, response_tokens) from evaluation results.

This script processes evaluation result files in evaluations/[dataset]_test/injection/
and calculates token usage statistics (total, mean, median, min, max) for each 
model, rotation, and negative type combination.

Usage:
    python3 experiments/analyze_token_usage.py <dataset>

Example:
    python3 experiments/analyze_token_usage.py bird
    python3 experiments/analyze_token_usage.py ottqa
"""

import json
import sys
from pathlib import Path
from typing import Tuple, Dict, List
import statistics
import csv


def parse_injection_filename(name: str) -> Tuple[int, str, str]:
    """
    Parse injection filename to extract recall@K (K), rotation, and negative type.

    Examples:
    - 1_perfect.jsonl -> (1, "", "")
    - 2_end_hard.jsonl -> (2, "end", "hard")
    - 4_middle_soft.jsonl -> (4, "middle", "soft")
    """
    base = name.rsplit(".", 1)[0]
    parts = base.split("_")
    if len(parts) == 1 and parts[0] == "1":
        return 1, "", ""
    if base == "1_perfect":
        return 1, "", ""
    try:
        k = int(parts[0])
    except Exception:
        k = 0
    rotation = parts[1] if len(parts) > 1 else ""
    negative = parts[2] if len(parts) > 2 else ""
    return k, rotation, negative


def extract_token_data(file_path: Path) -> Tuple[List[int], List[int]]:
    """
    Extract request_tokens and response_tokens from evaluation result file.
    
    File structure: single JSON object with 'records' array containing query results.
    
    Args:
        file_path: Path to evaluation result JSON file
    
    Returns:
        Tuple of (request_tokens_list, response_tokens_list)
    """
    request_tokens = []
    response_tokens = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract tokens from records array
        if isinstance(data, dict) and 'records' in data:
            for record in data['records']:
                if 'request_tokens' in record:
                    request_tokens.append(record['request_tokens'])
                if 'response_tokens' in record:
                    response_tokens.append(record['response_tokens'])
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {file_path}: {e}")
    
    return request_tokens, response_tokens


def compute_stats(values: List[int]) -> Dict[str, float]:
    """
    Compute statistics for a list of values.
    
    Args:
        values: List of numeric values
    
    Returns:
        Dict with keys: total, mean, median, min, max
    """
    if not values:
        return {"total": 0, "mean": 0, "median": 0, "min": 0, "max": 0}
    
    return {
        "total": sum(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 experiments/analyze_token_usage.py <dataset>")
        sys.exit(1)
    
    dataset = sys.argv[1]
    base_dir = Path(__file__).parent.parent
    eval_dir = base_dir / "evaluations" / f"{dataset}_test" / "injection"
    
    if not eval_dir.exists():
        print(f"Evaluation directory not found: {eval_dir}")
        sys.exit(1)
    
    # Collect all token data, split by normal vs real_result outputs
    # Key: output_csv -> Dict[(model, k, rotation, negative), stats]
    grouped_data = {}

    def get_output_csv(model_name: str) -> Path:
        return eval_dir / ("token_usage_summary_real_result.csv" if model_name.endswith("_real_result") else "token_usage_summary.csv")

    # Iterate through model folders
    for model_dir in sorted(eval_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        output_csv = get_output_csv(model_name)
        grouped_data.setdefault(output_csv, {})

        print(f"Processing model: {model_name}")

        # Iterate through injection result files
        for result_file in sorted(model_dir.glob("*.jsonl")):
            # Parse filename
            k, rotation, negative = parse_injection_filename(result_file.name)

            # Extract token data
            req_tokens, resp_tokens = extract_token_data(result_file)

            # Compute statistics
            req_stats = compute_stats(req_tokens)
            resp_stats = compute_stats(resp_tokens)

            # Store data
            key = (model_name, k, rotation or "perfect", negative or "")
            grouped_data[output_csv][key] = {
                "request_tokens": req_stats,
                "response_tokens": resp_stats,
            }
            print(f"  {result_file.name}: req_total={req_stats['total']}, resp_total={resp_stats['total']}")

    if not grouped_data:
        print("No data found to write.")
        return

    header = [
        "Models", "K", "Rotation", "Negatives",
        "request_tokens_total", "request_tokens_mean", "request_tokens_median", "request_tokens_min", "request_tokens_max",
        "response_tokens_total", "response_tokens_mean", "response_tokens_median", "response_tokens_min", "response_tokens_max"
    ]

    for output_csv, all_data in sorted(grouped_data.items(), key=lambda item: item[0].name):
        # Check if summary already exists
        if output_csv.exists():
            print(f"Summary already exists: {output_csv}")
            print("Skipping to avoid duplicates. Delete file if you want to regenerate.")
            continue

        if not all_data:
            print(f"No data found to write for {output_csv.name}.")
            continue

        # Write CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            # Data rows (sorted by model, k, rotation, negative)
            for (model, k, rotation, negative), stats in sorted(all_data.items()):
                req = stats["request_tokens"]
                resp = stats["response_tokens"]
                writer.writerow([
                    model, k, rotation, negative,
                    req["total"], f"{req['mean']:.2f}", req["median"], req["min"], req["max"],
                    resp["total"], f"{resp['mean']:.2f}", resp["median"], resp["min"], resp["max"]
                ])

        print(f"\nToken usage summary saved to: {output_csv}")


if __name__ == "__main__":
    main()

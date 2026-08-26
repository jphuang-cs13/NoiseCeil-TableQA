#!/usr/bin/env python3
"""
Error Analysis Stage 2: Feature Distribution Analysis

Purpose:
========
Analyze feature distributions for cases under a specific negative type and
success/failure category:
- Number of retrieved tables (K)
- Gold table position (start, middle, end)
- Cross-tabulated statistics by position and table count

Usage:
======
python3 experiments/error_analysis_stage2_features.py \
  --dataset e2ewtq \
  --model qwen3-32b \
  --target-k 30 \
  --category perfect_to_fail \
  --negative hard \
  --eval-root evaluation_results/ \
  --inject-root injections/ \
  --output-dir error_analysis/

Output:
=======
- error_analysis/qwen3-32b_e2ewtq_K30_hard_perfect_to_fail_features.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_injection_filename(name: str) -> Tuple[int, str, str]:
    """Parse injection filename to extract K, rotation, and negative type."""
    base = name.rsplit(".", 1)[0]
    
    if base.startswith("real_result_"):
        try:
            k = int(base.replace("real_result_", ""))
            return k, "real_result", "real_result"
        except ValueError:
            pass
    
    parts = base.split("_")
    if base == "1_perfect":
        return 1, "perfect", "perfect"
    
    try:
        k = int(parts[0])
    except (ValueError, IndexError):
        return 0, "", ""
    
    rotation = parts[1] if len(parts) > 1 else ""
    negative = parts[2] if len(parts) > 2 else ""
    
    return k, rotation, negative


def load_reasoning_record(injection_file: Path, query_id: str) -> Optional[Dict]:
    """Load a specific reasoning record by query_id from an injection file."""
    if not injection_file.exists():
        return None
    
    try:
        with injection_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("query_id") == query_id:
                        return record
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    
    return None


def extract_features(target_k: int, rotation: str) -> Dict:
    """
    Extract relevant features from target K and rotation.
    
    Features:
    - num_tables: Number of retrieved tables (K value)
    - gold_position: Position of gold table (start/middle/end)
    """
    features = {
        "num_tables": target_k,
        "gold_position": rotation if rotation in ["start", "middle", "end"] else None,
    }
    
    return features


def analyze_features(
    dataset: str,
    model: str,
    target_k: int,
    category: str,
    negative_type: str,
    distribution_json: Path,
    inject_root: Path,
) -> Dict:
    """Analyze feature distribution for a specific category and negative type."""
    
    # Load distribution data
    with distribution_json.open("r", encoding="utf-8") as f:
        dist_data = json.load(f)
    
    query_ids = dist_data["distribution"][negative_type].get(category, [])
    if not query_ids:
        print(f"  ✗ No records found for {negative_type}/{category}")
        return {}
    
    # Group by features
    feature_groups = defaultdict(list)
    
    for record in query_ids:
        qid = record["query_id"]
        rotation = record["rotation"]
        
        # Try multiple possible paths for the injection/reasoning file
        possible_files = [
            # reasoning_results/e2ewtq/qwen3-32b/10_start_hard.jsonl
            inject_root / dataset / model / f"{target_k}_{rotation}_{negative_type}.jsonl",
            # reasoning_results/e2ewtq/qwen3-32b/10_start_hard.json
            inject_root / dataset / model / f"{target_k}_{rotation}_{negative_type}.json",
            # injections/e2ewtq/10_start_hard.jsonl
            inject_root / dataset / f"{target_k}_{rotation}_{negative_type}.jsonl",
            # injections/e2ewtq/10_start_hard.json
            inject_root / dataset / f"{target_k}_{rotation}_{negative_type}.json",
        ]
        
        injection_file = None
        for candidate in possible_files:
            if candidate.exists():
                injection_file = candidate
                break
        
        if not injection_file:
            print(f"    ⚠ Could not load reasoning record for {qid} from {target_k}_{rotation}_{negative_type}.*")
            continue
        
        # Extract features from target_k and rotation
        features = extract_features(target_k, rotation)
        num_tables = features["num_tables"]
        position = features["gold_position"]
        
        # Create a key for grouping
        key = f"K={num_tables},pos={position}"
        feature_groups[key].append({
            "query_id": qid,
            "rotation": rotation,
            "features": features,
        })
    
    return feature_groups


def generate_feature_summary(feature_groups: Dict, category: str, negative_type: str) -> str:
    """Generate a summary of feature distribution."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"Feature Distribution: [{negative_type.upper()}] {category}")
    lines.append("=" * 60)
    
    if not feature_groups:
        lines.append("  No data found.")
        return "\n".join(lines)
    
    total = sum(len(items) for items in feature_groups.values())
    
    # Sort keys for consistent output
    for key in sorted(feature_groups.keys()):
        items = feature_groups[key]
        count = len(items)
        pct = (count / total * 100) if total > 0 else 0
        lines.append(f"  {key:30s}: {count:4d} ({pct:5.1f}%)")
    
    lines.append("-" * 60)
    lines.append(f"  {'TOTAL':30s}: {total:4d} (100.0%)")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Error Analysis Stage 2: Feature Distribution"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--target-k", type=int, default=30, help="Target K")
    parser.add_argument(
        "--category",
        default="perfect_to_fail",
        choices=["both_success", "both_fail", "perfect_to_fail", "perfect_from_fail"],
        help="Category to analyze"
    )
    parser.add_argument(
        "--negative",
        default="hard",
        choices=["hard", "soft"],
        help="Negative type to analyze"
    )
    parser.add_argument(
        "--eval-root",
        default="evaluation_results/",
        help="Evaluation results root"
    )
    parser.add_argument(
        "--inject-root",
        default="reasoning_results/",
        help="Injections/reasoning results root (default: reasoning_results/)"
    )
    parser.add_argument(
        "--output-dir",
        default="error_analysis/",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    eval_root = Path(args.eval_root) / args.dataset
    inject_root = Path(args.inject_root)
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dist_file = output_dir / f"{args.model}_K{args.target_k}_distribution.json"
    
    print(f"\n[Stage 2] Feature Analysis")
    print(f"  Dataset: {args.dataset}, Model: {args.model}, K: {args.target_k}")
    print(f"  Category: {args.category}, Negative: {args.negative}")
    print(f"  Using distribution file: {dist_file}")
    
    if not dist_file.exists():
        print(f"  ✗ Distribution file not found. Run Stage 1 first.")
        return
    
    # Analyze features
    feature_groups = analyze_features(
        args.dataset,
        args.model,
        args.target_k,
        args.category,
        args.negative,
        dist_file,
        inject_root,
    )
    
    # Generate summary
    summary = generate_feature_summary(feature_groups, args.category, args.negative)
    print("\n" + summary)
    
    # Save JSON output
    output_file = output_dir / f"{args.model}_K{args.target_k}_{args.negative}_{args.category}_features.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "dataset": args.dataset,
                "model": args.model,
                "target_k": args.target_k,
                "category": args.category,
                "negative": args.negative,
            },
            "feature_groups": {k: v for k, v in feature_groups.items()}
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Feature analysis saved to: {output_file}")


if __name__ == "__main__":
    main()

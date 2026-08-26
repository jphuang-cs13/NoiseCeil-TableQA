#!/usr/bin/env python3
"""
Error Analysis Stage 1: Distribution Analysis

Purpose:
========
Analyze evaluation results for a specified dataset and model, and compute
the distribution across four outcome groups:
- Both Success: both K=1 and K=target succeed (score=1)
- Both Fail: both K=1 and K=target fail (score!=1)
- Perfect→Fail (K=1 succeeds but K=target fails): failure induced by negatives
- Perfect←Fail (K=1 fails but K=target succeeds): rare case where negatives help

Statistics are reported separately for hard and soft negatives.

Usage:
======
python3 experiments/error_analysis_stage1_distribution.py \
  --dataset e2ewtq \
  --model qwen3-32b \
  --target-k 30 \
  --eval-root evaluation_results/ \
  --output-dir error_analysis/

Output:
=======
- error_analysis/qwen3-32b_e2ewtq_K30_distribution.json
- error_analysis/qwen3-32b_e2ewtq_K30_distribution_summary.txt
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_eval_records(eval_path: Path) -> List[Dict]:
    """Load evaluation results from a JSON file."""
    if not eval_path.exists():
        return []
    try:
        with eval_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "records" in data:
                return data["records"]
            elif isinstance(data, list):
                return data
    except Exception as e:
        print(f"  ✗ Failed to load {eval_path}: {e}")
    return []


def parse_injection_filename(name: str) -> Tuple[int, str, str]:
    """
    Parse injection filename to extract K, rotation, and negative type.
    Examples: 1_perfect.jsonl, 2_end_hard.jsonl, 4_middle_soft.jsonl
    """
    base = name.rsplit(".", 1)[0]
    
    # Handle real_result format
    if base.startswith("real_result_"):
        try:
            k = int(base.replace("real_result_", ""))
            return k, "real_result", "real_result"
        except ValueError:
            pass
    
    # Handle K_position_negative format
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


def classify_relationship(k1_success: bool, kn_success: bool) -> str:
    """Classify the success/failure relationship between K=1 and K=target."""
    if k1_success and kn_success:
        return "both_success"
    elif not k1_success and not kn_success:
        return "both_fail"
    elif k1_success and not kn_success:
        return "perfect_to_fail"
    else:  # k1_success=False and kn_success=True
        return "perfect_from_fail"


def analyze_distribution(
    dataset: str,
    model: str,
    target_k: int,
    eval_root: Path,
) -> Dict:
    """Analyze the distribution of success/failure patterns."""
    
    model_dir = eval_root / model
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model evaluation directory not found: {model_dir}")
    
    # Load K=1 (perfect) baseline
    # Support both .json and .jsonl file formats
    perfect_json = model_dir / "1_perfect.json"
    perfect_jsonl = model_dir / "1_perfect.jsonl"
    perfect_file = perfect_json if perfect_json.exists() else perfect_jsonl
    
    perfect_records = load_eval_records(perfect_file)
    perfect_scores = {r.get("query_id"): r.get("score", 0) for r in perfect_records}
    
    # Load K=target evaluations (both hard and soft)
    distribution = {
        "hard": defaultdict(list),
        "soft": defaultdict(list),
    }
    
    # Find files matching both .json and .jsonl patterns
    eval_files = list(model_dir.glob("*.json")) + list(model_dir.glob("*.jsonl"))
    for eval_file in eval_files:
        k, rotation, negative = parse_injection_filename(eval_file.stem)
        
        # Skip if not the target K or if it's perfect (K=1)
        if k != target_k or negative == "perfect":
            continue
        
        if negative not in ["hard", "soft"]:
            continue
        
        records = load_eval_records(eval_file)
        for rec in records:
            qid = rec.get("query_id")
            target_score = rec.get("score", 0)
            perfect_score = perfect_scores.get(qid, 0)
            
            # Determine success (score=1)
            k1_success = perfect_score == 1
            kn_success = target_score == 1
            
            relationship = classify_relationship(k1_success, kn_success)
            
            # Store the record along with relationship info
            distribution[negative][relationship].append({
                "query_id": qid,
                "perfect_score": perfect_score,
                "target_score": target_score,
                "rotation": rotation,
                "file": eval_file.name,
            })
    
    return distribution


def generate_summary(distribution: Dict, target_k: int, model: str, dataset: str) -> str:
    """Generate a human-readable summary of the distribution."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Error Analysis Distribution Summary")
    lines.append(f"Dataset: {dataset} | Model: {model} | Target K: {target_k}")
    lines.append("=" * 80)
    
    for negative_type in ["hard", "soft"]:
        lines.append(f"\n[{negative_type.upper()} NEGATIVES]")
        lines.append("-" * 40)
        
        counts = {k: len(v) for k, v in distribution[negative_type].items()}
        total = sum(counts.values())
        
        if total == 0:
            lines.append("  No data found.")
            continue
        
        for category in ["both_success", "both_fail", "perfect_to_fail", "perfect_from_fail"]:
            count = counts.get(category, 0)
            pct = (count / total * 100) if total > 0 else 0
            lines.append(f"  {category:20s}: {count:4d} ({pct:5.1f}%)")
        
        lines.append(f"  {'TOTAL':20s}: {total:4d} (100.0%)")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Error Analysis Stage 1: Distribution Analysis"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., e2ewtq, feta, ottqa)")
    parser.add_argument("--model", required=True, help="Model name (e.g., qwen3-32b)")
    parser.add_argument("--target-k", type=int, default=30, help="Target retrieval depth K to analyze")
    parser.add_argument(
        "--eval-root",
        default="evaluation_results/",
        help="Root path to evaluation results (default: evaluation_results/)"
    )
    parser.add_argument(
        "--output-dir",
        default="error_analysis/",
        help="Output directory for analysis results (default: error_analysis/)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    # Check if eval_root already contains the model directory
    eval_root = Path(args.eval_root)
    potential_model_dir = eval_root / args.model
    
    # If model dir doesn't exist at eval_root, assume eval_root needs dataset appended
    if not potential_model_dir.exists():
        eval_root = eval_root / args.dataset
    
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Stage 1] Analyzing {args.dataset}/{args.model} with target K={args.target_k}")
    print(f"  Evaluation root: {eval_root}")
    print(f"  Output dir: {output_dir}")
    
    # Perform distribution analysis
    try:
        distribution = analyze_distribution(
            args.dataset,
            args.model,
            args.target_k,
            eval_root,
        )
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        return
    
    # Generate summary
    summary_text = generate_summary(distribution, args.target_k, args.model, args.dataset)
    print("\n" + summary_text)
    
    # Save JSON output
    output_file = output_dir / f"{args.model}_K{args.target_k}_distribution.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "dataset": args.dataset,
                "model": args.model,
                "target_k": args.target_k,
            },
            "distribution": {
                neg_type: {cat: [r for r in records] 
                          for cat, records in distribution[neg_type].items()}
                for neg_type in distribution
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Distribution analysis saved to: {output_file}")
    
    # Save summary text
    summary_file = output_dir / f"{args.model}_K{args.target_k}_distribution_summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"✓ Summary text saved to: {summary_file}")


if __name__ == "__main__":
    main()

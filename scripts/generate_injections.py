#!/usr/bin/env python3
"""
Generate injections from hard_soft_negative results.

This script processes hard_soft_negative data to create injections with different configurations:
- K: Total number of tables (gold + negatives) to include (1, 2, 3, 5, 10, ...)
- rotation: Position of gold_tables (start, middle, end)
- negative: Type of negatives to use (hard, soft)

Directory structure:
- Input: hard_soft_negative/[dataset]/[representation_file].[negative_type].jsonl
- Output: injections/[dataset]/[representation_file]/[k]_[rotation]_[negative].jsonl
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse

BASE_DIR = Path(__file__).resolve().parents[1]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate injections from hard_soft_negative results"
    )
    parser.add_argument("dataset", help="Dataset name (e.g., ottqa)")
    parser.add_argument("retriever", help="Retriever name (e.g., bge-m3)")
    parser.add_argument("representation", help="Representation type (e.g., TextTableRepresentation, TopNRowsRepresentation)")
    parser.add_argument(
        "other_params",
        nargs="*",
        help="Additional parameters (e.g., include_file_name=True max_length=8192)"
    )
    parser.add_argument("-k", type=int, required=True, help="K value (total tables to include)")
    parser.add_argument(
        "-r", "--rotation",
        choices=["start", "middle", "end"],
        required=True,
        help="Rotation position for gold_tables"
    )
    parser.add_argument(
        "-n", "--negative",
        choices=["hard", "soft"],
        required=True,
        help="Type of negative samples to use"
    )
    parser.add_argument(
        "-b", "--base-dir",
        default=str(BASE_DIR),
        help="Base directory path"
    )
    
    return parser.parse_args()


def normalize_retriever_name(retriever: str) -> str:
    # Remove trailing 'Retriever' if present
    if retriever.endswith('Retriever'):
        retriever = retriever[:-9]  # Remove 'Retriever'
    
    # Map of retriever names to their filename format
    name_map = {
        "bge-m3": "BGEM3",
        "bgem3": "BGEM3",
    }
    
    normalized = name_map.get(retriever.lower(), retriever)
    return normalized


def build_representation_filename(dataset: str, retriever: str, representation: str, 
                                   other_params: List[str]) -> str:
    """Build the representation filename from components."""
    # Construct the filename pattern
    parts = [representation]
    
    # Add other parameters
    parts.extend(other_params)
    
    # Add retriever (normalize the name)
    normalized_retriever = normalize_retriever_name(retriever)
    parts.append(normalized_retriever + "Retriever")
    
    # Add test prefix (k value will be matched via glob pattern)
    parts.append("test")
    
    return "_".join(parts)


def load_hard_soft_negative_data(base_dir: str, dataset: str, 
                                 representation_filename: str) -> Tuple[Dict[str, Any], str]:
    """
    Load hard_soft_negative data from disk.
    
    Returns:
        Tuple of (data_dict, full_representation_filename)
    """
    hard_soft_neg_dir = Path(base_dir) / "hard_soft_negative" / dataset
    
    # Find the matching file
    matching_files = list(hard_soft_neg_dir.glob(f"{representation_filename}*.jsonl"))
    
    if not matching_files:
        raise FileNotFoundError(
            f"No hard_soft_negative file found matching: {representation_filename}\n"
            f"in directory: {hard_soft_neg_dir}"
        )
    
    if len(matching_files) > 1:
        print(f"Warning: Multiple matching files found, using: {matching_files[0]}")
    
    file_path = matching_files[0]
    full_filename = file_path.stem  # filename without .jsonl
    
    # Load data
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                query_id = record["query_id"]
                data[query_id] = record
    
    return data, full_filename


def create_injection(gold_tables: List[str], negatives: List[Dict], 
                    k: int, rotation: str, negative_type: str) -> Tuple[List[str], bool]:
    """
    Create an injection for the given parameters.
    
    Args:
        gold_tables: List of gold table IDs
        negatives: List of negative records with "id" field
        k: Total number of tables to include
        rotation: Position of gold tables (start, middle, end)
        negative_type: Type of negatives (hard, soft) - for info only
    
    Returns:
        Tuple of (injection_list, is_valid)
        is_valid is False if K < len(gold_tables)
    """
    num_gold = len(gold_tables)
    
    # Special case for k=1: always return gold tables
    if k == 1:
        return gold_tables, True
    
    # Check if K is sufficient
    if k < num_gold:
        return [], False
    
    # Calculate how many negatives we need
    num_negatives_needed = k - num_gold
    
    # Extract negative IDs (preserving order)
    negative_ids = [neg["id"] for neg in negatives[:num_negatives_needed]]
    
    # If we don't have enough negatives, skip
    if len(negative_ids) < num_negatives_needed:
        return [], False
    
    # Build injection based on rotation
    if rotation == "start":
        injection = gold_tables + negative_ids
    elif rotation == "end":
        injection = negative_ids + gold_tables
    elif rotation == "middle":
        # Split negatives in half
        mid = len(negative_ids) // 2
        injection = negative_ids[:mid] + gold_tables + negative_ids[mid:]
    else:
        raise ValueError(f"Unknown rotation: {rotation}")
    
    return injection, True


def generate_injections(base_dir: str, dataset: str, retriever: str, 
                       representation: str, other_params: List[str],
                       k: int, rotation: str, negative_type: str) -> None:
    """
    Generate injections for the specified configuration.
    
    Args:
        base_dir: Base directory path
        dataset: Dataset name
        retriever: Retriever name
        representation: Representation type
        other_params: Additional parameters
        k: K value
        rotation: Rotation type
        negative_type: Negative type (hard/soft)
    """
    # Build representation filename
    rep_filename = build_representation_filename(dataset, retriever, representation, other_params)
    
    # Load hard_soft_negative data
    print(f"Loading hard_soft_negative data for {rep_filename}...")
    data, full_rep_filename = load_hard_soft_negative_data(base_dir, dataset, rep_filename)
    print(f"Loaded {len(data)} records")
    
    # Create output directory
    output_dir = Path(base_dir) / "injections" / dataset / full_rep_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Special handling for k=1: always use "1_perfect.jsonl"
    if k == 1:
        output_file = output_dir / "1_perfect.jsonl"
    else:
        output_file = output_dir / f"{k}_{rotation}_{negative_type}.jsonl"
    
    # Generate injections
    valid_count = 0
    skipped_count = 0
    
    with open(output_file, 'w') as f:
        for query_id in sorted(data.keys()):
            record = data[query_id]
            gold_tables = record.get("gold_tables", [])
            
            # For k=1, always use gold tables regardless of negative_type
            if k == 1:
                negatives = []  # Not used
            else:
                # Get the appropriate negative type
                if negative_type == "hard":
                    negatives = record.get("hard_negatives", [])
                else:  # soft
                    negatives = record.get("soft_negatives", [])
            
            # Create injection
            injection, is_valid = create_injection(
                gold_tables, negatives, k, rotation, negative_type
            )
            
            if not is_valid:
                skipped_count += 1
                continue
            
            # Write injection record
            injection_record = {
                "query_id": query_id,
                "question": record.get("question", ""),
                "injection": injection,
                "k": k,
                "rotation": rotation if k > 1 else "perfect",
                "negative": negative_type if k > 1 else "perfect"
            }
            
            f.write(json.dumps(injection_record) + "\n")
            valid_count += 1
    
    print(f"\nGeneration completed!")
    print(f"Output file: {output_file}")
    print(f"Valid injections: {valid_count}")
    print(f"Skipped (K < gold_tables): {skipped_count}")
    print(f"Total processed: {valid_count + skipped_count}")


def main():
    args = parse_arguments()
    
    try:
        generate_injections(
            base_dir=args.base_dir,
            dataset=args.dataset,
            retriever=args.retriever,
            representation=args.representation,
            other_params=args.other_params,
            k=args.k,
            rotation=args.rotation,
            negative_type=args.negative
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

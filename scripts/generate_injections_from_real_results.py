#!/usr/bin/env python3
"""
Generate injections directly from real retrieval results (without hard/soft negatives).

This script processes retrieval results to create injections that represent 
actual retriever rankings without any artificial manipulation:
- Injections are created directly from the ranked retrieval results
- No hard/soft negative classification - just real retriever rankings
- Used to validate if real retriever performance matches artificially injected results

Directory structure:
- Input: retrieval_results/[dataset]/[representation_file]_test_k50_retrieval.jsonl
- Output: injections/[dataset]/[representation_file]/real_result_[k].jsonl
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
        description="Generate injections from real retrieval results"
    )
    parser.add_argument("dataset", help="Dataset name (e.g., feta, e2ewtq, ottqa)")
    parser.add_argument("retriever", help="Retriever name (e.g., bge-m3)")
    parser.add_argument("representation", help="Representation type (e.g., TextTableRepresentation)")
    parser.add_argument(
        "other_params",
        nargs="*",
        help="Additional parameters (e.g., include_file_name=True max_length=8192)"
    )
    parser.add_argument(
        "--k-values",
        default="1,5,10,20,30,40,50",
        help="Comma-separated K values to generate (default: 1,5,10,20,30,40,50)"
    )
    parser.add_argument(
        "-b", "--base-dir",
        default=str(BASE_DIR),
        help="Base directory path"
    )
    
    return parser.parse_args()


def normalize_retriever_name(retriever: str) -> str:
    """
    Normalize retriever name to match filename format.
    
    Examples:
        bge-m3 -> BGEM3
        bgem3 -> BGEM3
    """
    # Remove trailing 'Retriever' if present
    if retriever.endswith('Retriever'):
        retriever = retriever[:-9]
    
    name_map = {
        "bge-m3": "BGEM3",
        "bgem3": "BGEM3",
    }
    
    normalized = name_map.get(retriever.lower(), retriever)
    return normalized


def build_retrieval_filename(dataset: str, retriever: str, representation: str, 
                             other_params: List[str]) -> str:
    """Build the retrieval results filename from components."""
    parts = [representation]
    parts.extend(other_params)
    
    normalized_retriever = normalize_retriever_name(retriever)
    parts.append(normalized_retriever + "Retriever")
    parts.append("test")
    parts.append("k50")
    parts.append("retrieval")
    
    return "_".join(parts)


def load_queries(base_dir: str, dataset: str) -> Dict[int, str]:
    """
    Load queries to map query_id to questions.
    
    Returns:
        Dict mapping question text to query_id
    """
    query_file = Path(base_dir) / 'data' / 'single-table-retrieval' / 'test' / dataset / 'query.jsonl'
    
    if not query_file.exists():
        raise FileNotFoundError(f"Query file not found: {query_file}")
    
    question_to_id = {}
    with open(query_file, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                query_id = record.get('id', record.get('query_id'))
                question = record.get('question', '')
                question_to_id[question] = query_id
    
    return question_to_id


def load_retrieval_results(base_dir: str, dataset: str, 
                          retrieval_filename: str) -> Tuple[Dict[str, Any], str]:
    """
    Load retrieval results from disk.
    
    Returns:
        Tuple of (data_dict, full_filename)
    """
    retrieval_dir = Path(base_dir) / "retrieval_results" / dataset
    
    # Find the matching file
    matching_files = list(retrieval_dir.glob(f"{retrieval_filename}.jsonl"))
    
    if not matching_files:
        raise FileNotFoundError(
            f"No retrieval results file found matching: {retrieval_filename}.jsonl\n"
            f"in directory: {retrieval_dir}"
        )
    
    if len(matching_files) > 1:
        print(f"Warning: Multiple matching files found, using: {matching_files[0]}")
    
    file_path = matching_files[0]
    full_filename = file_path.stem
    
    # Load queries to get query_id mapping
    question_to_id = load_queries(base_dir, dataset)
    
    # Load data
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                question = record.get("question", "")
                gold_tables = record.get("gold_tables", [])
                retrieved_tables = record.get("retrieved_tables", [])
                
                # Extract just the table IDs from retrieved results
                table_ids = [t["id"] for t in retrieved_tables]
                
                # Get query_id from question mapping
                query_id = question_to_id.get(question)
                if query_id is None:
                    print(f"Warning: No query_id found for question: {question[:50]}...")
                    continue
                
                # Store with query_id as key
                data[query_id] = {
                    "query_id": query_id,
                    "question": question,
                    "gold_tables": gold_tables,
                    "retrieved_table_ids": table_ids  # Real ranking from retriever
                }
    
    return data, full_filename


def create_injection_from_real_results(retrieved_table_ids: List[str], k: int) -> Tuple[List[str], bool]:
    """
    Create an injection from real retrieval results.
    
    Args:
        retrieved_table_ids: List of retrieved table IDs in ranking order
        k: Number of tables to include
    
    Returns:
        Tuple of (injection_list, is_valid)
    """
    if len(retrieved_table_ids) < k:
        # Not enough retrieved results
        return [], False
    
    # Just take top-k from real retriever ranking
    injection = retrieved_table_ids[:k]
    return injection, True


def generate_injections_from_real_results(base_dir: str, dataset: str, retriever: str, 
                                          representation: str, other_params: List[str],
                                          k_values: List[int]) -> None:
    """
    Generate injections for all K values from real retrieval results.
    
    Args:
        base_dir: Base directory path
        dataset: Dataset name
        retriever: Retriever name
        representation: Representation type
        other_params: Additional parameters
        k_values: List of K values to generate
    """
    # Build retrieval filename
    ret_filename = build_retrieval_filename(dataset, retriever, representation, other_params)
    
    # Load retrieval results (which also loads queries internally)
    print(f"Loading retrieval results for {ret_filename}...")
    data, full_ret_filename = load_retrieval_results(base_dir, dataset, ret_filename)
    print(f"Loaded {len(data)} retrieval records")
    
    # Create output directory (use the full representation filename for consistency)
    output_dir = Path(base_dir) / "injections" / dataset / full_ret_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate injections for each K value
    for k in k_values:
        print(f"\nGenerating real_result injections for k={k}...")
        
        # Injection file naming: real_result_{k}.jsonl
        output_file = output_dir / f"real_result_{k}.jsonl"
        
        valid_count = 0
        skipped_count = 0
        
        with open(output_file, 'w') as f:
            for query_id in sorted(data.keys()):
                record = data[query_id]
                question = record["question"]
                gold_tables = record["gold_tables"]
                retrieved_table_ids = record["retrieved_table_ids"]
                
                # Create injection from real result
                injection, is_valid = create_injection_from_real_results(retrieved_table_ids, k)
                
                if not is_valid:
                    skipped_count += 1
                    continue
                
                # Write injection record (must include query_id for compatibility with experiment runner)
                injection_record = {
                    "query_id": query_id,
                    "question": question,
                    "injection": injection,
                    "k": k,
                    "rotation": "real_result",
                    "negative": "real_result"
                }
                
                f.write(json.dumps(injection_record) + "\n")
                valid_count += 1
        
        print(f"  Output file: {output_file}")
        print(f"  Valid injections: {valid_count}")
        print(f"  Skipped (not enough results): {skipped_count}")
        print(f"  Total processed: {valid_count + skipped_count}")


def main():
    args = parse_arguments()
    
    # Parse K values
    try:
        k_values = sorted([int(k.strip()) for k in args.k_values.split(',')])
    except ValueError:
        print(f"Error: Invalid K values: {args.k_values}", file=sys.stderr)
        sys.exit(1)
    
    try:
        generate_injections_from_real_results(
            base_dir=args.base_dir,
            dataset=args.dataset,
            retriever=args.retriever,
            representation=args.representation,
            other_params=args.other_params,
            k_values=k_values
        )
        print("\n✓ Real result injection generation completed successfully!")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

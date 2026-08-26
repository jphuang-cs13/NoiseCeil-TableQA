#!/usr/bin/env python3
"""
Post-process reasoning results to calculate tokens retroactively.

For result files that were generated before proper token counting was implemented,
this script recalculates request_tokens and response_tokens based on:
- request_tokens: Computed from the prompt text (system + user) using len() / 4 estimation
- response_tokens: Computed from llm_response using len() / 4 estimation

Usage:
    python3 scripts/calculate_result_tokens.py <dataset> [llm_model] [--injection-base <base>]

Arguments:
    dataset: Dataset name (e.g., ottqa)
    llm_model: LLM model name (optional, default: gpt-oss-20b)
    --injection-base: Injection base name for filtering (optional)

Examples:
    # Calculate for all results in ottqa dataset
    python3 scripts/calculate_result_tokens.py ottqa

    # Calculate for specific LLM model
    python3 scripts/calculate_result_tokens.py ottqa gpt-4

    # Calculate for specific injection configuration
    python3 scripts/calculate_result_tokens.py ottqa gpt-oss-20b --injection-base TextTableRepresentation_include_file_name
"""

import argparse
import json
import sys
from pathlib import Path


def get_token_estimate(text: str) -> int:
    """
    Estimate token count from text length.
    Rough approximation: 1 token per 4 characters.
    """
    return max(1, len(text) // 4)


def calculate_prompt_tokens(question: str, tables: list, system_prompt: str, user_prompt_template: str) -> int:
    """
    Estimate request tokens by reconstructing the prompt.
    
    This mimics how the reasoner constructs prompts.
    For now, we use a simple estimation based on text length.
    """
    # Reconstruct approximate prompt
    prompt_parts = [
        system_prompt,
        question
    ]
    
    # Add table information (crude estimation)
    for table in tables:
        if isinstance(table, dict):
            # From representations
            prompt_parts.append(str(table.get('representation', '')))
    
    full_prompt = ' '.join(prompt_parts)
    return get_token_estimate(full_prompt)


def process_result_file(result_file: Path, query_file: Path, rep_file: Path, injection_file: Path) -> None:
    """
    Process a single result file to add token counts.
    """
    # Load original data
    queries = {}
    with open(query_file) as f:
        for line in f:
            data = json.loads(line)
            # Support both 'id' and 'query_id' field names
            query_id = data.get('query_id') or data.get('id')
            queries[query_id] = data
    
    reps = {}
    with open(rep_file) as f:
        for line in f:
            data = json.loads(line)
            reps[str(data['id'])] = data
    
    injections = {}
    with open(injection_file) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                injections[data.get('query_id')] = data
            except json.JSONDecodeError:
                continue
    
    # Process result file
    updated_results = []
    with open(result_file) as f:
        for line in f:
            if not line.strip():
                continue
            result = json.loads(line)
            query_id = result['query_id']
            llm_response = result.get('llm_response', '')
            
            # If tokens are already set (non-zero), skip
            if result.get('request_tokens', 0) > 0 and result.get('response_tokens', 0) > 0:
                updated_results.append(result)
                continue
            
            # Calculate response_tokens from response length
            response_tokens = get_token_estimate(llm_response)
            
            # Calculate request_tokens from question and tables
            request_tokens = 0
            if query_id in queries and query_id in injections:
                question = queries[query_id].get('question', '')
                injection_table_ids = injections[query_id].get('injection', [])
                
                # Get tables from injection
                tables = []
                for tid in injection_table_ids:
                    tid_str = str(tid)
                    if tid_str in reps:
                        tables.append(reps[tid_str])
                
                # Estimate request tokens
                prompt_text = question
                for table in tables:
                    prompt_text += str(table.get('representation', ''))
                
                request_tokens = get_token_estimate(prompt_text)
            
            # Update result
            result['request_tokens'] = request_tokens
            result['response_tokens'] = response_tokens
            updated_results.append(result)
    
    # Write back
    with open(result_file, 'w') as f:
        for result in updated_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✓ Updated {len(updated_results)} results in {result_file.name}")


def get_result_roots(base_dir: Path, dataset: str, llm_model: str) -> list:
    """Return result directories to process, including a separate real_result folder if present."""
    roots = [base_dir / 'reasoning_results' / dataset / llm_model]
    real_root = base_dir / 'reasoning_results' / dataset / f'{llm_model}_real_result'
    if real_root.exists() and real_root not in roots:
        roots.append(real_root)
    return roots


def main():
    parser = argparse.ArgumentParser(description='Calculate tokens for reasoning results')
    parser.add_argument('dataset', help='Dataset name (e.g., bird, ottqa)')
    parser.add_argument('llm_model', nargs='?', default='gpt-oss-20b', help='LLM model name (default: gpt-oss-20b)')
    parser.add_argument('--injection-base', help='Injection base name filter (optional)')

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    
    # Determine if single-table or multi-table
    single_path = base_dir / 'data' / 'single-table-retrieval' / 'test' / args.dataset

    
    if single_path.exists():
        is_multi = False
        data_dir = single_path
    else:
        print(f"Error: Dataset '{args.dataset}' not found")
        sys.exit(1)
    
    # Paths
    query_file = data_dir / 'query.jsonl'
    
    print(f"Processing results for dataset '{args.dataset}' with model '{args.llm_model}'")
    result_roots = get_result_roots(base_dir, args.dataset, args.llm_model)
    existing_result_roots = [p for p in result_roots if p.exists()]

    if not existing_result_roots:
        for root in result_roots:
            print(f"Result directory not found: {root}")
        sys.exit(1)

    for result_root in existing_result_roots:
        print(f"Result directory: {result_root}")
        print()

        # Find all result files
        result_files = list(result_root.glob('*.jsonl'))

        if not result_files:
            print("No result files found")
            continue

        print(f"Found {len(result_files)} result files to process:")
        for f in sorted(result_files):
            print(f"  - {f.name}")
        print()
    
        # Find corresponding injection files
        injection_root = base_dir / 'injections' / args.dataset

        # Prefer retrieval-based injection dirs for real_result outputs, otherwise prefer hard_soft_negative dirs.
        prefer_retrieval = result_root.name.endswith('_real_result')
        if prefer_retrieval:
            injection_dirs = list(injection_root.glob('*retrieval')) + list(injection_root.glob('*hard_soft_negative'))
        else:
            injection_dirs = list(injection_root.glob('*hard_soft_negative')) + list(injection_root.glob('*retrieval'))

        if args.injection_base:
            injection_dirs = [d for d in injection_dirs if args.injection_base in d.name]

        if not injection_dirs:
            print("Error: No injection directories found")
            continue

        injection_dir = injection_dirs[0]
        print(f"Using injection directory: {injection_dir.name}")
        print()

        
        inj_name = injection_dir.name
        rep_prefix = inj_name.split('_BGEM3Retriever')[0] if '_BGEM3Retriever' in inj_name else \
                                          inj_name.split('_retrieval')[0]

        # Try to find matching representation file
        rep_root = base_dir / 'representations' / args.dataset
        rep_candidates = list(rep_root.glob(f"{rep_prefix}.jsonl")) + \
                         list(rep_root.glob(f"{rep_prefix}*.jsonl"))

        if rep_candidates:
            rep_file = rep_candidates[0]
        else:
            # Fallback to default
            rep_file = rep_root / 'TextTableRepresentation.jsonl'
            if not rep_file.exists():
                # Try other defaults
                alt_reps = list(rep_root.glob('*.jsonl'))
                if alt_reps:
                    rep_file = alt_reps[0]
                else:
                    print(f"Error: No representation files found in {rep_root}")
                    continue

        print(f"Using representation file: {rep_file.name}")
        print()

        # Process each result file
        processed_count = 0
        for result_file in sorted(result_files):
            # Find corresponding injection file
            injection_file = injection_dir / result_file.name

            if not injection_file.exists():
                print(f"⚠ Skipping {result_file.name} (no corresponding injection file)")
                continue

            try:
                process_result_file(result_file, query_file, rep_file, injection_file)
                processed_count += 1
            except Exception as e:
                import traceback
                print(f"✗ Error processing {result_file.name}: {e}")
                traceback.print_exc()

        print()
        print(f"✓ Successfully processed {processed_count} result files in {result_root.name}")


if __name__ == '__main__':
    main()

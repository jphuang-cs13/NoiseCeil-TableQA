#!/usr/bin/env python3
"""
Run injection experiment script.

This script runs reasoning experiments on injection data using LLM.
It automatically processes from k=1 to max_k, resuming from last progress if interrupted.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Add app and llm to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'app'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.dataset_loader import DatasetLoader
from app.reasoners import LLMReasoner
from app.utils.token_utils import truncate_to_max_tokens


def parse_filename(filename: str) -> tuple:
    """
    Parse filename to extract k, rotation, and negative type for proper ordering.

    Returns: (k, rotation_order, negative_order, filename)
    where rotation_order: real_result=-1, perfect=0, start=1, middle=2, end=3
    negative_order: hard=0, soft=1, real_result=-1
    """
    # Remove .jsonl extension
    name = filename.replace('.jsonl', '')

    # Handle new real_result format: real_result_{k}.jsonl
    if name.startswith('real_result_'):
        # Format: real_result_{k}
        k_str = name.replace('real_result_', '')
        try:
            k = int(k_str)
            # real_result files sort first (rotation_order=-1, negative_order=-1)
            return (k, -1, -1, filename)
        except ValueError:
            # Fallback to treating as regular file
            pass

    # Handle old formats
    if '_perfect' in name:
        # Format: {k}_perfect
        k = int(name.split('_perfect')[0])
        return (k, 0, 0, filename)

    # Format: {k}_{rotation}_{negative}
    parts = name.split('_')
    if len(parts) >= 3:
        try:
            k = int(parts[0])
            rotation = parts[1]
            negative = parts[2]

            # Define order: perfect(0) -> start(1) -> middle(2) -> end(3)
            rotation_order = {'perfect': 0, 'start': 1, 'middle': 2, 'end': 3}.get(rotation, 99)

            # Define order: hard(0) -> soft(1)
            negative_order = {'hard': 0, 'soft': 1}.get(negative, 99)

            return (k, rotation_order, negative_order, filename)
        except (ValueError, KeyError):
            pass

    # Fallback: treat as unparseable
    return (999, 999, 999, filename)


def load_representations(rep_file: Path) -> dict:
    """Load table representations from jsonl file."""
    reps = {}
    with open(rep_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            reps[str(data['id'])] = data
    return reps


def load_queries(query_file: Path) -> dict:
    """Load queries from jsonl file."""
    queries = {}
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            queries[data['id']] = data
    return queries


def get_llm_model_name_from_reasoner(reasoner) -> str:
    """Extract LLM model name from reasoner."""
    try:
        # Use the LLM client's display model name method for proper formatting
        return reasoner.llm_client.get_display_model_name(reasoner.llm_client.model)
    except AttributeError:
        # Fallback to environment variable
        return get_llm_model_name()


def get_llm_model_name() -> str:
    """Extract LLM model name from environment variable."""
    model = os.getenv('LLM_MODEL', 'unknown')
    # Extract from xxx/MODEL_NAME:xxx format
    if '/' in model:
        parts = model.split('/')
        if len(parts) > 1:
            model = parts[1]
    if ':' in model:
        model = model.split(':')[0]
    return model


def _has_real_result_files(directory: Path) -> bool:
    """Whether this injection directory contains real_result_{k}.jsonl files."""
    return any(directory.glob("real_result_*.jsonl"))


def find_injection_dir(injection_root: Path, base_name: str, prefer_retrieval: bool = False) -> Path:
    """Find the injection directory matching the base name.

    Args:
        injection_root: Root folder under injections/<dataset>
        base_name: Injection base stem
        prefer_retrieval: When True (e.g., --real-only), prefer folders that
            contain real_result_*.jsonl and retrieval-style suffixes.
    """
    hard_soft_candidates = list(injection_root.glob(f"{base_name}_*hard_soft_negative"))
    if not hard_soft_candidates:
        hard_soft_candidates = list(injection_root.glob(f"{base_name}_k*hard_soft_negative"))

    retrieval_candidates = list(injection_root.glob(f"{base_name}_*retrieval"))
    fallback_candidates = [p for p in injection_root.iterdir() if p.is_dir() and p.name.startswith(f"{base_name}_")]

    if not hard_soft_candidates and not retrieval_candidates and not fallback_candidates:
        raise FileNotFoundError(f"No injection directory found for {base_name}")

    # Preserve original behavior for normal runs.
    if not prefer_retrieval:
        if hard_soft_candidates:
            hard_soft_candidates.sort(key=lambda p: (len(p.name), p.name))
            return hard_soft_candidates[0]
        if retrieval_candidates:
            retrieval_candidates.sort(key=lambda p: (len(p.name), p.name))
            return retrieval_candidates[0]
        fallback_candidates.sort(key=lambda p: (len(p.name), p.name))
        return fallback_candidates[0]

    # Real-only mode: prefer directories that actually contain real_result files.
    ordered = retrieval_candidates + hard_soft_candidates + fallback_candidates
    seen = set()
    candidates = []
    for p in ordered:
        if p not in seen:
            seen.add(p)
            candidates.append(p)

    if prefer_retrieval:
        real_candidates = [p for p in candidates if _has_real_result_files(p)]
        if real_candidates:
            # Prefer retrieval-style directories first, then shortest stable name.
            real_candidates.sort(key=lambda p: ("retrieval" not in p.name, len(p.name), p.name))
            return real_candidates[0]

    # Fallback in real-only mode when no real_result_*.jsonl exists anywhere.
    candidates.sort(key=lambda p: (len(p.name), p.name))
    return candidates[0]


def find_representation_file(rep_root: Path, representation: str, other_params: list) -> Path:
    """Find representation file matching representation + params."""
    if not rep_root.exists():
        raise FileNotFoundError(f"Representation directory not found: {rep_root}")

    # Try exact filename first (param order sensitive)
    other_str = '_'.join(other_params) if other_params else ''
    exact_name = f"{representation}_{other_str}.jsonl" if other_str else f"{representation}.jsonl"
    exact_path = rep_root / exact_name
    if exact_path.exists():
        return exact_path

    # Fallback: find any file that contains all required tokens
    candidates = sorted(rep_root.glob(f"{representation}*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No representation files found for {representation} in {rep_root}"
        )

    if other_params:
        filtered = [
            p for p in candidates
            if all(param in p.stem for param in other_params)
        ]
        if filtered:
            # Prefer shortest match to avoid accidentally selecting unrelated variants
            filtered.sort(key=lambda p: len(p.name))
            return filtered[0]

    # Last resort: use base representation file if available
    base_file = rep_root / f"{representation}.jsonl"
    if base_file.exists():
        return base_file

    raise FileNotFoundError(
        f"No matching representation file found for {representation} with params {other_params} in {rep_root}"
    )


def get_existing_ks(injection_dir: Path) -> list:
    """Get list of existing k values from injection files."""
    files = list(injection_dir.glob("*.jsonl"))
    ks = set()
    for f in files:
        name = f.stem
        # Handle real_result_{k}.jsonl
        if name.startswith('real_result_'):
            try:
                k = int(name.replace('real_result_', ''))
                ks.add(k)
                continue
            except ValueError:
                pass

        parts = name.split('_')
        if parts and parts[0].isdigit():
            ks.add(int(parts[0]))
    return sorted(list(ks))


def main():
    parser = argparse.ArgumentParser(description='Run injection experiment')
    parser.add_argument('dataset', help='Dataset name (e.g., ottqa)')
    parser.add_argument('retriever', help='Retriever name (e.g., BGEM3Retriever)')
    parser.add_argument('representation', help='Representation name (e.g., TextTableRepresentation)')
    parser.add_argument('other_params', nargs='*', help='Other parameters for representation')
    parser.add_argument('--max-queries', type=int, default=None, help='Maximum number of queries to process in this run (optional)')
    parser.add_argument('--injection-file', default=None, help='Only process the specified injection file (optional)')
    parser.add_argument('--real-only', action='store_true', help='Only run real_result_* injection files (real retriever outputs)')
    parser.add_argument('--retry-attempts', type=int, default=5, help='Retry attempts for same provider when LLM call fails or returns empty response (default: 5)')
    parser.add_argument('--retry-initial-wait', type=float, default=5.0, help='Initial wait seconds before retry (default: 5.0)')
    parser.add_argument('--retry-backoff', type=float, default=2.0, help='Exponential backoff multiplier between retries (default: 2.0)')
    parser.add_argument('--retry-max-wait', type=float, default=120.0, help='Maximum wait seconds between retries (default: 120.0)')

    args = parser.parse_args()

    # Build injection directory base name
    other_str = '_'.join(args.other_params) if args.other_params else ''
    injection_base = f"{args.representation}"
    if other_str:
        injection_base += f"_{other_str}"
    injection_base += f"_{args.retriever}"

    print(f"Injection base: {injection_base}")

    # Paths
    base_dir = Path(__file__).resolve().parents[1]
    injection_root = base_dir / 'injections' / args.dataset
    rep_root = base_dir / 'representations' / args.dataset
    query_file = base_dir / 'data' / 'single-table-retrieval' / 'test' / args.dataset / 'query.jsonl'

    # Find representation file
    rep_file = find_representation_file(rep_root, args.representation, args.other_params)
    print(f"Using representation file: {rep_file}")

    # Find injection directory
    try:
        injection_dir = find_injection_dir(
            injection_root,
            injection_base,
            prefer_retrieval=args.real_only,
        )
        print(f"Found injection dir: {injection_dir}")
    except FileNotFoundError as e:
        print(e)
        return

    # Get existing k values
    existing_ks = get_existing_ks(injection_dir)
    if not existing_ks:
        print("No injection files found")
        return

    max_k = max(existing_ks)
    print(f"Existing k values: {existing_ks}, max_k: {max_k}")

    # Load data
    print("Loading representations...")
    reps = load_representations(rep_file)
    print(f"Loaded {len(reps)} representations")

    print("Loading queries...")
    queries = load_queries(query_file)
    print(f"Loaded {len(queries)} queries")

    # Initialize reasoner
    print("Initializing LLM reasoner...")
    reasoner = LLMReasoner(strategy_name='zero-shot')
    
    # Get LLM model name from reasoner
    llm_model = get_llm_model_name_from_reasoner(reasoner)
    print(f"LLM model: {llm_model}")

    # Output directory
    # When running real-only experiments, keep results separate to avoid mixing
    if args.real_only:
        output_root = base_dir / 'reasoning_results' / args.dataset / f"{llm_model}_real_result"
    else:
        output_root = base_dir / 'reasoning_results' / args.dataset / llm_model
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_root}")

    # Determine progress file based on whether --injection-file is specified
    # When injection-file is specified, use a separate progress file for parallel execution
    if args.injection_file:
        # Create progress file name based on injection file name (e.g., "40_start_soft_progress.json")
        progress_file_base = args.injection_file.replace('.jsonl', '')
        progress_file = output_root / f"{progress_file_base}_progress.json"
        print(f"Using independent progress file for parallel execution: {progress_file.name}")
    else:
        # Original shared progress file for sequential execution
        progress_file = output_root / 'progress.json'
        print(f"Using shared progress file for sequential execution")

    current_file = None
    current_query_id = -1
    # Track last successfully saved query (used when stopping on failures)
    last_successful_query_id = -1

    # Query counter for max-queries limit
    processed_count = 0

    # Progress file is only used for error recovery, not for determining resume point
    # We always determine resume point from the actual output files

    reasoner.load_model()

    # Get all injection files and sort them by processing order
    injection_files = []
    for file_path in injection_dir.glob("*.jsonl"):
        injection_files.append(file_path.name)

    # Sort files by processing order
    injection_files.sort(key=parse_filename)

    # Filter if --injection-file is specified
    if args.injection_file:
        injection_files = [f for f in injection_files if f == args.injection_file]
        if not injection_files:
            print(f"No injection file named {args.injection_file} found in {injection_dir}")
            return

    # If --real-only, filter to only files containing 'real_result' (do not change directory selection)
    if args.real_only:
        real_files = [f for f in injection_files if 'real_result' in f]
        if not real_files:
            print(f"No real_result files found in {injection_dir}")
            return
        injection_files = real_files

    print(f"Found {len(injection_files)} injection files to process")
    for filename in injection_files:
        print(f"  {filename}")

    # Process each injection file
    for filename in injection_files:
        print(f"\nProcessing {filename}")

        input_file = injection_dir / filename
        output_file = output_root / filename

        with open(input_file, 'r', encoding='utf-8') as f_in:
            lines = [json.loads(l) for l in f_in if l.strip()]

        # Load already processed query_ids for this file (if any)
        processed_query_ids = set()
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f_out:
                for l in f_out:
                    if not l.strip():
                        continue
                    try:
                        d = json.loads(l)
                        processed_query_ids.add(d['query_id'])
                    except Exception:
                        continue

        total_injections = len(lines)
        processed_so_far = len(processed_query_ids)

        # Determine unprocessed indices in original order
        unprocessed_indices = [idx for idx, d in enumerate(lines) if d['query_id'] not in processed_query_ids]

        if not unprocessed_indices:
            print(f"  Skipping {filename} (completely processed: {processed_so_far}/{total_injections})")
            continue

        if processed_so_far > 0:
            print(f"  Resuming {filename} ({processed_so_far}/{total_injections} queries already processed)")
        else:
            print(f"  Processing {filename} (0/{total_injections} queries processed)")

        # Reset progress tracking for this file
        file_last_successful_query_id = max(processed_query_ids) if processed_query_ids else -1

        # Process lines in original order, skipping those already done
        for idx in unprocessed_indices:
            data = lines[idx]
            query_id = data['query_id']
            question = data['question']
            table_ids = data['injection']
            
            # Track if we need to retry with reduced tables (for qwen models that fail with 400)
            tables_reduced = False
            max_tables = len(table_ids)  # Start with all tables

            def reload_tables(table_ids_list, limit_to_top_n=None):
                """Helper function to reload table representations"""
                if limit_to_top_n:
                    table_ids_list = table_ids_list[:limit_to_top_n]
                
                loaded_tables = []
                for tid in table_ids_list:
                    tid_str = str(tid)
                    if tid_str in reps:
                        rep_data = reps[tid_str]
                        rep_text = rep_data['representation']
                        rep_text = truncate_to_max_tokens(rep_text, max_tokens=8000)
                        loaded_tables.append({
                            'representation': rep_text,
                            'metadata': rep_data['metadata']
                        })
                return loaded_tables

            # Initial load: use all tables
            tables = reload_tables(table_ids)
            actual_table_count = len(tables)

            if not tables:
                print(f"    Warning: No tables found for query_id {query_id}")
                progress = {'current_file': filename, 'current_query_id': file_last_successful_query_id}

            # Get gold answer
            gold_answer = queries.get(query_id, {}).get('answer', '')

            print(f"    About to call LLM API for query_id {query_id}...")

            # Reason
            llm_response = ''
            request_tokens = 0
            response_tokens = 0
            execution_time = 0
            retry_wait = max(0.0, args.retry_initial_wait)
            retry_limit = max(0, args.retry_attempts)
            last_error_msg = ''
            success = False

            for attempt in range(retry_limit + 1):
                if attempt > 0:
                    print(
                        f"    Retrying same provider for query_id {query_id} "
                        f"(attempt {attempt}/{retry_limit}) after {retry_wait:.1f}s..."
                    )
                    if retry_wait > 0:
                        time.sleep(retry_wait)

                try:
                    reasoning_result = reasoner.reason(question, tables, return_metadata=True)
                    if isinstance(reasoning_result, dict):
                        llm_response = reasoning_result.get('text', '')
                        request_tokens = reasoning_result.get('request_tokens', 0)
                        response_tokens = reasoning_result.get('response_tokens', 0)
                        execution_time = reasoning_result.get('execution_time', 0)

                        if 'error' in reasoning_result and reasoning_result['error']:
                            last_error_msg = str(reasoning_result['error'])
                            print(f"    Error from LLM API: {last_error_msg}")
                            
                            # Check if this is a 400 error (size-related) on qwen model
                            if '400' in last_error_msg and 'qwen' in llm_model.lower() and not tables_reduced:
                                print(f"    Detected 400 error on qwen model, retrying with top-20 tables...")
                                tables_reduced = True
                                tables = reload_tables(table_ids, limit_to_top_n=20)
                                print(f"    Reloaded with {len(tables)} tables (top-20)")
                                # Don't break, continue to next attempt with reduced tables
                            
                            llm_response = ''
                        elif llm_response and str(llm_response).strip():
                            success = True
                            break
                        else:
                            last_error_msg = 'Empty LLM response'
                            print(f"    {last_error_msg} for query_id {query_id}.")
                    else:
                        # Fallback for old API
                        llm_response = reasoning_result
                        request_tokens = 0
                        response_tokens = 0
                        execution_time = 0
                        if llm_response and str(llm_response).strip():
                            success = True
                            break
                        last_error_msg = 'Empty LLM response'
                        print(f"    {last_error_msg} for query_id {query_id}.")

                except Exception as e:
                    last_error_msg = str(e)
                    print(f"    Error at query_id {query_id}: {last_error_msg}")
                    
                    # Check if this is a 400 error on qwen model
                    if '400' in last_error_msg and 'qwen' in llm_model.lower() and not tables_reduced:
                        print(f"    Detected 400 error in exception on qwen model, retrying with top-20 tables...")
                        tables_reduced = True
                        tables = reload_tables(table_ids, limit_to_top_n=20)
                        print(f"    Reloaded with {len(tables)} tables (top-20)")
                        # Continue to next attempt with reduced tables

                retry_wait = min(args.retry_max_wait, max(0.0, retry_wait) * max(1.0, args.retry_backoff))

            if not success:
                print(
                    f"    Failed after {retry_limit + 1} attempts for query_id {query_id}. "
                    f"Last error: {last_error_msg if last_error_msg else 'unknown'}"
                )
                progress = {'current_file': filename, 'current_query_id': file_last_successful_query_id}
                with open(progress_file, 'w') as f:
                    json.dump(progress, f)
                print(f"Progress saved at last successful query_id: {file_last_successful_query_id}")
                sys.exit(1)

            # Save result
            result = {
                'query_id': query_id,
                'question': question,
                'llm_response': llm_response,
                'gold_answer': gold_answer,
                'request_tokens': request_tokens,
                'response_tokens': response_tokens,
                'execution_time': execution_time
            }

            # Append successful result
            with open(output_file, 'a', encoding='utf-8') as f_out:
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

            # Update last successful and progress
            file_last_successful_query_id = query_id
            progress = {'current_file': filename, 'current_query_id': file_last_successful_query_id}
            with open(progress_file, 'w') as f:
                json.dump(progress, f)

            processed_count += 1
            print(f"    Processed query_id {query_id} (total: {processed_count})")

            # Check max queries limit
            if args.max_queries and processed_count >= args.max_queries:
                print(f"Reached max queries limit ({args.max_queries}). Stopping.")
                return

    # Clean up progress file
    if progress_file.exists():
        progress_file.unlink()
        print("\nExperiment completed successfully!")


if __name__ == '__main__':
    main()

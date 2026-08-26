#!/usr/bin/env python3
"""
Generate table representations for a dataset.

This script generates table representations using specified representation types
and parameters, saving them to the representations directory.

Usage examples:
    python scripts/generate_representations.py e2ewtq TextTableRepresentation 8192    
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import importlib

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'app'))

from utils.dataset_loader import DatasetLoader
from representations.basic_representations import (
    TableRepresentation,
    TextTableRepresentation,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate table representations for a dataset"
    )
    parser.add_argument("dataset", help="Dataset name (e.g., e2ewtq, bird, ottqa)")
    parser.add_argument("representation", help="Representation type (e.g., TextTableRepresentation)")
    parser.add_argument(
        "other_params",
        nargs="*",
        help="Additional parameters (e.g., max_length=8192 include_file_name=True)"
    )
    return parser.parse_args()


def create_representation_instance(representation_name: str, params: Dict[str, Any]) -> TableRepresentation:
    """Create a representation instance with given parameters."""
    rep_classes = {
        'TextTableRepresentation': TextTableRepresentation,
    }
    
    if representation_name not in rep_classes:
        raise ValueError(f"Unknown representation: {representation_name}")
    
    cls = rep_classes[representation_name]
    try:
        instance = cls(**params)
        instance.load_model()
        return instance
    except Exception as e:
        raise ValueError(f"Failed to create {representation_name} with params {params}: {e}")


def parse_params(param_strings: List[str]) -> Dict[str, Any]:
    """Parse parameter strings into a dictionary."""
    params = {}
    for param in param_strings:
        if '=' in param:
            key, value = param.split('=', 1)
            # Try to convert to appropriate type
            if value.lower() in ('true', 'false'):
                params[key] = value.lower() == 'true'
            elif value.isdigit():
                params[key] = int(value)
            elif value.replace('.', '').isdigit():
                params[key] = float(value)
            else:
                params[key] = value
        else:
            # Boolean flag
            params[param] = True
    return params


def generate_representations(dataset: str, representation_name: str, params: Dict[str, Any]):
    """Generate representations for the dataset."""
    # Determine if multi-table
    is_multi_table = dataset in ['bird', 'mmqa_two_table']
    category = 'multi-table-retrieval' if is_multi_table else 'single-table-retrieval'
    
    # Load dataset
    loader = DatasetLoader()
    data = loader.load_data(category, dataset, 'test')
    
    # Create representation instance
    rep = create_representation_instance(representation_name, params)
    
    # Collect all tables
    all_tables = []
    if is_multi_table:
        for sample in data:
            if isinstance(sample['table'], list):
                all_tables.extend(sample['table'])
            else:
                all_tables.append(sample['table'])
    else:
        for sample in data:
            all_tables.append(sample['table'])
    
    # Remove duplicates based on id
    unique_tables = {}
    for table in all_tables:
        table_id = table['id']
        if table_id not in unique_tables:
            unique_tables[table_id] = table
    
    tables = list(unique_tables.values())
    
    # Generate representations
    representations = []
    for table in tables:
        try:
            rep_str = rep.build_representation(table)
            representations.append({
                'id': table['id'],
                'representation': rep_str,
                'metadata': {
                    'file_name': table.get('file_name', ''),
                    'sheet_name': table.get('sheet_name', '')
                }
            })
        except Exception as e:
            print(f"Error generating representation for table {table['id']}: {e}")
            continue
    
    # Create output filename
    param_str = '_'.join(f"{k}={v}" for k, v in sorted(params.items()))
    if param_str:
        filename = f"{representation_name}_{param_str}.jsonl"
    else:
        filename = f"{representation_name}.jsonl"
    
    # Output path
    output_dir = Path(__file__).parent.parent / 'representations' / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename
    
    # Save to jsonl
    with open(output_file, 'w', encoding='utf-8') as f:
        for rep_data in representations:
            json.dump(rep_data, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Generated {len(representations)} representations for {dataset} using {representation_name}")
    print(f"Saved to {output_file}")


def main():
    args = parse_arguments()
    
    # Parse parameters
    params = parse_params(args.other_params)
    
    # Generate representations
    generate_representations(args.dataset, args.representation, params)


if __name__ == "__main__":
    main()
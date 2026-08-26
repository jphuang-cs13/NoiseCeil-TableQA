#!/usr/bin/env python3
"""
Retrieval Experiment Runner

This script runs retrieval experiments with configurable parameters.
Supports different datasets, retrievers, representations, and evaluation metrics.

Usage:
    python experiments/run_experiment.py --dataset feta --retriever BGEM3Retriever --representation TextTableRepresentation --k_values 5,10,20

Arguments:
    --dataset: Dataset name (feta, e2ewtq, ottqa)
    --retriever: Retriever name (BGEM3Retriever)
    --representation: Representation class name (TextTableRepresentation)
    --representation_params: JSON string of representation parameters (e.g., '{"max_length": 512}')
    --k_values: Comma-separated list of k values for evaluation (default: 2,5,10,20,50,100)
    --split: Dataset split (default: test)
    --force_reindex: Force re-encoding of indices even if they exist
    --skip_retrieval: Skip retrieval step if results already exist
    --skip_evaluation: Skip evaluation step
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.table_qa_api import TableQAApi
from app.retrievers import (
    BGEM3Retriever
)
from app.representations import (
    TextTableRepresentation
)
from app.evaluation.evaluator import RetrieverEvaluator
from app.utils.dataset_loader import DatasetLoader


class RetrievalExperiment:
    """Manages retrieval experiments with different configurations."""

    def __init__(self, args):
        self.args = args
        self.api = TableQAApi()
        self.dataset_loader = DatasetLoader()

        # Configuration
        self.dataset = args.dataset
        self.retriever_name = args.retriever
        self.representation_name = args.representation
        self.representation_params = json.loads(args.representation_params) if args.representation_params else {}
        self.k_values = [int(k) for k in args.k_values.split(',')]
        self.split = args.split
        self.dry_run = args.dry_run
        self.skip_retrieval = args.skip_retrieval
        self.skip_evaluation = args.skip_evaluation

        # Setup paths
        self.experiments_dir = project_root / "experiments"
        self.retrieval_results_dir = project_root / "retrieval_results"
        self.evaluations_dir = project_root / "evaluations"

        # Ensure directories exist
        self.experiments_dir.mkdir(exist_ok=True)
        self.retrieval_results_dir.mkdir(exist_ok=True)
        self.evaluations_dir.mkdir(exist_ok=True)

    def get_retriever(self):
        """Get retriever instance based on name."""
        retrievers = {
            'bge-m3': BGEM3Retriever,
        }

        if self.retriever_name not in retrievers:
            raise ValueError(f"Unknown retriever: {self.retriever_name}")

        return retrievers[self.retriever_name]()

    def get_representation_name_with_params(self):
        """Generate representation name with parameters for consistent naming."""
        # Create a temporary representation instance to extract all parameters
        temp_representation = self.get_representation()
        
        rep_name = self.representation_name
        
        # Get representation parameters for filename (same logic as cache_retrieval_results)
        rep_params = {}
        if hasattr(temp_representation, '__dict__'):
            # Extract relevant parameters
            for attr in ['max_length', 'n_rows', 'include_file_name', 'include_sheet_name', 'include_metadata']:
                if hasattr(temp_representation, attr):
                    rep_params[attr] = getattr(temp_representation, attr)
        
        # Create parameter string for filename
        param_str = ""
        if rep_params:
            param_parts = [f"{k}={v}" for k, v in sorted(rep_params.items())]
            param_str = f"_{'_'.join(param_parts)}"
        
        return f"{rep_name}{param_str}"

    def get_representation(self):
        """Get representation instance based on name."""
        representations = {
            'TextTableRepresentation': TextTableRepresentation,
        }

        if self.representation_name not in representations:
            raise ValueError(f"Unknown representation: {self.representation_name}")
        
        return representations[self.representation_name](**self.representation_params)

    def get_max_length_for_model(self, retriever_name: str) -> int:
        """Get appropriate max length for different models."""
        # Model-specific max lengths (based on typical limits)
        model_lengths = {            
            'bge-m3': 8192,              
        }
        return model_lengths.get(retriever_name, 512)

    def run(self):
        """Run the complete experiment."""
        print("=" * 80)
        print("RETRIEVAL EXPERIMENT")
        if self.dry_run:
            print("(DRY RUN - No actual processing)")
        print("=" * 80)
        print(f"Dataset: {self.dataset}")
        print(f"Retriever: {self.retriever_name}")
        print(f"Representation: {self.representation_name}")
        print(f"Representation params: {self.representation_params}")
        print(f"K values: {self.k_values}")
        print(f"Split: {self.split}")
        print("=" * 80)

        if self.dry_run:
            print("\n✓ Dry run completed - configuration validated")
            return True

        start_time = time.time()

        try:
            # 1. Load dataset
            print("\n1. Loading dataset...")
            dataset_data = self.dataset_loader.load_data("single-table-retrieval", self.dataset, self.split)
            queries = [sample['question'] for sample in dataset_data]
            ground_truths = [[str(gt['id']) for gt in sample['ground_truth_tables']] for sample in dataset_data]
            tables = [sample['table'] for sample in dataset_data]
            print(f"Loaded {len(queries)} queries and {len(tables)} tables from {self.dataset}")

            # 2. Prepare representations over the FULL table corpus of the split
            print("\n2. Creating representations...")

            # For single-table-retrieval we should index the entire table corpus, not only GT tables
            tables_path = Path(self.dataset_loader.base_path) / "single-table-retrieval" / self.split / self.dataset / "table.jsonl"
            all_tables_dict = self.dataset_loader._load_tables(str(tables_path))
            unique_tables: List[Dict[str, Any]] = list(all_tables_dict.values())

            representation = self.get_representation()
            representations = representation.build_representations(unique_tables)
            print(f"Created {len(representations)} representations over {len(unique_tables)} unique tables")

            # 3. Setup retriever
            print(f"\n3. Setting up {self.retriever_name} retriever...")
            retriever = self.get_retriever()
            retriever.load_model()

            # Create API instance with the retriever and representation
            self.api = TableQAApi(retriever=retriever, representation=representation)

            retrieved_results = None

            if not self.skip_retrieval:
                # 4. Encode indices (if needed)
                print("\n4. Encoding indices...")
                index_key = self.api.encode_indices(
                    category="single-table-retrieval",
                    dataset_name=self.dataset,
                    split=self.split,
                    representation_type=self.representation_name,
                    representation_params=self.representation_params
                )
                print(f"Index key: {index_key}")

                # 5. Run batch retrieval
                print("\n5. Running batch retrieval...")
                max_k = max(self.k_values)
                retrieved_results = []
                for i, query in enumerate(queries):
                    if i % 50 == 0:
                        print(f"Retrieving for query {i+1}/{len(queries)}...")
                    results = self.api.retrieve_tables(query, index_key, top_k=max_k)
                    retrieved_results.append(results)

                print(f"Retrieved results for {len(retrieved_results)} queries")

                # 6. Cache retrieval results
                print("\n6. Caching retrieval results...")
                cache_file = self.api.cache_retrieval_results(
                    queries=queries,
                    retrieved_results=retrieved_results,
                    ground_truth_tables=ground_truths,
                    dataset_name=self.dataset,
                    max_k=max_k,
                    category="single-table-retrieval",
                    split=self.split
                )
                print(f"Results cached to: {cache_file}")
            else:
                # Load cached results for evaluation
                print("\n4. Loading cached retrieval results...")
                max_k = max(self.k_values)
                rep_name_with_params = self.get_representation_name_with_params()
                cached_data = self.api.load_retrieval_results(
                    dataset_name=self.dataset,
                    representation_name=rep_name_with_params,
                    retriever_name=retriever.__class__.__name__,
                    split=self.split,
                    max_k=max_k
                )
                if cached_data is None:
                    raise FileNotFoundError(f"No cached results found for {rep_name_with_params}_{self.retriever_name}_{self.split}_k{max_k}")
                
                # Convert cached data to retrieved_results format
                retrieved_results = []
                for entry in cached_data:
                    # Convert retrieved_tables back to (table, score) format
                    results = []
                    for rt in entry['retrieved_tables']:
                        # We need to reconstruct the table dict - for now, create minimal dict
                        table = {'id': rt['id']}
                        results.append((table, rt['score']))
                    retrieved_results.append(results)
                print(f"Loaded {len(retrieved_results)} cached retrieval results")

            # 7. Run evaluation
            if not self.skip_evaluation:
                print("\n7. Running evaluation...")
                evaluator = RetrieverEvaluator(
                    retriever=retriever,
                    dataset_name=f"{self.dataset}_{self.split}",
                    representation_name=self.get_representation_name_with_params(),
                    retriever_name=self.retriever_name,
                    output_dir=str(self.evaluations_dir)
                )

                # Extract available table IDs from unique_tables
                available_table_ids = [str(tbl.get('id')) for tbl in unique_tables if isinstance(tbl, dict) and 'id' in tbl]
                
                evaluation_report = evaluator.evaluate(
                    queries, 
                    ground_truths, 
                    self.k_values, 
                    retrieved_results,
                    available_tables=available_table_ids
                )
                print("Evaluation completed")

            # Cleanup
            retriever.unload_model()
            self.api.close()

            total_time = time.time() - start_time
            print(f"Total time: {total_time:.2f}s")
            print("=" * 80)
            print("EXPERIMENT COMPLETED SUCCESSFULLY")
            print("=" * 80)

        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True


def main():
    parser = argparse.ArgumentParser(description="Run retrieval experiments")
    parser.add_argument('--dataset', required=True, choices=['feta', 'e2ewtq', 'ottqa'],
                       help='Dataset name')
    parser.add_argument('--retriever', required=True,
                       choices=['bge-m3'],
                       help='Retriever name')
    parser.add_argument('--representation', required=True,
                       choices=['TextTableRepresentation'],
                       help='Representation class name')
    parser.add_argument('--representation_params', default='{}',
                       help='JSON string of representation parameters')
    parser.add_argument('--include-metadata', action='store_true',
                       help='Include metadata field from table in representation')
    parser.add_argument('--k_values', default='1,5,10,20,30,40,50',
                       help='Comma-separated list of k values for evaluation')
    parser.add_argument('--split', default='test',
                       help='Dataset split')
    parser.add_argument('--dry_run', action='store_true',
                       help='Validate configuration without running actual processing')
    parser.add_argument('--skip_retrieval', action='store_true',
                       help='Skip retrieval step if results exist')
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='Skip evaluation step')

    args = parser.parse_args()

    # Validate and merge representation params
    try:
        representation_params = json.loads(args.representation_params)
        # Add include_metadata if flag is set
        if args.include_metadata:
            representation_params['include_metadata'] = True
        args.representation_params = json.dumps(representation_params)
    except json.JSONDecodeError:
        print("❌ Invalid representation_params JSON")
        return 1

    # Run experiment
    experiment = RetrievalExperiment(args)
    success = experiment.run()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
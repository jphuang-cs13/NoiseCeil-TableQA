"""
Table QA API - Unified interface for Table Question Answering pipeline.

This module provides a high-level API for the complete Table QA pipeline,
including table representation, retrieval, and reasoning stages.
"""

import os
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple
from .utils.dataset_loader import DatasetLoader
from .evaluation.evaluator import RetrieverEvaluator
from .reasoners import TableReasoner, LLMReasoner
from .representations import (
    TableRepresentation,
    TextTableRepresentation
)
from .retrievers import TableRetriever

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class TableQAApi:
    """
    Unified API for Table Question Answering pipeline.

    This class orchestrates the different stages of Table QA:
    1. Table representation building
    2. Index encoding for retrieval
    3. Table retrieval
    4. Reasoning for answer generation
    """

    def __init__(self,
                 reasoner: Optional[TableReasoner] = None,
                 representation: Optional[TableRepresentation] = None,
                 retriever: Optional[TableRetriever] = None,
                 strategy_name: Optional[str] = 'zero-shot',
                 **reasoner_kwargs):
        """
        Initialize the Table QA API with components.

        Args:
            reasoner: TableReasoner instance (default: LLMReasoner)
            representation: TableRepresentation instance (default: TextTableRepresentation)
            retriever: TableRetriever instance (default: BM25Retriever)
        """
        # Default to LLMReasoner with zero-shot strategy
        self.reasoner = reasoner or LLMReasoner(strategy_name=strategy_name, **reasoner_kwargs)
        # Default to TextTableRepresentation for meaningful retrieval
        self.representation = representation or TextTableRepresentation(max_length=512, include_file_name=True)
        # Default to BM25Retriever
        self.retriever = retriever
        self.dataset_loader = DatasetLoader()

        # Load models by default
        self.reasoner.load_model()
        self.representation.load_model()
        if self.retriever is not None:
            self.retriever.load_model()

    def build_representations(self,
                             category: str,
                             dataset_name: str,
                             split: str,
                             **kwargs) -> List[Any]:
        """
        Build representations for all tables in a dataset.

        Args:
            category: Dataset category ('multi-table-retrieval' or 'single-table-retrieval')
            dataset_name: Name of the dataset
            split: 'train' or 'test'
            **kwargs: Additional parameters for representation building

        Returns:
            List of table representations
        """
        import os
        import json

        # Get representation method name with parameters
        rep_method = self.representation.__class__.__name__
        
        # Get representation parameters for filename
        rep_params = {}
        if hasattr(self.representation, '__dict__'):
            # Extract relevant parameters
            for attr in ['max_length', 'n_rows', 'include_file_name', 'include_sheet_name']:
                if hasattr(self.representation, attr):
                    rep_params[attr] = getattr(self.representation, attr)
        
        # Create parameter string for filename
        param_str = ""
        if rep_params:
            param_parts = [f"{k}={v}" for k, v in sorted(rep_params.items())]
            param_str = f"_{'_'.join(param_parts)}"
        
        rep_file_name = f"{rep_method}{param_str}.jsonl"

        # Check if pre-built representations exist
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rep_dir = os.path.join(project_root, "representations", dataset_name)
        rep_file = os.path.join(rep_dir, rep_file_name)

        # Assemble the table corpus
        if category == "single-table-retrieval":
            tables_path = Path(self.dataset_loader.base_path) / category / split / dataset_name / "table.jsonl"
            all_tables_dict = self.dataset_loader._load_tables(str(tables_path))
            unique_tables = list(all_tables_dict.values())
        else:
            data = self.dataset_loader.load_data(category, dataset_name, split)
            all_tables = []
            for sample in data:
                if isinstance(sample['table'], list):
                    all_tables.extend(sample['table'])
                else:
                    all_tables.append(sample['table'])

            unique_tables = []
            seen_ids = set()
            for table in all_tables:
                if table['id'] not in seen_ids:
                    unique_tables.append(table)
                    seen_ids.add(table['id'])

        expected_count = len(unique_tables)

        if os.path.exists(rep_file):
            print(f"Loading pre-built representations from {rep_file}...")
            representations = []
            with open(rep_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        representations.append(data['representation'])
            print(f"Loaded {len(representations)} representations from cache")
            if len(representations) == expected_count:
                return representations
            else:
                print(f"Cache size mismatch (expected {expected_count}, found {len(representations)}); rebuilding...")

        # If no pre-built representations or mismatch, build them
        print(f"Building representations for {category}/{dataset_name}/{split}...")

        representations = self.representation.build_representations(unique_tables, **kwargs)

        # Save representations to cache
        os.makedirs(rep_dir, exist_ok=True)
        with open(rep_file, 'w', encoding='utf-8') as f:
            for i, (table, rep) in enumerate(zip(unique_tables, representations)):
                data = {
                    'id': table['id'],
                    'representation': rep,
                    'metadata': {
                        'file_name': table.get('file_name', ''),
                        'sheet_name': table.get('sheet_name', '')
                    }
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

        print(f"Built and cached representations for {len(unique_tables)} unique tables")
        return representations

    def encode_indices(self,
                      category: str,
                      dataset_name: str,
                      split: str,
                      **kwargs) -> Any:
        """
        Encode indices for retrieval from a dataset.

        Args:
            category: Dataset category
            dataset_name: Name of the dataset
            split: 'train' or 'test'
            **kwargs: Additional parameters for encoding

        Returns:
            Encoded retrieval index or index key
        """
        print(f"Encoding indices for {category}/{dataset_name}/{split}...")
        if category == "single-table-retrieval":
            tables_path = Path(self.dataset_loader.base_path) / category / split / dataset_name / "table.jsonl"
            all_tables_dict = self.dataset_loader._load_tables(str(tables_path))
            unique_tables = list(all_tables_dict.values())
        else:
            data = self.dataset_loader.load_data(category, dataset_name, split)

            all_tables = []
            for sample in data:
                if isinstance(sample['table'], list):
                    all_tables.extend(sample['table'])
                else:
                    all_tables.append(sample['table'])

            # Remove duplicates
            unique_tables = []
            seen_ids = set()
            for table in all_tables:
                if table['id'] not in seen_ids:
                    unique_tables.append(table)
                    seen_ids.add(table['id'])

        # Check if retriever is BM25Retriever or BGE-M3 (which use representations)
        if hasattr(self.retriever, '_get_index_key'):
            # BM25Retriever/BGE-M3: need representations and additional params
            representations = self.build_representations(category, dataset_name, split, **kwargs)

            # Extract table IDs in the same order as unique_tables
            table_ids = [table['id'] for table in unique_tables]

            # Get representation info for index key
            rep_type = self.representation.__class__.__name__
            rep_params = {}
            if hasattr(self.representation, '__dict__'):
                # Extract relevant parameters
                for attr in ['max_length', 'n_rows', 'include_file_name', 'include_sheet_name']:
                    if hasattr(self.representation, attr):
                        rep_params[attr] = getattr(self.representation, attr)

            # Override with explicitly passed representation_params if provided
            if 'representation_params' in kwargs:
                rep_params.update(kwargs['representation_params'])

            # Filter out parameters that are already explicitly passed
            filtered_kwargs = {k: v for k, v in kwargs.items()
                             if k not in ['representation_type', 'representation_params']}

            indices = self.retriever.encode_indices(
                representations,
                table_ids=table_ids,
                dataset_category=category,
                dataset_name=dataset_name,
                split=split,
                representation_type=rep_type,
                representation_params=rep_params,
                **filtered_kwargs
            )
        else:
            # Other retrievers: use tables directly
            indices = self.retriever.encode_indices(unique_tables, **kwargs)

        print(f"Encoded indices for {len(unique_tables)} unique tables")
        return indices

    def retrieve_tables(self,
                       query: str,
                       indices: Any,
                       top_k: int = 5,
                       **kwargs) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve relevant tables for a query.

        Args:
            query: The question/query string
            indices: Pre-encoded retrieval index or index key
            top_k: Number of tables to retrieve
            **kwargs: Additional parameters

        Returns:
            List of (table_dict, score) tuples
        """
        print(f"Retrieving tables for query: {query[:50]}...")

        # Check if retriever is BM25Retriever
        if hasattr(self.retriever, '_get_index_key'):
            # BM25Retriever: indices is the index_key
            results = self.retriever.retrieve(query, indices, top_k=top_k, **kwargs)
        else:
            # Other retrievers: indices is the actual index object
            results = self.retriever.retrieve(query, indices, top_k=top_k, **kwargs)

        print(f"Retrieved {len(results)} tables")
        return results

    def cache_retrieval_results(self,
                              queries: List[str],
                              retrieved_results: List[List[Tuple[Dict[str, Any], float]]],
                              ground_truth_tables: List[List[str]],
                              dataset_name: str,
                              max_k: int,
                              category: str = "single-table-retrieval",
                              split: str = "test") -> str:
        """
        Cache retrieval results for evaluating multiple reasoning-stage k values.

        Args:
            queries: List of query strings
            retrieved_results: List of retrieved results for each query (ordered by score)
            ground_truth_tables: List of ground truth table IDs for each query
            dataset_name: Name of the dataset
            max_k: Maximum k value (will cache top max_k results)
            category: Dataset category
            split: Dataset split ('train', 'test', etc.)

        Returns:
            Path to the saved cache file
        """
        # Create cache directory structure
        rep_name = self.representation.__class__.__name__
        
        # Get representation parameters for filename
        rep_params = {}
        if hasattr(self.representation, '__dict__'):
            # Extract relevant parameters
            for attr in ['max_length', 'n_rows', 'include_file_name', 'include_sheet_name']:
                if hasattr(self.representation, attr):
                    rep_params[attr] = getattr(self.representation, attr)
        
        # Create parameter string for filename
        param_str = ""
        if rep_params:
            param_parts = [f"{k}={v}" for k, v in sorted(rep_params.items())]
            param_str = f"_{'_'.join(param_parts)}"
        
        retriever_name = self.retriever.__class__.__name__
        
        cache_dir = PROJECT_ROOT / "retrieval_results" / dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate cache filename with representation parameters
        cache_file = cache_dir / f"{rep_name}{param_str}_{retriever_name}_{split}_k{max_k}_retrieval.jsonl"

        print(f"Caching retrieval results to {cache_file}...")

        # Write results in JSONL format
        with open(cache_file, 'w', encoding='utf-8') as f:
            for query, retrieved, gold_tables in zip(queries, retrieved_results, ground_truth_tables):
                # Extract top max_k results with order
                retrieved_at_max_k = retrieved[:max_k]

                # Convert gold_tables to set for lookup
                gold_set = set(str(gt) for gt in gold_tables)

                # Build retrieved tables list with gold tag
                retrieved_tables = []
                for idx, (table, score) in enumerate(retrieved_at_max_k):
                    table_id = str(table.get('id', ''))
                    retrieved_tables.append({
                        'rank': idx + 1,
                        'id': table_id,
                        'score': float(score),
                        'is_gold': table_id in gold_set
                    })

                # Cache entry
                cache_entry = {
                    'question': query,
                    'retrieved_tables': retrieved_tables,
                    'gold_tables': gold_tables,
                    'metadata': {
                        'dataset': dataset_name,
                        'category': category,
                        'split': split,
                        'representation': rep_name,
                        'retriever': retriever_name,
                        'max_k': max_k
                    }
                }

                f.write(json.dumps(cache_entry, ensure_ascii=False) + '\n')

        print(f"Cached {len(queries)} retrieval results")
        return str(cache_file)

    def load_retrieval_results(self,
                             dataset_name: str,
                             representation_name: str,
                             retriever_name: str,
                             split: str = "test",
                             max_k: int = 100) -> Optional[List[Dict[str, Any]]]:
        """
        Load cached retrieval results.

        Args:
            dataset_name: Name of the dataset
            representation_name: Name of the representation method
            retriever_name: Name of the retriever
            split: Dataset split
            max_k: Maximum k value

        Returns:
            List of cached retrieval results, or None if not found
        """
        cache_dir = PROJECT_ROOT / "retrieval_results" / dataset_name
        cache_file = cache_dir / f"{representation_name}_{retriever_name}_{split}_k{max_k}_retrieval.jsonl"

        if not cache_file.exists():
            return None

        print(f"Loading cached retrieval results from {cache_file}...")
        results = []

        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line.strip()))

        print(f"Loaded {len(results)} cached retrieval results")
        return results

    def reason(self,
              question: str,
              tables: List[Dict[str, Any]],
              **kwargs) -> str:
        """
        Perform reasoning to answer a question using provided tables.

        Args:
            question: The question to answer
            tables: List of relevant table data
            **kwargs: Additional parameters

        Returns:
            The reasoned answer
        """
        print(f"Reasoning answer for question: {question[:50]}...")
        answer = self.reasoner.reason(question, tables, **kwargs)
        print(f"Generated answer: {answer[:100]}...")
        return answer

    def full_pipeline(self,
                     category: str,
                     dataset_name: str,
                     split: str,
                     sample_index: int = 0,
                     top_k: int = 5,
                     cache_retrieval: bool = True,
                     **kwargs) -> Dict[str, Any]:
        """
        Run the complete Table QA pipeline on a dataset sample.

        Args:
            category: Dataset category
            dataset_name: Name of the dataset
            split: 'train' or 'test'
            sample_index: Index of the sample to process
            top_k: Number of tables to retrieve
            cache_retrieval: Whether to cache retrieval results
            **kwargs: Additional parameters

        Returns:
            Dict with 'question', 'ground_truth_tables', 'answer', 'retrieved_tables', 'reasoned_answer'
        """
        print(f"Running full pipeline on {category}/{dataset_name}/{split} sample {sample_index}...")

        # Load data
        data = self.dataset_loader.load_data(category, dataset_name, split)
        if sample_index >= len(data):
            raise IndexError(f"Sample index {sample_index} out of range for {len(data)} samples")

        sample = data[sample_index]
        question = sample['question']
        ground_truth_tables = sample['ground_truth_tables']
        expected_answer = sample['answer']

        # Build representations and encode indices
        indices = self.encode_indices(category, dataset_name, split, **kwargs)

        # Retrieve tables
        retrieved_results = self.retrieve_tables(question, indices, top_k=top_k, **kwargs)
        retrieved_tables = [table for table, score in retrieved_results]

        # Reason answer
        reasoned_answer = self.reason(question, retrieved_tables, **kwargs)

        result = {
            'question': question,
            'ground_truth_tables': ground_truth_tables,
            'expected_answer': expected_answer,
            'retrieved_tables': retrieved_results,
            'reasoned_answer': reasoned_answer
        }

        print("Pipeline completed successfully")
        return result

    def evaluate(self,
                 category: str,
                 dataset_name: str,
                 split: str,
                 k_values: List[int] = None,
                 cache_retrieval: bool = True,
                 output_dir: str = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Run evaluation on a dataset using the current pipeline configuration.

        Args:
            category: Dataset category
            dataset_name: Name of the dataset
            split: 'train' or 'test'
            k_values: List of k values to evaluate (default: [1, 5, 10])
            cache_retrieval: Whether to cache retrieval results for reasoning stage
            output_dir: Directory to save evaluation results (default: {PROJECT_ROOT}/evaluations)
            **kwargs: Additional parameters for representation/retrieval

        Returns:
            Evaluation report dictionary
        """
        if output_dir is None:
            output_dir = str(PROJECT_ROOT / "evaluations")
        if k_values is None:
            k_values = [1, 5, 10]

        print(f"Starting evaluation for {category}/{dataset_name}/{split}...")

        # 1. Load Data
        data = self.dataset_loader.load_data(category, dataset_name, split)
        queries = []
        ground_truths = []
        ground_truth_objects = []  # For caching
        
        for sample in data:
            queries.append(sample['question'])
            # Extract IDs from ground truth tables
            gt_ids = [str(gt['id']) for gt in sample['ground_truth_tables']]
            ground_truths.append(gt_ids)
            ground_truth_objects.append(gt_ids)

        # 2. Prepare Indices
        indices = self.encode_indices(category, dataset_name, split, **kwargs)

        # 3. Create Adapter for RetrieverEvaluator
        # Evaluator expects retrieve(query, k) -> List[Item], but API needs indices
        class BoundRetriever:
            def __init__(self, api_instance, indices_obj):
                self.api = api_instance
                self.indices = indices_obj
            
            def retrieve(self, query: str, k: int) -> List[Any]:
                results = self.api.retrieve_tables(query, self.indices, top_k=k)
                # Return just the table objects/dicts (which contain 'id')
                return [item[0] for item in results]

        bound_retriever = BoundRetriever(self, indices)

        # 4. Run Evaluation
        evaluator = RetrieverEvaluator(
            retriever=bound_retriever,
            dataset_name=dataset_name,
            representation_name=self.representation.__class__.__name__,
            retriever_name=self.retriever.__class__.__name__,
            output_dir=output_dir
        )

        report = evaluator.evaluate(queries, ground_truths, k_values)

        # 5. Cache retrieval results if requested
        if cache_retrieval and k_values:
            max_k = max(k_values)
            
            # Retrieve all results at max_k
            all_retrieved_results = []
            for query in queries:
                results = bound_retriever.retrieve(query, max_k)
                # Convert back to (table, score) format
                # We need to get scores - for now, we'll use indices
                scored_results = []
                for table in results:
                    # Find original result with score
                    retrieval_result = self.retrieve_tables(query, indices, top_k=max_k)
                    for res_table, score in retrieval_result:
                        if str(res_table.get('id')) == str(table.get('id')):
                            scored_results.append((table, score))
                            break
                all_retrieved_results.append(scored_results)

            self.cache_retrieval_results(
                queries=queries,
                retrieved_results=all_retrieved_results,
                ground_truth_tables=ground_truth_objects,
                dataset_name=dataset_name,
                max_k=max_k,
                category=category,
                split=split
            )

        return report

    def close(self):
        """Clean up resources."""
        self.reasoner.unload_model()
        self.representation.unload_model()
        self.retriever.unload_model()
        print("TableQAApi: All models unloaded")

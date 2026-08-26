import json
import time
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from app.evaluation.metrics import RetrievalMetrics

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

class RetrieverEvaluator:
    """
    Evaluator for table retrieval systems.
    Executes retrieval, calculates metrics, and saves results.
    """

    def __init__(self, 
                 retriever: Any,
                 dataset_name: str,
                 representation_name: str,
                 retriever_name: str,
                 output_dir: str = None,
                 is_multi_table: bool = False):
        """
        Initialize the evaluator.

        Args:
            retriever: Retriever instance. Must have a retrieve(query, k) method.
            dataset_name: Name of the dataset (e.g., 'fetaqa').
            representation_name: Name of the representation (e.g., 'schema_only').
            retriever_name: Name of the retriever (e.g., 'bm25').
            output_dir: Root directory for saving evaluation results. 
                       Defaults to {PROJECT_ROOT}/evaluations
            is_multi_table: Whether this is a multi-table evaluation (affects CSV output)
        """
        self.retriever = retriever
        self.dataset_name = dataset_name
        self.representation_name = representation_name
        self.retriever_name = retriever_name
        self.is_multi_table = is_multi_table
        
        if output_dir is None:
            output_dir = str(PROJECT_ROOT / "evaluations")
        
        self.output_dir = Path(output_dir)

    def evaluate(self, 
                 queries: List[str], 
                 ground_truths: List[List[str]], 
                 k_values: List[int],
                 retrieved_results: List[List[Tuple[Dict[str, Any], float]]] = None,
                 available_tables: List[str] = None) -> Dict[str, Any]:
        """
        Run evaluation on a set of queries.

        Args:
            queries: List of query strings.
            ground_truths: List of lists of relevant table IDs corresponding to queries.
            k_values: List of k values to calculate metrics for.
            retrieved_results: Optional list of retrieved results. If None, will perform retrieval.
            available_tables: Optional list of table IDs that are actually available in the index.
                             Used to exclude unavailable ground truth tables from metric calculations.

        Returns:
            Dictionary containing evaluation results (config, averages, details).
        """
        if len(queries) != len(ground_truths):
            raise ValueError(f"Queries count ({len(queries)}) and ground truths count ({len(ground_truths)}) must match.")

        if retrieved_results is not None and len(queries) != len(retrieved_results):
            raise ValueError(f"Queries count ({len(queries)}) and retrieved results count ({len(retrieved_results)}) must match.")

        if not k_values:
            raise ValueError("k_values list cannot be empty.")

        max_k = max(k_values)
        results = []
        
        use_provided_results = retrieved_results is not None
        
        if use_provided_results:
            print(f"Starting evaluation with provided results: {self.dataset_name} | {self.representation_name} | {self.retriever_name}")
        else:
            print(f"Starting evaluation with live retrieval: {self.dataset_name} | {self.representation_name} | {self.retriever_name}")
            
        if available_tables is not None:
            unavailable_count = sum(1 for gt in ground_truths for table_id in gt if table_id not in available_tables)
            if unavailable_count > 0:
                print(f"Note: {unavailable_count} ground truth tables are unavailable in the index and will be excluded from metrics.")
            
        print(f"Processing {len(queries)} queries with max_k={max_k}...")
        
        start_time = time.time()

        # Iterate through queries
        for i, (query, relevant_ids) in enumerate(zip(queries, ground_truths)):
            # 1. Get retrieved results
            if use_provided_results:
                retrieved_items = retrieved_results[i]
                retrieved_ids = self._extract_ids(retrieved_items)
            else:
                # Perform live retrieval
                try:
                    # Assuming retriever returns a list of items
                    retrieved_items = self.retriever.retrieve(query, k=max_k)
                    retrieved_ids = self._extract_ids(retrieved_items)
                except Exception as e:
                    print(f"Error retrieving for query '{query}': {e}")
                    retrieved_ids = []

            # 2. Calculate metrics for all k values
            # This uses the logic in metrics.py to calculate for all k using the max_k results
            metrics = RetrievalMetrics.calculate_metrics_at_multiple_k(
                retrieved_ids, relevant_ids, k_values, available_relevant=available_tables
            )
            
            results.append({
                'query': query,
                'relevant_ids': relevant_ids,
                'retrieved_ids': retrieved_ids,
                'metrics': metrics
            })

        total_time = time.time() - start_time
        
        # 3. Calculate average metrics
        metrics_list = [r['metrics'] for r in results]
        avg_metrics = RetrievalMetrics.average_metrics(metrics_list)

        # 4. Prepare report
        evaluation_report = {
            'config': {
                'dataset': self.dataset_name,
                'representation': self.representation_name,
                'retriever': self.retriever_name,
                'k_values': k_values,
                'timestamp': datetime.now().isoformat(),
                'total_time_seconds': total_time,
                'queries_count': len(queries),
                'available_tables_count': len(available_tables) if available_tables else None
            },
            'average_metrics': avg_metrics,
            'detailed_results': results
        }

        # 5. Save results
        self._save_report(evaluation_report)
        
        return evaluation_report

    def _extract_ids(self, items: List[Any]) -> List[str]:
        """
        Helper to extract IDs from retrieved items.
        Handles strings, dicts with 'id', objects with 'id' attribute, or tuples (table_dict, score).
        """
        ids = []
        for item in items:
            if isinstance(item, str):
                ids.append(item)
            elif isinstance(item, tuple) and len(item) >= 2:
                # Handle (table_dict, score) tuples
                table_dict = item[0]
                if isinstance(table_dict, dict) and 'id' in table_dict:
                    ids.append(str(table_dict['id']))
                elif hasattr(table_dict, 'id'):
                    ids.append(str(table_dict.id))
                else:
                    ids.append(str(table_dict))
            elif isinstance(item, dict) and 'id' in item:
                ids.append(str(item['id']))
            elif hasattr(item, 'id'):
                ids.append(str(item.id))
            else:
                # Fallback: convert to string
                ids.append(str(item))
        return ids

    def _save_report(self, report: Dict[str, Any]):
        """
        Save evaluation report to disk in JSON and CSV formats.
        Structure: evaluations/dataset/representation/retriever/
        """
        save_dir = self.output_dir / self.dataset_name / self.representation_name / self.retriever_name
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON
        json_filename = f"results_{timestamp}.json"
        json_path = save_dir / json_filename
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        # Save summary CSV (append mode for tracking history)
        csv_path = save_dir / "summary.csv"
        
        # Flatten metrics for CSV
        flat_metrics = {}
        
        # For multi-table evaluation, only include recall_strict and recall_soft
        if self.is_multi_table:
            metrics_to_save = ['recall_strict', 'recall_soft']
        else:
            # For single-table, include all metrics
            metrics_to_save = list(report['average_metrics'].keys())
        
        for metric_name, k_dict in report['average_metrics'].items():
            if metric_name in metrics_to_save:
                for k, value in k_dict.items():
                    flat_metrics[f"{metric_name}@{k}"] = value
                
        summary_row = {
            'timestamp': report['config']['timestamp'],
            'queries_count': report['config']['queries_count'],
            'execution_time': report['config']['total_time_seconds'],
            **flat_metrics
        }
        
        df = pd.DataFrame([summary_row])
        
        if csv_path.exists():
            # Append without header
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            # Create with header
            df.to_csv(csv_path, index=False)
            
        print(f"Results saved to: {save_dir}")
"""
Retrieval evaluation metrics.

This module implements various metrics for evaluating table retrieval systems,
including recall, precision, and NDCG.
"""

import math
import re
from typing import List, Dict, Any, Tuple
import numpy as np


class RetrievalMetrics:
    """
    Collection of retrieval evaluation metrics.
    """

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int, available_relevant: List[str] = None) -> float:
        """
        Calculate Recall@K (flattened for multi-table datasets).
        
        For multi-table retrieval: Returns 1.0 if ANY relevant table is found in top-k, else 0.0.
        This treats all relevant tables as a single unit - success if we hit any one of them.
        For single-table retrieval: Works identically (binary: 0.0 or 1.0).

        Args:
            retrieved: List of retrieved table IDs
            relevant: List of relevant table IDs (can contain multiple ground truth tables)
            k: Number of top results to consider
            available_relevant: List of relevant IDs actually present in the index (for handling missing tables)

        Returns:
            Recall@K score (0.0 or 1.0) - 1.0 if any relevant table found, else 0.0
        """
        if not relevant:
            return 1.0 if not retrieved[:k] else 0.0

        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        # If available_relevant is provided, only evaluate against available tables
        if available_relevant is not None:
            available_set = set(available_relevant)
            relevant_set = relevant_set.intersection(available_set)
            if not relevant_set:
                return 1.0 if not retrieved_at_k else 0.0
        
        intersection = retrieved_at_k.intersection(relevant_set)
        
        # Return 1.0 if any relevant table is found (flattened/binary approach for multi-table)
        # This works for both single-table (1 relevant) and multi-table (N relevant) datasets
        return 1.0 if intersection else 0.0

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int, available_relevant: List[str] = None) -> float:
        """
        Calculate Precision@K.

        Args:
            retrieved: List of retrieved table IDs
            relevant: List of relevant table IDs
            k: Number of top results to consider
            available_relevant: List of relevant IDs actually present in the index (for handling missing tables)

        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if k == 0:
            return 0.0

        retrieved_at_k = retrieved[:k]
        if not retrieved_at_k:
            return 0.0

        relevant_set = set(relevant)
        # If available_relevant is provided, only count available relevant tables
        if available_relevant is not None:
            available_set = set(available_relevant)
            relevant_set = relevant_set.intersection(available_set)
        
        relevant_retrieved = sum(1 for table_id in retrieved_at_k if table_id in relevant_set)

        return relevant_retrieved / k

    @staticmethod
    def recall_strict_at_k(retrieved: List[str], relevant: List[str], k: int, available_relevant: List[str] = None) -> float:
        """
        Calculate Strict Recall@K (for multi-table retrieval).
        Returns 1.0 if ALL relevant tables are found in top-k, otherwise 0.0.

        Args:
            retrieved: List of retrieved table IDs
            relevant: List of relevant table IDs
            k: Number of top results to consider
            available_relevant: List of relevant IDs actually present in the index

        Returns:
            Strict Recall@K score (0.0 or 1.0)
        """
        if not relevant:
            return 1.0

        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        # If available_relevant is provided, only evaluate against available tables
        if available_relevant is not None:
            available_set = set(available_relevant)
            relevant_set = relevant_set.intersection(available_set)
            if not relevant_set:
                return 1.0
        
        # Check if all relevant tables are in top-k
        return 1.0 if relevant_set.issubset(retrieved_at_k) else 0.0

    @staticmethod
    def recall_soft_at_k(retrieved: List[str], relevant: List[str], k: int, available_relevant: List[str] = None) -> float:
        """
        Calculate Soft Recall@K (for multi-table retrieval).
        Returns the proportion of relevant tables found in top-k (partial credit).

        Args:
            retrieved: List of retrieved table IDs
            relevant: List of relevant table IDs
            k: Number of top results to consider
            available_relevant: List of relevant IDs actually present in the index

        Returns:
            Soft Recall@K score (0.0 to 1.0)
        """
        if not relevant:
            return 1.0

        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        # If available_relevant is provided, only evaluate against available tables
        if available_relevant is not None:
            available_set = set(available_relevant)
            relevant_set = relevant_set.intersection(available_set)
            if not relevant_set:
                return 1.0
        
        # Calculate proportion of relevant tables found
        intersection = retrieved_at_k.intersection(relevant_set)
        return len(intersection) / len(relevant_set)

    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int, available_relevant: List[str] = None) -> float:
        """
        Calculate NDCG@K.

        Args:
            retrieved: List of retrieved table IDs (ordered by relevance score)
            relevant: List of relevant table IDs
            k: Number of top results to consider
            available_relevant: List of relevant IDs actually present in the index (for handling missing tables)

        Returns:
            NDCG@K score (0.0 to 1.0)
        """
        if k == 0 or not retrieved:
            return 0.0

        relevant_set = set(relevant)
        # If available_relevant is provided, only evaluate against available tables
        if available_relevant is not None:
            available_set = set(available_relevant)
            relevant_set = relevant_set.intersection(available_set)
            if not relevant_set:
                return 0.0
        
        retrieved_at_k = retrieved[:k]

        # Calculate DCG
        dcg = 0.0
        for i, table_id in enumerate(retrieved_at_k):
            if table_id in relevant_set:
                dcg += 1.0 / math.log2(i + 2)  # i + 2 because positions start from 1

        # Calculate IDCG (ideal DCG)
        num_relevant = min(len(relevant_set), k)
        idcg = 0.0
        for i in range(num_relevant):
            idcg += 1.0 / math.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_metrics_at_multiple_k(retrieved: List[str], relevant: List[str],
                                      k_values: List[int], available_relevant: List[str] = None) -> Dict[str, Dict[int, float]]:
        """
        Calculate recall, precision, NDCG, recall_strict, and recall_soft at multiple k values.

        Args:
            retrieved: List of retrieved table IDs (ordered by relevance score)
            relevant: List of relevant table IDs
            k_values: List of k values to evaluate
            available_relevant: List of relevant IDs actually present in the index (for handling missing tables)

        Returns:
            Dictionary with metrics at each k value
        """
        max_k = max(k_values) if k_values else 0
        retrieved_at_max_k = retrieved[:max_k] if max_k > 0 else []

        results = {
            'recall': {},
            'precision': {},
            'ndcg': {},
            'recall_strict': {},
            'recall_soft': {}
        }

        for k in sorted(k_values):
            results['recall'][k] = RetrievalMetrics.recall_at_k(retrieved_at_max_k, relevant, k, available_relevant)
            results['precision'][k] = RetrievalMetrics.precision_at_k(retrieved_at_max_k, relevant, k, available_relevant)
            results['ndcg'][k] = RetrievalMetrics.ndcg_at_k(retrieved_at_max_k, relevant, k, available_relevant)
            results['recall_strict'][k] = RetrievalMetrics.recall_strict_at_k(retrieved_at_max_k, relevant, k, available_relevant)
            results['recall_soft'][k] = RetrievalMetrics.recall_soft_at_k(retrieved_at_max_k, relevant, k, available_relevant)

        return results

    @staticmethod
    def average_metrics(metrics_list: List[Dict[str, Dict[int, float]]]) -> Dict[str, Dict[int, float]]:
        """
        Calculate average metrics across multiple queries/samples.

        Args:
            metrics_list: List of metric dictionaries from individual queries

        Returns:
            Averaged metrics dictionary
        """
        if not metrics_list:
            return {}

        # Collect all k values
        all_k_values = set()
        for metrics in metrics_list:
            for metric_name in metrics:
                all_k_values.update(metrics[metric_name].keys())

        k_values = sorted(all_k_values)

        # Initialize result structure
        avg_metrics = {
            'recall': {k: [] for k in k_values},
            'precision': {k: [] for k in k_values},
            'ndcg': {k: [] for k in k_values},
            'recall_strict': {k: [] for k in k_values},
            'recall_soft': {k: [] for k in k_values}
        }

        # Collect values for averaging
        for metrics in metrics_list:
            for metric_name in ['recall', 'precision', 'ndcg', 'recall_strict', 'recall_soft']:
                for k in k_values:
                    if k in metrics.get(metric_name, {}):
                        avg_metrics[metric_name][k].append(metrics[metric_name][k])

        # Calculate averages
        result = {
            'recall': {},
            'precision': {},
            'ndcg': {},
            'recall_strict': {},
            'recall_soft': {}
        }

        for metric_name in ['recall', 'precision', 'ndcg', 'recall_strict', 'recall_soft']:
            for k in k_values:
                values = avg_metrics[metric_name][k]
                if values:
                    result[metric_name][k] = np.mean(values)
                else:
                    result[metric_name][k] = 0.0

        return result

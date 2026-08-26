import csv
import io
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional


def _parse_csv_row(row_str: str) -> List[str]:
    """
    Parse a CSV row string, properly handling quoted fields with commas.
    
    Args:
        row_str: CSV row as string
        
    Returns:
        List of cell values
    """
    # Use csv.reader to properly parse CSV with quotes
    csv_reader = csv.reader(io.StringIO(row_str), delimiter=',', quotechar='"', escapechar='\\')
    try:
        row = next(csv_reader)
        return [cell.strip() for cell in row]
    except StopIteration:
        return []


class DatasetLoader:
    """
    A unified loader for table QA datasets.

    Supports loading data from multi-table-retrieval and single-table-retrieval categories.

    Categories:
    - multi-table-retrieval: Queries that may require multiple tables (e.g., bird, mmqa_two_table)
    - single-table-retrieval: Queries based on single tables (e.g., e2ewtq, feta, ottqa)

    Each loaded sample contains:
    - 'question': The question string
    - 'ground_truth_tables': List of dicts with table identifiers {'id', 'file_name', 'sheet_name'}
    - 'answer': Answer string (SQL for multi-table, text for some single-table, empty for others)
    - 'table': Table data dict (single-table) or list of dicts (multi-table)
      Each table dict has: 'id', 'file_name', 'sheet_name', 'header' (list of str), 'instances' (list of list of str), 'metadata'
    """

    def __init__(self, base_path: str = None):
        project_root = Path(__file__).resolve().parents[2]
        self.base_path = base_path or str(project_root / "data")

    def load_data(self, category: str, dataset_name: str, split: str) -> List[Dict[str, Any]]:
        """
        Load dataset from the specified category, dataset name, and split.

        Args:
            category: 'multi-table-retrieval' or 'single-table-retrieval'
            dataset_name: Name of the dataset (e.g., 'bird', 'e2ewtq')
            split: 'train' or 'test'

        Returns:
            List of samples, each containing:
            - 'question': str
            - 'ground_truth_tables': List[Dict] with 'id', 'file_name', 'sheet_name'
            - 'answer': str or None
            - 'table': Dict or List[Dict] of table data with 'id', 'file_name', 'sheet_name', 'header', 'instances'
        """
        if category not in ['multi-table-retrieval', 'single-table-retrieval']:
            raise ValueError(f"Invalid category: {category}")

        if split not in ['train', 'test']:
            raise ValueError(f"Invalid split: {split}")

        # Path to the dataset
        dataset_path = os.path.join(self.base_path, category, split, dataset_name)
        query_file = os.path.join(dataset_path, 'query.jsonl')
        table_file = os.path.join(dataset_path, 'table.jsonl')

        if not os.path.exists(query_file) or not os.path.exists(table_file):
            raise FileNotFoundError(f"Files not found: {query_file} or {table_file}")

        # Load tables
        tables = self._load_tables(table_file)

        # Load queries
        queries = self._load_queries(query_file)

        # Combine
        data = []
        for query in queries:
            table_data = self._get_tables_for_query(query['ground_truth_list'], tables, category)
            if table_data is not None:  # Skip queries with missing tables
                sample = {
                    'question': query['question'],
                    'ground_truth_tables': query['ground_truth_list'],
                    'answer': query.get('answer') or query.get('sql') or "",  # Handle different answer fields
                    'table': table_data
                }
                data.append(sample)

        return data

    def _load_tables(self, table_file: str) -> Dict[int, Dict[str, Any]]:
        """Load table data from table.jsonl"""
        tables = {}
        with open(table_file, 'r', encoding='utf-8') as f:
            for line in f:
                table = json.loads(line.strip())
                table_id = table['id']
                # Parse header
                if isinstance(table['header'], str):
                    header = _parse_csv_row(table['header'])
                elif isinstance(table['header'], list) and len(table['header']) == 1 and isinstance(table['header'][0], str):
                    # Handle case where header is a list containing a single comma-separated string
                    header = _parse_csv_row(table['header'][0])
                else:
                    header = table['header']
                # Parse instances
                instances = []
                for inst in table['instances']:
                    if isinstance(inst, str):
                        row = _parse_csv_row(inst)
                    else:
                        row = inst
                    instances.append(row)
                tables[table_id] = {
                    'id': table_id,
                    'file_name': table['file_name'],
                    'sheet_name': table['sheet_name'],
                    'header': header,
                    'instances': instances,
                    'metadata': table.get('metadata', {})
                }
        return tables

    def _load_queries(self, query_file: str) -> List[Dict[str, Any]]:
        """Load query data from query.jsonl"""
        queries = []
        with open(query_file, 'r', encoding='utf-8') as f:
            for line in f:
                query = json.loads(line.strip())
                queries.append(query)
        return queries

    def _get_tables_for_query(self, ground_truth_list: List[Dict], tables: Dict[int, Dict], category: str) -> Any:
        """Get table data for the query"""
        table_ids = [gt['id'] for gt in ground_truth_list]
        table_data = [tables[tid] for tid in table_ids if tid in tables]

        if not table_data:
            return None

        # For single-table-retrieval, return the single table dict
        # For multi-table-retrieval, return list of table dicts
        if category == 'single-table-retrieval':
            return table_data[0] if len(table_data) == 1 else table_data
        else:
            return table_data



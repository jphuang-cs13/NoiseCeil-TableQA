#!/usr/bin/env python3
"""
Generate hard and soft negatives from retrieval results.

This script takes retrieval results and generates:
- Hard negatives: Top K retrieved tables that are NOT in the gold/ground truth set
- Soft negatives: K random tables that are NOT in the gold/ground truth set

Usage:
    python scripts/generate_hard_soft_negatives.py \
        --k 10 \
        --dataset e2ewtq \
        --retriever bge-m3 \
        --representation TextTableRepresentation \
        --max_length 512
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HardSoftNegativeGenerator:
    """Generate hard and soft negatives from retrieval results."""
    
    def __init__(
        self,
        k: int,
        dataset: str,
        retriever: str,
        representation: str,
        max_length: Optional[int] = None,
        base_dir: Path = None
    ):
        """
        Initialize the generator.
        
        Args:
            k: Number of negatives to extract
            dataset: Dataset name (e.g., 'e2ewtq')
            retriever: Retriever name (e.g., 'bge-m3')
            representation: Representation name (e.g., 'TextTableRepresentation')
            max_length: Max length for TextTableRepresentation (optional)
            base_dir: Base directory for project (default: current directory)
        """
        self.k = k
        self.dataset = dataset
        self.retriever = retriever
        self.representation = representation
        self.max_length = max_length
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parents[1]
        

        self.split = 'test'
        
        # Set up paths
        self._setup_paths()
        
    def _setup_paths(self):
        """Set up all necessary paths."""
        # Determine dataset category
        self.dataset_category = 'single-table-retrieval'
        
        # Data paths
        self.data_base = self.base_dir / 'data' / self.dataset_category
        self.query_file = self.data_base / self.split / self.dataset / 'query.jsonl'
        self.table_file = self.data_base / self.split / self.dataset / 'table.jsonl'
        
        # Retrieval results path
        self.retrieval_result_dir = self.base_dir / 'retrieval_results' / self.dataset
        
        # Output path
        self.output_dir = self.base_dir / 'hard_soft_negative' / self.dataset
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate paths
        self._validate_paths()
    
    def _validate_paths(self):
        """Validate that all required paths exist."""
        if not self.query_file.exists():
            raise FileNotFoundError(f"Query file not found: {self.query_file}")
        if not self.table_file.exists():
            raise FileNotFoundError(f"Table file not found: {self.table_file}")
        if not self.retrieval_result_dir.exists():
            raise FileNotFoundError(f"Retrieval result directory not found: {self.retrieval_result_dir}")
        
        logger.info(f"Query file: {self.query_file}")
        logger.info(f"Table file: {self.table_file}")
        logger.info(f"Retrieval result dir: {self.retrieval_result_dir}")

    def _relative_source_path(self, path: Path) -> str:
        """Return a repo-relative path for serialized metadata."""
        return str(path.relative_to(self.base_dir))
    
    def _find_retrieval_result_file(self) -> Path:
        """Find the retrieval result file matching the parameters."""
        # Build expected filename pattern
        retriever_name = self._normalize_retriever_name()
        
        # List all files in retrieval result directory
        for file in self.retrieval_result_dir.glob('*.jsonl'):
            filename = file.name
            
            # Check if filename contains all required components
            if (self.representation in filename and 
                retriever_name in filename and
                f'{self.split}' in filename):
                
                # For TextTableRepresentation, check max_length if provided
                if self.representation == 'TextTableRepresentation' and self.max_length:
                    if f'max_length={self.max_length}' in filename:
                        return file
                    # Also check for max_length in a different format
                    if f'_{self.max_length}_' in filename or f'_{self.max_length}.' in filename:
                        return file
                else:
                    return file
        
        raise FileNotFoundError(
            f"No retrieval result file found for: "
            f"{self.representation} + {retriever_name} + {self.split}"
        )
    
    def _normalize_retriever_name(self) -> str:
        """Normalize retriever name to match filename format."""
        mapping = {
            'bge-m3': 'BGEM3',
        }
        base_name = mapping.get(self.retriever, self.retriever)
        return f'{base_name}Retriever'
    
    def _load_queries(self) -> Dict[int, Dict]:
        """Load queries from query.jsonl."""
        queries = {}
        with open(self.query_file, 'r', encoding='utf-8') as f:
            for line in f:
                query = json.loads(line)
                queries[query['id']] = query
        
        logger.info(f"Loaded {len(queries)} queries")
        return queries
    
    def _load_all_table_ids(self) -> Set[str]:
        """Load all table IDs from table.jsonl."""
        table_ids = set()
        with open(self.table_file, 'r', encoding='utf-8') as f:
            for line in f:
                table = json.loads(line)
                table_ids.add(table['id'])
        
        logger.info(f"Loaded {len(table_ids)} table IDs")
        return table_ids
    
    def _load_retrieval_results(self, retrieval_file: Path) -> List[Dict]:
        """Load retrieval results from file."""
        results = []
        with open(retrieval_file, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                results.append(result)
        
        logger.info(f"Loaded {len(results)} retrieval results from {retrieval_file.name}")
        return results
    
    def _extract_gold_tables(self, query: Dict) -> Set[str]:
        """Extract gold table IDs from ground_truth_list and normalize to strings."""
        gold_tables = set()
        if 'ground_truth_list' in query:
            for gt in query['ground_truth_list']:
                # Normalize to string for consistent comparison
                gold_tables.add(str(gt['id']))
        return gold_tables
    
    def _generate_hard_negatives(self, retrieval_result: Dict, gold_tables: Set[str]) -> List[Dict]:
        """Extract hard negatives (top K non-gold retrieved tables)."""
        # Normalize gold tables to strings for comparison
        gold_tables_str = {str(g) for g in gold_tables}
        
        hard_negatives = []
        for table in retrieval_result['retrieved_tables']:
            # Normalize table id to string for comparison
            table_id_str = str(table['id'])
            if table_id_str not in gold_tables_str:
                hard_negatives.append({
                    'rank': table['rank'],
                    'id': table['id']
                })
                if len(hard_negatives) >= self.k:
                    break
        
        return hard_negatives[:self.k]
    
    def _generate_soft_negatives(self, gold_tables: Set[str], all_table_ids: Set[str]) -> List[Dict]:
        """Generate soft negatives (random K non-gold tables)."""
        # Normalize gold tables to strings for comparison
        gold_tables_str = {str(g) for g in gold_tables}
        all_table_ids_str = {str(t) for t in all_table_ids}
        
        # Get candidate tables (all except gold)
        candidate_tables = all_table_ids_str - gold_tables_str
        
        if len(candidate_tables) < self.k:
            logger.warning(
                f"Not enough candidate tables for soft negatives. "
                f"Requested: {self.k}, Available: {len(candidate_tables)}"
            )
            # Return all available if less than K
            soft_negatives = list(candidate_tables)
        else:
            # Randomly sample K tables
            soft_negatives = random.sample(list(candidate_tables), self.k)
        
        return [{'id': table_id} for table_id in soft_negatives]
    
    def generate(self) -> Dict[str, any]:
        """Generate hard and soft negatives."""
        logger.info("Starting hard/soft negative generation")
        
        # Find retrieval result file
        retrieval_file = self._find_retrieval_result_file()
        logger.info(f"Using retrieval result file: {retrieval_file.name}")
        
        # Load data
        queries = self._load_queries()
        all_table_ids = self._load_all_table_ids()
        retrieval_results = self._load_retrieval_results(retrieval_file)
        
        # Generate negatives
        output_data = []
        total_hard_negatives = 0
        total_soft_negatives = 0
        
        for retrieval_result in retrieval_results:
            # Find corresponding query
            question = retrieval_result['question']
            matching_queries = [q for q in queries.values() if q['question'] == question]
            
            if not matching_queries:
                logger.warning(f"No matching query for question: {question[:50]}...")
                continue
            
            query = matching_queries[0]
            gold_tables = self._extract_gold_tables(query)
            
            # Generate hard and soft negatives
            hard_negatives = self._generate_hard_negatives(retrieval_result, gold_tables)
            soft_negatives = self._generate_soft_negatives(gold_tables, all_table_ids)
            
            total_hard_negatives += len(hard_negatives)
            total_soft_negatives += len(soft_negatives)
            
            # Create output record
            output_record = {
                'query_id': query['id'],
                'question': question,
                'gold_tables': list(gold_tables),
                'hard_negatives': hard_negatives,
                'soft_negatives': soft_negatives,
                'metadata': {
                    'dataset': self.dataset,
                    'category': self.dataset_category,
                    'split': self.split,
                    'representation': self.representation,
                    'retriever': self.retriever,
                    'max_length': self.max_length,
                    'k': self.k,
                    'sources': {
                        'retrieval_results': self._relative_source_path(retrieval_file),
                        'query_data': self._relative_source_path(self.query_file),
                        'table_data': self._relative_source_path(self.table_file)
                    }
                }
            }
            
            output_data.append(output_record)
        
        logger.info(f"Generated hard/soft negatives for {len(output_data)} queries")
        logger.info(f"Total hard negatives: {total_hard_negatives}")
        logger.info(f"Total soft negatives: {total_soft_negatives}")
        
        return {
            'data': output_data,
            'summary': {
                'total_queries': len(output_data),
                'total_hard_negatives': total_hard_negatives,
                'total_soft_negatives': total_soft_negatives,
                'avg_hard_negatives_per_query': total_hard_negatives / len(output_data) if output_data else 0,
                'avg_soft_negatives_per_query': total_soft_negatives / len(output_data) if output_data else 0
            }
        }
    
    def save(self, output_data: Dict):
        """Save generated negatives to file."""
        # Generate output filename (following retrieval_results naming convention)
        output_filename = self._generate_output_filename()
        output_file = self.output_dir / output_filename
        
        # Save data
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in output_data['data']:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved hard/soft negatives to {output_file}")
        logger.info(f"Summary: {output_data['summary']}")
        
        return output_file
    
    def _generate_output_filename(self) -> str:
        """Generate output filename following retrieval_results naming convention."""
        # Build representation params string
        rep_params = 'include_file_name=True_include_sheet_name=True'
        if self.representation == 'TextTableRepresentation' and self.max_length:
            rep_params += f'_max_length={self.max_length}'
        elif self.representation in ['TopNRowsRepresentation', 'MiddleNRowsRepresentation', 'BottomNRowsRepresentation']:
            rep_params += '_n_rows=10'
        
        # Normalize retriever name
        retriever_name = self._normalize_retriever_name()
        
        # Build filename
        filename = f"{self.representation}_{rep_params}_{retriever_name}_{self.split}_k{self.k}_hard_soft_negative.jsonl"
        
        return filename


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate hard and soft negatives from retrieval results'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of negatives to extract (default: 10)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['feta', 'e2ewtq', 'ottqa'],
        help='Dataset name'
    )
    parser.add_argument(
        '--retriever',
        type=str,
        required=True,
        choices=['bge-m3'],
        help='Retriever name'
    )
    parser.add_argument(
        '--representation',
        type=str,
        required=True,
        choices=['TextTableRepresentation'],
        help='Representation name'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=None,
        help='Max length for TextTableRepresentation (optional)'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default=None,
        help='Base directory for project (default: current directory)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create generator
        generator = HardSoftNegativeGenerator(
            k=args.k,
            dataset=args.dataset,
            retriever=args.retriever,
            representation=args.representation,
            max_length=args.max_length,
            base_dir=args.base_dir
        )
        
        # Generate negatives
        output_data = generator.generate()
        
        # Save results
        generator.save(output_data)
        
        logger.info("Hard/soft negative generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during generation: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

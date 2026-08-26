"""
Base classes for table retrievers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class TableRetriever(ABC):
    """
    Abstract base class for table retrieval models.

    Subclasses should implement methods for encoding indices and retrieving tables.
    """

    @abstractmethod
    def encode_indices(self, tables: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Encode table indices for retrieval.

        Args:
            tables: List of table data dicts
            **kwargs: Additional parameters

        Returns:
            Encoded indices (e.g., vector store, index object)
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, indices: Any, top_k: int = 5, **kwargs) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve relevant tables for a query.

        Args:
            query: The search query
            indices: The encoded indices
            top_k: Number of top results to return
            **kwargs: Additional parameters

        Returns:
            List of (table_dict, score) tuples
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str = None):
        """Load the retrieval model."""
        pass

    @abstractmethod
    def unload_model(self):
        """Unload the retrieval model to free resources."""
        pass
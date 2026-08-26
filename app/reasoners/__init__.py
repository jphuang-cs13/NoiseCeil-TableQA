"""
Table Reasoner interfaces and implementations.

This module provides:
1. TableReasoner - Abstract base class for reasoners
2. LLMReasoner - LLM-based reasoner with flexible prompt strategies
3. Prompt strategies - Various prompt engineering approaches

Usage:
    # Using LLMReasoner with different strategies
    reasoner = LLMReasoner(strategy_name='chain-of-thought')
    reasoner.load_model()
    answer = reasoner.reason(question, retrieved_tables)
    
    # With few-shot examples
    reasoner = LLMReasoner(
        strategy_name='few-shot',
        examples=[
            {'question': '...', 'answer': '...'},
            {'question': '...', 'answer': '...'},
        ]
    )
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Import new implementations
from .prompt_strategies import (
    PromptStrategy, PromptTemplate,
    ZeroShotStrategy,
    create_strategy
)
from .llm_reasoner import LLMReasoner, BaseTableReasoner

class TableReasoner(ABC):
    """
    Abstract base class for table reasoning models.

    Subclasses should implement the reason method to perform reasoning
    on questions given relevant tables.
    """

    @abstractmethod
    def reason(self, question: str, tables: List[Dict[str, Any]], **kwargs) -> str:
        """
        Perform reasoning on the question using the provided tables.

        Args:
            question: The question to answer
            tables: List of table data (from DatasetLoader)
            **kwargs: Additional parameters

        Returns:
            The reasoned answer as a string
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str = None):
        """Load the reasoning model."""
        pass

    @abstractmethod
    def unload_model(self):
        """Unload the reasoning model to free resources."""
        pass


__all__ = [
    # Base classes
    'TableReasoner',
    # LLM-based reasoner
    'LLMReasoner',
    'BaseTableReasoner',
    # Prompt strategies
    'PromptStrategy',
    'PromptTemplate',
    'ZeroShotStrategy',
    'create_strategy',    
]

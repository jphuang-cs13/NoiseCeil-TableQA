"""
Prompt engineering strategies for Table QA reasoning.

This module defines various prompt strategies that can be used with reasoners.
Each strategy implements a different approach to prompt engineering:
- Zero-shot: Direct question answering without examples
- One-shot: Single example demonstration
- Few-shot: Multiple examples for in-context learning
- Chain-of-Thought: Step-by-step reasoning
- Role-play: Specific role definition in system prompt

Usage:
    strategy = FewShotStrategy(examples=[...])
    prompt = strategy.format_prompt(
        question="...",
        tables=[...],
        role="SQL Expert"
    )
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path


class PromptTemplate:
    """
    Template for managing system and user prompts.
    
    This class separates concerns between system-level instructions
    (role, context, rules) and user-level input (question, examples, data).
    """
    
    def __init__(self,
                 system_prompt: str = "",
                 user_prompt_template: str = ""):
        """
        Initialize prompt template.
        
        Args:
            system_prompt: System-level instructions (role, rules, context)
            user_prompt_template: User prompt with placeholders (e.g., {question}, {tables})
        """
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template


# Prompts directory loader (loads optional .txt templates from project-level prompts/)
_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

def _load_prompt(fname: str) -> Optional[str]:
    try:
        with open(_PROMPTS_DIR / fname, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return None
    
    def format(self, **kwargs) -> tuple[str, str]:
        """
        Format both system and user prompts.
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system = self.system_prompt.format(**kwargs) if self.system_prompt else ""
        user = self.user_prompt_template.format(**kwargs) if self.user_prompt_template else ""
        return system, user
    
    def set_system_prompt(self, prompt: str):
        """Update system prompt."""
        self.system_prompt = prompt
    
    def set_user_prompt_template(self, prompt: str):
        """Update user prompt template."""
        self.user_prompt_template = prompt


class PromptStrategy(ABC):
    """
    Abstract base class for prompt strategies.
    
    Each strategy defines how to construct prompts for different reasoning approaches.
    """
    
    @abstractmethod
    def format_prompt(self,
                     question: str,
                     tables: List[Dict[str, Any]],
                     role: Optional[str] = None,
                     **kwargs) -> tuple[str, str]:
        """
        Format prompt according to this strategy.
        
        Args:
            question: The question to answer
            tables: List of retrieved table data
            role: Optional role definition for role-playing (e.g., "SQL Expert", "Data Analyst")
            **kwargs: Additional strategy-specific parameters
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        pass
    
    def _format_tables(self, tables: List[Dict[str, Any]], max_chars: Optional[int] = None) -> str:
        """
        Helper method to format tables for inclusion in prompts.
        
        Args:
            tables: List of table data
            max_chars: Optional character limit for table content
        
        Returns:
            Formatted table string
        """
        result = []
        for i, table in enumerate(tables, 1):
            # Use pre-formatted representation if available
            if 'representation' in table:
                result.append(f"Table {i}:\n{table['representation']}")
            # Otherwise format from raw data
            else:
                header = table.get('header', [])
                instances = table.get('instances', [])
                
                # Format header
                header_str = ', '.join(header) if isinstance(header, list) else str(header)
                
                # Format rows: convert each row to string if it's a list
                formatted_rows = []
                for row in instances[:5]:  # Limit to first 5 rows
                    if isinstance(row, list):
                        formatted_rows.append(" | ".join(str(cell) for cell in row))
                    else:
                        formatted_rows.append(str(row))
                
                result.append(f"Table {i} (columns: {header_str}):\n" +
                            "\n".join(formatted_rows))
        
        formatted = "\n\n".join(result)
        if max_chars:
            formatted = formatted[:max_chars]
        return formatted

    def get_tool_schemas(self, **kwargs) -> List[Dict[str, Any]]:
        """Return function-calling tool schemas for this strategy (default: none)."""
        return []

    def init_tool_runtime(self,
                          question: str,
                          tables: List[Dict[str, Any]],
                          role: Optional[str] = None,
                          **kwargs) -> Optional[Dict[str, Any]]:
        """Initialize mutable runtime state for tool calls (default: none)."""
        return None

    def execute_tool_call(self,
                          tool_name: str,
                          arguments: Dict[str, Any],
                          runtime: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """Execute a function-calling tool call.

        Strategies that expose tool schemas should override this method.
        """
        raise NotImplementedError(f"Tool execution not implemented for strategy: {self.__class__.__name__}")


class ZeroShotStrategy(PromptStrategy):
    """
    Zero-shot strategy: Direct question answering without examples.
    
    Simple and straightforward: just ask the question directly.
    Best for: Well-structured problems, experienced models
    """
    
    def __init__(self, role_template: Optional[str] = None):
        """
        Initialize zero-shot strategy.
        
        Args:
            role_template: Optional system prompt template for role-playing.
                          Use {role} placeholder for dynamic role insertion.
        """
        self.role_template = role_template or "You are a {role} assistant that answers questions about tables."
    
    def format_prompt(self,
                     question: str,
                     tables: List[Dict[str, Any]],
                     role: Optional[str] = None,
                     **kwargs) -> tuple[str, str]:
        """
        Format zero-shot prompt.
        
        System: Role definition (or generic assistant message)
        User: Question + tables, no examples
        """
        # Attempt to load system/user templates from prompts/; fall back to defaults
        system_template = _load_prompt("zero_shot_system.txt")
        user_template = _load_prompt("zero_shot_user.txt")

        # System prompt with role
        if system_template:
            system_prompt = system_template.format(role=role or 'helpful')
        else:
            system_prompt = self.role_template.format(role=role) if role else "You are a helpful assistant that answers questions about tables."

        # User prompt: just the question and tables
        tables_str = self._format_tables(tables, max_chars=kwargs.get('max_table_chars'))

        if user_template:
            user_prompt = user_template.format(question=question, tables_str=tables_str)
        else:
            # Handle empty tables gracefully
            if tables_str.strip():
                user_prompt = f"""Answer the following question about the given table(s).

Question: {question}

Tables:
{tables_str}

Answer:"""
            else:
                user_prompt = f"""Answer the following question.

Question: {question}

Answer:"""

        return system_prompt, user_prompt

# Strategy factory for easy instantiation
STRATEGY_REGISTRY = {
    'zero-shot': ZeroShotStrategy,
}


def create_strategy(strategy_name: str, **kwargs) -> PromptStrategy:
    """
    Factory function to create prompt strategies.
    
    Args:
        strategy_name: Name of strategy ('zero-shot', 'few-shot', etc.)
        **kwargs: Strategy-specific parameters
    
    Returns:
        Initialized PromptStrategy instance
    """
    strategy_class = STRATEGY_REGISTRY.get(strategy_name.lower())
    if not strategy_class:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Available: {list(STRATEGY_REGISTRY.keys())}")
    return strategy_class(**kwargs)

"""
LLM-based reasoner with flexible prompt engineering strategies.

This module implements reasoners that use LLM providers for Table QA reasoning,
integrated with flexible prompt strategies for experimentation.
"""

from typing import Any, Dict, List, Optional
import os
import time
import json
from llm.llm_client import LLMClient
from llm.llm_logger import LLMLogger
from .prompt_strategies import PromptStrategy, ZeroShotStrategy, create_strategy


class LLMReasoner:
    """
    Reasoner implementation using LLM providers with flexible prompt strategies.
    
    This reasoner:
    1. Accepts different prompt strategies for experimentation
    2. Integrates with LLMClient for multi-provider support
    3. Manages system and user prompts flexibly
    4. Supports role-playing and custom system prompts
    
    Usage:
        # With default zero-shot strategy
        reasoner = LLMReasoner(strategy_name='zero-shot')
        
        # With few-shot examples
        examples = [
            {'question': 'Q1?', 'answer': 'A1'},
            {'question': 'Q2?', 'answer': 'A2'},
        ]
        reasoner = LLMReasoner(strategy_name='few-shot', examples=examples)
        
        # With custom strategy
        reasoner = LLMReasoner(strategy_name='chain-of-thought')
        
        # Get answer for a question
        answer = reasoner.reason(question, retrieved_tables)
    """
    
    def __init__(self,
                 strategy: Optional[PromptStrategy] = None,
                 strategy_name: Optional[str] = None,
                 provider: Optional[str] = None,
                 role: Optional[str] = None,
                 system_prompt_override: Optional[str] = None,
                 max_table_chars: Optional[int] = None,
                 temperature: float = 0.7,
                 max_tokens: Optional[int] = None,
                 enable_logging: bool = True,
                 **strategy_kwargs):
        """
        Initialize LLM-based reasoner.
        
        Args:
            strategy: Explicit PromptStrategy instance (overrides strategy_name)
            strategy_name: Name of strategy to use ('zero-shot', 'few-shot', etc.)
            provider: LLM provider ('groq', 'openai', 'google', etc.)
                     If None, uses LLM_PROVIDER from env
            role: Role definition for role-playing (e.g., "SQL Expert", "Data Analyst")
            system_prompt_override: Override the strategy's system prompt entirely
            max_table_chars: Maximum characters to include from tables in prompt
            temperature: LLM temperature for response generation
            max_tokens: Maximum tokens in response
            enable_logging: Whether to log LLM usage statistics (default: True)
            **strategy_kwargs: Additional arguments for strategy instantiation
                             (e.g., examples for few-shot)
        """
        # Initialize strategy
        if strategy:
            self.strategy = strategy
        elif strategy_name:
            self.strategy = create_strategy(strategy_name, **strategy_kwargs)
        else:
            self.strategy = ZeroShotStrategy()
        
        # Initialize LLM client
        self.provider = provider or os.getenv('LLM_PROVIDER', 'groq')
        self.llm_client = LLMClient(provider=self.provider)
        
        # Initialize logger with strategy-specific stage name
        self.enable_logging = enable_logging
        
        # Compute strategy log name for potential future use
        strategy_class_name = self.strategy.__class__.__name__
        self._strategy_log_name = self._strategy_class_to_log_name(strategy_class_name)
        
        if enable_logging:
            # Create logger with strategy as subdirectory
            self.logger = LLMLogger(stage=f'reasoning/{self._strategy_log_name}')
        else:
            self.logger = None
        
        # Configuration
        self.role = role
        self.system_prompt_override = system_prompt_override
        self.max_table_chars = max_table_chars
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.loaded = False
    
    def load_model(self, model_path: str = None):
        """Load LLM model (initializes LLMClient)."""
        self.llm_client.load_model()
        self.loaded = True
    
    def unload_model(self):
        """Unload LLM model."""
        self.llm_client.unload_model()
        self.loaded = False
    
    @staticmethod
    def _strategy_class_to_log_name(class_name: str) -> str:
        """
        Convert strategy class name to log directory name.
        
        E.g., ZeroShotStrategy -> zero-shot, SQLStrategy -> sql
        """
        # Remove 'Strategy' suffix
        if class_name.endswith('Strategy'):
            name = class_name[:-8]  # Remove 'Strategy'
        else:
            name = class_name
        
        # Special cases for acronyms
        if name == 'SQL':
            return 'sql'
        
        # Convert camelCase to kebab-case
        result = []
        for i, char in enumerate(name):
            if char.isupper():
                if i > 0 and name[i-1].islower():
                    # Previous char is lowercase, add dash
                    result.append('-')
                elif i > 0 and i + 1 < len(name) and name[i + 1].islower():
                    # Next char is lowercase, add dash
                    result.append('-')
                result.append(char.lower())
            else:
                result.append(char)
        
        return ''.join(result)
    
    def reason(self, question: str, tables: List[Dict[str, Any]], return_metadata: bool = False, **kwargs) -> str | Dict[str, Any]:
        """
        Perform reasoning using LLM and current prompt strategy.
        
        Args:
            question: Question to answer
            tables: Retrieved table data
            return_metadata: If True, return dict with text, tokens, and timing info
            **kwargs: Additional parameters (e.g., custom_role for runtime role override)
        
        Returns:
            LLM-generated answer (str) or dict with metadata if return_metadata=True
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Track execution time and token usage
        start_time = time.time()
        
        # Prepare role (can be overridden at runtime)
        role = kwargs.pop('custom_role', None) or self.role
        
        # Prepare strategy parameters
        strategy_kwargs = {
            'max_table_chars': self.max_table_chars,
            **kwargs
        }
        
        # Get formatted prompts from strategy
        system_prompt, user_prompt = self.strategy.format_prompt(
            question=question,
            tables=tables,
            role=role,
            **strategy_kwargs
        )
        
        # Override system prompt if specified
        if self.system_prompt_override:
            system_prompt = self.system_prompt_override
        
        # Call LLM with formatted prompts and request full response with token info
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        tool_schemas: List[Dict[str, Any]] = []
        if hasattr(self.strategy, 'get_tool_schemas'):
            try:
                tool_schemas = self.strategy.get_tool_schemas(
                    question=question,
                    tables=tables,
                    role=role,
                    **strategy_kwargs
                ) or []
            except Exception:
                tool_schemas = []

        tool_runtime = None
        if tool_schemas and hasattr(self.strategy, 'init_tool_runtime'):
            try:
                tool_runtime = self.strategy.init_tool_runtime(
                    question=question,
                    tables=tables,
                    role=role,
                    **strategy_kwargs
                )
            except Exception:
                tool_runtime = None

        aggregated_prompt_tokens = 0
        aggregated_completion_tokens = 0
        used_tool_loop = False

        if tool_schemas:
            used_tool_loop = True
            max_tool_iterations = int(strategy_kwargs.get('max_tool_iterations', 6))
            final_response: Optional[Dict[str, Any]] = None

            for _ in range(max_tool_iterations):
                response = self.llm_client.chat_completion(
                    model=self.llm_client.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=tool_schemas,
                    tool_choice='auto',
                    return_full_response=True
                )

                usage = response.get('usage', {}) if isinstance(response, dict) else {}
                aggregated_prompt_tokens += int(usage.get('prompt_tokens', 0) or 0)
                aggregated_completion_tokens += int(usage.get('completion_tokens', 0) or 0)

                if isinstance(response, dict) and 'error' in response:
                    if return_metadata:
                        return {
                            'text': response.get('text', ''),
                            'request_tokens': aggregated_prompt_tokens,
                            'response_tokens': aggregated_completion_tokens,
                            'execution_time': response.get('execution_time', time.time() - start_time),
                            'error': response.get('error')
                        }
                    raise RuntimeError(f"LLM error: {response.get('error')}")

                tool_calls = response.get('tool_calls', []) if isinstance(response, dict) else []
                assistant_text = response.get('text', '') if isinstance(response, dict) else str(response)

                if not tool_calls:
                    final_response = response if isinstance(response, dict) else {'text': assistant_text, 'usage': {}}
                    break

                messages.append({
                    'role': 'assistant',
                    'content': assistant_text or '',
                    'tool_calls': tool_calls,
                })

                for tool_call in tool_calls:
                    function_block = tool_call.get('function', {}) if isinstance(tool_call, dict) else {}
                    tool_name = function_block.get('name', '')
                    raw_arguments = function_block.get('arguments', '{}')

                    if isinstance(raw_arguments, str):
                        try:
                            arguments = json.loads(raw_arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    elif isinstance(raw_arguments, dict):
                        arguments = raw_arguments
                    else:
                        arguments = {}

                    try:
                        tool_result = self.strategy.execute_tool_call(
                            tool_name=tool_name,
                            arguments=arguments,
                            runtime=tool_runtime,
                            question=question,
                            tables=tables,
                            role=role,
                            **strategy_kwargs
                        )
                    except Exception as exc:
                        tool_result = {
                            'status': 'error',
                            'tool_name': tool_name,
                            'error': str(exc),
                        }

                    tool_result_content = tool_result if isinstance(tool_result, str) else json.dumps(tool_result, ensure_ascii=False)
                    messages.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.get('id', ''),
                        'name': tool_name,
                        'content': tool_result_content,
                    })

            if final_response is None:
                error_msg = f"Maximum tool iterations reached without final answer ({max_tool_iterations})"
                if return_metadata:
                    return {
                        'text': '',
                        'request_tokens': aggregated_prompt_tokens,
                        'response_tokens': aggregated_completion_tokens,
                        'execution_time': time.time() - start_time,
                        'error': error_msg
                    }
                raise RuntimeError(error_msg)

            response = final_response
        else:
            response = self.llm_client.chat_completion(
                model=self.llm_client.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                return_full_response=True  # Request full response with token info
            )
        
        # Calculate execution time
        execution_time = time.time() - start_time

        # If underlying LLM client returned an error payload, propagate it
        if isinstance(response, dict) and 'error' in response:
            if return_metadata:
                return {
                    'text': response.get('text', ''),
                    'request_tokens': response.get('usage', {}).get('prompt_tokens', 0),
                    'response_tokens': response.get('usage', {}).get('completion_tokens', 0),
                    'execution_time': response.get('execution_time', execution_time),
                    'error': response.get('error')
                }
            else:
                # For non-metadata calls, raise to let caller decide
                raise RuntimeError(f"LLM error: {response.get('error')}")

        # Extract text and token usage from response
        if isinstance(response, dict) and 'text' in response:
            result_text = response['text']
            usage = response.get('usage', {})
            request_tokens = usage.get('prompt_tokens', 0)
            response_tokens = usage.get('completion_tokens', 0)
            response_execution_time = response.get('execution_time', execution_time)

            if used_tool_loop:
                request_tokens = aggregated_prompt_tokens
                response_tokens = aggregated_completion_tokens
                response_execution_time = execution_time
            
            # If API didn't provide token counts, estimate them
            if request_tokens == 0 and response_tokens == 0:
                # Estimate tokens: roughly 1 token per 4 characters
                prompt_text = system_prompt + user_prompt
                request_tokens = len(prompt_text) // 4
                # Estimate response tokens based on response length
                response_tokens = len(result_text) // 4
            
            # Log usage statistics
            if self.enable_logging and self.logger:
                strategy_name = self.strategy.__class__.__name__
                notes = f"strategy={strategy_name}"
                if self.role:
                    notes += f", role={self.role}"
                
                self.logger.log_usage(
                    model=self.llm_client.model,
                    provider=self.provider,
                    request_tokens=request_tokens,
                    response_tokens=response_tokens,
                    execution_time=response_execution_time,
                    notes=notes
                )
            
            if return_metadata:
                return {
                    'text': result_text,
                    'request_tokens': request_tokens,
                    'response_tokens': response_tokens,
                    'execution_time': response_execution_time
                }
            else:
                return result_text
        else:
            # Fallback if response is just text (shouldn't happen with return_full_response=True)
            result_text = str(response)
            
            # Estimate tokens for failed requests
            prompt_text = system_prompt + user_prompt
            estimated_request_tokens = len(prompt_text) // 4
            estimated_response_tokens = len(result_text) // 4 if result_text else 0
            
            # Log even if no token info available
            if self.enable_logging and self.logger:
                strategy_name = self.strategy.__class__.__name__
                notes = f"strategy={strategy_name}"
                if self.role:
                    notes += f", role={self.role}"
                
                self.logger.log_usage(
                    model=self.llm_client.model,
                    provider=self.provider,
                    request_tokens=estimated_request_tokens,
                    response_tokens=estimated_response_tokens,
                    execution_time=execution_time,
                    notes=notes
                )
            
            if return_metadata:
                return {
                    'text': result_text,
                    'request_tokens': estimated_request_tokens,
                    'response_tokens': estimated_response_tokens,
                    'execution_time': execution_time
                }
            else:
                return result_text
    
    def set_strategy(self, strategy: PromptStrategy):
        """
        Change the prompt strategy at runtime.
        
        Useful for experimenting with different strategies without recreating reasoner.
        """
        self.strategy = strategy
    
    def set_role(self, role: str):
        """Set the role for role-playing."""
        self.role = role
    
    def set_system_prompt_override(self, prompt: str):
        """Override the strategy's system prompt entirely."""
        self.system_prompt_override = prompt
    
    def clear_system_prompt_override(self):
        """Remove system prompt override, use strategy's prompt again."""
        self.system_prompt_override = None
    
    def disable_logging(self):
        """Disable logging for subsequent requests."""
        self.enable_logging = False
    
    def enable_logging_feature(self):
        """Enable logging for subsequent requests."""
        self.enable_logging = True
        if not self.logger and self._strategy_log_name:
            self.logger = LLMLogger(stage=f'reasoning/{self._strategy_log_name}')
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """
        Get all logged requests.
        
        Returns:
            List of log entries with strategy, tokens, and execution time
        """
        if not self.logger:
            return []
        return self.logger.get_all_logs()
    
    def get_log_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of all logged requests.
        
        Returns:
            Dictionary with aggregated statistics
        """
        if not self.logger:
            return {}
        return self.logger.get_summary()


class BaseTableReasoner:
    """
    Base reasoner class maintaining backward compatibility.
    
    Can be subclassed for specific reasoning approaches.
    """
    
    def __init__(self):
        self.loaded = False
    
    def load_model(self, model_path: str = None):
        self.loaded = True
    
    def unload_model(self):
        self.loaded = False
    
    def reason(self, question: str, tables: List[Dict[str, Any]], **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement reason()")



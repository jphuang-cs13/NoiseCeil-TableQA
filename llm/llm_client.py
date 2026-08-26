"""
Flexible LLM API Client

Supports multiple LLM providers including:
- Ollama (self-hosted with OpenAI compatibility)
- OpenAI
- Anthropic Claude
- Groq (OpenAI-compatible API)
- Google Gemini (OpenAI-compatible API)
- GitHub Models (OpenAI-compatible API)
- Cerebras (OpenAI-compatible API)
- OpenRouter (OpenAI-compatible API)
- Fireworks (OpenAI-compatible API)
- Other OpenAI-compatible APIs

Features:
- Automatic retry with exponential backoff
- Rate limit and token usage monitoring
- Provider-specific configuration
- Usage logging and tracking

Usage:
    from llm.llm_client import LLMClient

    # For Ollama
    client = LLMClient(provider='ollama', base_url='https://ollama.com/api', api_key='')
    response = client.chat_completion(model='llama3.1:8b-instruct-fp16', messages=[...])

    # Check rate limits
    limits = client.check_limits()
    print(f"Status: {limits['status']}, Remaining: {limits['rate_limit_remaining']}")

    # For OpenAI
    client = LLMClient(provider='openai', api_key='your-key')
    response = client.chat_completion(model='gpt-4', messages=[...])

    # For Groq
    client = LLMClient(provider='groq', api_key='your-groq-key')
    response = client.chat_completion(model='llama-3.3-70b-versatile', messages=[...])

    # For Google Gemini
    client = LLMClient(provider='google', api_key='your-google-key')
    response = client.chat_completion(model='gemini-pro', messages=[...])

    # For GitHub Models
    client = LLMClient(provider='github', api_key='your-github-token')
    response = client.chat_completion(model='openai/gpt-4o', messages=[...])

    # For cerebras Models
    client = LLMClient(provider='cerebras', api_key='your-cerebras-key')
    response = client.chat_completion(model='gpt-oss-120b', messages=[...])

    # For Fireworks Models
    client = LLMClient(provider='fireworks', api_key='your-fireworks-key')
    response = client.chat_completion(model='accounts/fireworks/models/llama-v3p1-405b-instruct', messages=[...])

    # For OpenRouter provider routing
    client = LLMClient(provider='openrouter', api_key='your-openrouter-key')
    response = client.chat_completion(
        model='openai/gpt-5-mini',
        messages=[...],
        openrouter_provider={'only': ['azure']}
    )
    
    # Get display-friendly model name
    display_name = client.get_display_model_name('accounts/fireworks/models/llama-v3p1-405b-instruct')
    print(f"Display name: {display_name}")  # Output: llama-v3p1-405b-instruct

Environment Variables (.env):
    LLM_PROVIDER=groq              # Provider: ollama, openai, anthropic, groq, google, github, cerebras, openrouter, fireworks, custom
    BASE_URL=<api-url>             # For custom/ollama providers (optional for groq/google/github)
    API_KEY=<api-key>              # API key for authentication
    LLM_MODEL=llama-3.3-70b-versatile  # Default model name

Provider Base URLs (auto-configured):
    - ollama:   {base_url}/v1/
    - openai:   https://api.openai.com/v1
    - groq:     https://api.groq.com/openai/v1
    - google:   https://generativelanguage.googleapis.com/v1beta/openai/
    - github:   https://models.github.ai/inference
    - cerebras: https://api.cerebras.ai/v1
    - openrouter: https://openrouter.ai/api/v1
    - fireworks: https://api.fireworks.ai/inference/v1
    - custom:   {base_url}
"""

import requests
import json
import time
import functools
import os
from typing import List, Dict, Callable, Any, Optional, Tuple
from openai import OpenAI
import anthropic
from dotenv import load_dotenv
from llm.llm_logger import LLMLogger

# Load environment variables from .env file
load_dotenv()


# Provider-specific base URLs (auto-configured)
PROVIDER_BASE_URLS = {
    'ollama': None,  # Use BASE_URL from env or default
    'openai': 'https://api.openai.com/v1',
    'groq': 'https://api.groq.com/openai/v1',
    'google': 'https://generativelanguage.googleapis.com/v1beta/openai/',
    'github': 'https://models.github.ai/inference',
    'cerebras': 'https://api.cerebras.ai/v1',
    'openrouter': 'https://openrouter.ai/api/v1',
    'fireworks': 'https://api.fireworks.ai/inference/v1',
    'anthropic': None,  # Not used for OpenAI client
    'custom': None,  # Use BASE_URL from env
}


def load_provider_config(provider: str) -> Tuple[str, str, str]:
    """
    Load provider-specific configuration from environment variables.
    
    Supports both generic and provider-specific parameter names:
    - Generic: BASE_URL, API_KEY, LLM_MODEL
    - Provider-specific: {PROVIDER}_BASE_URL, {PROVIDER}_API_KEY, {PROVIDER}_LLM_MODEL
    
    Priority: Provider-specific > Generic > Default
    
    Args:
        provider: Provider name (e.g., 'groq', 'google', 'openai')
    
    Returns:
        Tuple of (base_url, api_key, model)
    """
    provider_upper = provider.upper()
    
    # Load API key (provider-specific takes priority)
    # Special handling for GitHub which uses TOKEN instead of API_KEY
    if provider == 'github':
        api_key = os.getenv('GITHUB_TOKEN') or os.getenv('API_KEY') or 'your_github_token'
    elif provider == 'cerebras':
        # Support both CEREBRAS_API_KEY and CEREBRAS_TOKEN from .env
        api_key = (
            os.getenv('CEREBRAS_API_KEY')
            or os.getenv('CEREBRAS_TOKEN')
            or os.getenv('API_KEY')
            or 'your_api_key'
        )
    elif provider == 'openrouter':
        api_key = (
            os.getenv('OPENROUTER_API_KEY')
            or os.getenv('API_KEY')
            or 'your_api_key'
        )
    else:
        api_key = os.getenv(f'{provider_upper}_API_KEY') or os.getenv('API_KEY') or 'your_api_key'
    
    # Load model (provider-specific takes priority)
    model = os.getenv(f'{provider_upper}_LLM_MODEL') or os.getenv('LLM_MODEL') or 'default-model'
    
    # Load base URL (provider-specific takes priority)
    # Use provider default if available
    default_base_url = PROVIDER_BASE_URLS.get(provider)
    base_url = (
        os.getenv(f'{provider_upper}_BASE_URL') or
        os.getenv('BASE_URL') or
        default_base_url or
        'https://api.example.com'  # Fallback
    )
    
    return base_url, api_key, model


def _parse_env_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def load_openrouter_provider_config() -> Optional[Dict[str, Any]]:
    """
    Load default OpenRouter provider routing from environment variables.

    Supported environment variables:
    - OPENROUTER_PROVIDER_JSON: Full JSON object, e.g. {"only": ["openai"]}
    - OPENROUTER_PROVIDER_ONLY: Comma-separated list of allowed providers
    - OPENROUTER_PROVIDER_IGNORE: Comma-separated list of excluded providers
    - OPENROUTER_PROVIDER_PREFER: Comma-separated list of preferred providers

    Returns:
        A dict suitable for OpenRouter's provider field, or None if unset.
    """
    provider_json = os.getenv('OPENROUTER_PROVIDER_JSON')
    if provider_json:
        try:
            parsed = json.loads(provider_json)
            if isinstance(parsed, dict) and parsed:
                return parsed
        except Exception:
            print('Warning: invalid OPENROUTER_PROVIDER_JSON; falling back to specific env vars')

    provider_config: Dict[str, Any] = {}

    allowed_providers = _parse_env_csv(os.getenv('OPENROUTER_PROVIDER_ONLY'))
    if allowed_providers:
        provider_config['only'] = allowed_providers

    ignored_providers = _parse_env_csv(os.getenv('OPENROUTER_PROVIDER_IGNORE'))
    if ignored_providers:
        provider_config['ignore'] = ignored_providers

    preferred_providers = _parse_env_csv(os.getenv('OPENROUTER_PROVIDER_PREFER'))
    if preferred_providers:
        provider_config['prefer'] = preferred_providers

    return provider_config or None


def retry_with_backoff(max_retries: int = 3, initial_wait: float = 1.0, backoff_factor: float = 2.0):
    """
    Decorator to automatically retry API calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_wait: Initial wait time in seconds
        backoff_factor: Multiplier for wait time after each retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            wait_time = initial_wait
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.Timeout, 
                        requests.exceptions.ConnectionError,
                        requests.exceptions.HTTPError,
                        ConnectionError) as e:
                    last_exception = e
                    
                    # Check if it's a retryable HTTP error (5xx or 429)
                    if isinstance(e, requests.exceptions.HTTPError):
                        status_code = e.response.status_code if hasattr(e, 'response') else None
                        if status_code and status_code < 500 and status_code != 429:
                            # Don't retry for 4xx errors (except 429 Too Many Requests)
                            raise
                    
                    if attempt < max_retries:
                        print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        wait_time *= backoff_factor
                    else:
                        print(f"All {max_retries + 1} attempts failed. Last error: {e}")
                except (anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError) as e:
                    # Handle Anthropic-specific errors
                    last_exception = e
                    
                    if attempt < max_retries:
                        print(f"Attempt {attempt + 1} failed (Anthropic): {e}. Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        wait_time *= backoff_factor
                    else:
                        print(f"All {max_retries + 1} attempts failed. Last error: {e}")
                except Exception as e:
                    # For other exceptions, don't retry
                    print(f"Non-retryable error: {e}")
                    raise
            
            # If we get here, all retries failed
            raise last_exception if last_exception else RuntimeError(f"Failed after {max_retries + 1} attempts")
        
        return wrapper
    return decorator


class LLMClient:
    """Flexible LLM API client supporting multiple providers with limit checking."""
    
    def __init__(self, provider: str = None, base_url: str = None, api_key: str = None, 
                 max_retries: int = 3, initial_wait: float = 1.0, backoff_factor: float = 2.0, 
                 enable_logging: bool = True, log_stage: str = 'general', **kwargs):
        """
        Initialize LLM client.

        Auto-loads from .env with provider-specific parameter names:
        - LLM_PROVIDER: Provider name (required - used to select which config to load)
        - {PROVIDER}_API_KEY: Provider-specific API key (fallback to API_KEY)
        - {PROVIDER}_LLM_MODEL: Provider-specific model name (fallback to LLM_MODEL)
        - {PROVIDER}_BASE_URL: Provider-specific base URL (fallback to BASE_URL)

        Supported providers:
        - 'ollama': Self-hosted Ollama server (OpenAI-compatible)
        - 'openai': OpenAI GPT models
        - 'anthropic': Anthropic Claude models
        - 'groq': Groq LLM inference (OpenAI-compatible)
        - 'google': Google Gemini (OpenAI-compatible)
        - 'github': GitHub Models (OpenAI-compatible)
        - 'cerebras': Cerebras Models (OpenAI-compatible)
        - 'openrouter': OpenRouter (OpenAI-compatible)
        - 'fireworks': Fireworks AI (OpenAI-compatible)
        - 'custom': Any OpenAI-compatible API endpoint

        Example .env configuration:
            LLM_PROVIDER=groq
            GROQ_API_KEY=<your_groq_api_key>
            GROQ_LLM_MODEL=llama-3.3-70b-versatile

            # For GitHub Models
            # LLM_PROVIDER=github
            # GITHUB_TOKEN=<your_github_token>
            # GITHUB_LLM_MODEL=openai/gpt-4o

        Args:
            provider: Provider name (auto-loads from LLM_PROVIDER env var)
            base_url: Base URL for the API (override provider-specific config)
            api_key: API key for authentication (override provider-specific config)
            max_retries: Maximum number of retry attempts for API calls (default: 3)
            initial_wait: Initial wait time in seconds before first retry (default: 1.0)
            backoff_factor: Multiplier for wait time after each retry (default: 2.0)
            enable_logging: Whether to enable LLM usage logging (default: True)
            log_stage: Stage name for logging (e.g., 'representation', 'retrieval', 'reasoning')
            **kwargs: Additional provider-specific parameters
        """
        # Auto-load provider from environment if not provided
        if provider is None:
            provider = os.getenv("LLM_PROVIDER") or "ollama"
        
        self.provider = provider.lower()
        
        # Auto-load provider-specific configuration
        loaded_base_url, loaded_api_key, loaded_model = load_provider_config(self.provider)
        
        # Use provided parameters (if any) as overrides
        self.api_key = api_key or loaded_api_key
        self.base_url = base_url or loaded_base_url
        self.model = loaded_model
        self.openrouter_provider = load_openrouter_provider_config() if self.provider == 'openrouter' else None
        
        self.max_retries = max_retries
        self.initial_wait = initial_wait
        self.backoff_factor = backoff_factor
        
        # Initialize logger
        self.enable_logging = enable_logging
        self.logger = LLMLogger(stage=log_stage) if enable_logging else None

        # Initialize clients
        if self.provider == 'ollama':
            self.openai_client = OpenAI(
                api_key=self.api_key,
                base_url=f"{self.base_url}/v1/"
            )
            self.ollama_url = self.base_url

        elif self.provider == 'openai':
            self.openai_client = OpenAI(api_key=self.api_key)

        elif self.provider == 'anthropic':
            self.anthropic_client = anthropic.Anthropic(api_key=self.api_key)

        elif self.provider == 'groq':
            # Groq uses OpenAI-compatible API
            self.openai_client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1"
            )

        elif self.provider == 'google':
            # Google Gemini uses OpenAI-compatible API
            self.openai_client = OpenAI(
                api_key=self.api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )

        elif self.provider == 'github':
            # GitHub Models uses OpenAI-compatible API
            self.openai_client = OpenAI(
                api_key=self.api_key,
                base_url="https://models.github.ai/inference"
            )

        elif self.provider == 'cerebras':
            # Cerebras Models uses OpenAI-compatible API
            self.openai_client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.cerebras.ai/v1"
            )

        elif self.provider == 'openrouter':
            # OpenRouter uses OpenAI-compatible API
            self.openai_client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )

        elif self.provider == 'fireworks':
            # Fireworks uses OpenAI-compatible API
            self.openai_client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.fireworks.ai/inference/v1"
            )

        elif self.provider == 'custom':
            # For other OpenAI-compatible APIs
            self.openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def list_models(self) -> List[str]:
        """List available models."""
        if self.provider == 'ollama':
            return self._ollama_list_models()
        elif self.provider in ['openai', 'custom', 'groq', 'google', 'github', 'cerebras', 'openrouter', 'fireworks']:
            return self._openai_list_models()
        elif self.provider == 'anthropic':
            return self._anthropic_list_models()
        else:
            return []

    def load_model(self):
        """Stub method for compatibility with reasoners that call load_model()."""
        # For API-based clients, there is no local model to load.
        # This method exists to maintain a consistent interface.
        return True

    def unload_model(self):
        """Stub method for compatibility with reasoners that call unload_model()."""
        # For API-based clients, nothing to unload.
        return True

    def get_running_models(self) -> List[str]:
        """Get currently running models (Ollama specific)."""
        if self.provider == 'ollama':
            return self._ollama_get_running_models()
        return []

    def check_limits(self) -> Dict[str, Any]:
        """
        Check current rate limit and token usage status for the provider.
        
        Returns:
            Dict containing limit information:
            - rate_limit_remaining: Remaining requests for current window
            - rate_limit_reset: Time until rate limit resets (seconds)
            - token_limit_remaining: Remaining tokens for current window (if available)
            - token_limit_reset: Time until token limit resets (seconds, if available)
            - provider: Provider name
            - status: 'ok', 'warning', 'critical' based on remaining limits
        """
        if self.provider == 'groq':
            return self._check_groq_limits()
        elif self.provider in ['openai', 'google', 'github', 'custom', 'cerebras', 'openrouter', 'fireworks']:
            return self._check_openai_limits()
        elif self.provider == 'anthropic':
            return self._check_anthropic_limits()
        elif self.provider == 'ollama':
            return self._check_ollama_limits()
        else:
            return {
                'provider': self.provider,
                'status': 'unknown',
                'error': f'Limit checking not supported for provider: {self.provider}'
            }

    def chat_completion(self, model: str, messages: List[Dict], stage: Optional[str] = None, return_full_response: bool = False, openrouter_provider: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Generate chat completion.

        Args:
            model: Model name
            messages: List of message dicts with 'role' and 'content'
            stage: Optional stage name to override default logging stage
            return_full_response: If True, return dict with text, usage, and metadata.
                                 If False (default), return only text string for backward compatibility.
            openrouter_provider: OpenRouter provider routing options, e.g. {'only': ['azure']}
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            If return_full_response=False: Generated response text (str)
            If return_full_response=True: Dict with 'text', 'usage', 'model', 'provider' keys
        """
        if self.provider in ['ollama', 'openai', 'custom', 'groq', 'google', 'github', 'cerebras', 'openrouter', 'fireworks']:
            return self._openai_chat_completion(
                model,
                messages,
                stage=stage,
                return_full_response=return_full_response,
                openrouter_provider=openrouter_provider if openrouter_provider is not None else self.openrouter_provider,
                **kwargs,
            )
        elif self.provider == 'anthropic':
            return self._anthropic_chat_completion(model, messages, stage=stage, return_full_response=return_full_response, **kwargs)
        else:
            raise ValueError(f"Chat completion not supported for provider: {self.provider}")

    def get_display_model_name(self, model: str) -> str:
        """
        Get display-friendly model name based on provider-specific rules.

        Args:
            model: Raw model name from API

        Returns:
            Display-friendly model name
        """
        if self.provider == 'fireworks':
            # Fireworks format: accounts/fireworks/models/{model_name}
            # Extract the model name after the last slash
            if '/' in model:
                return model.split('/')[-1]
            return model
        elif self.provider == 'github':
            # GitHub format: openai/gpt-4o -> gpt-4o
            if '/' in model:
                return model.split('/')[-1]
            return model
        elif self.provider == 'groq':
            # Groq format: openai/gpt-oss-20b -> gpt-oss-20b
            if '/' in model:
                return model.split('/')[-1]
            return model
        elif self.provider == 'openrouter':
            # OpenRouter format: openai/gpt-oss-20b:free -> gpt-oss-20b
            if '/' in model:
                base_name = model.split('/')[-1]
                # Remove suffix after colon if present
                if ':' in base_name:
                    base_name = base_name.split(':')[0]
                return base_name
            return model
        elif self.provider == 'ollama':
            # Ollama format: gpt-oss:20b -> gpt-oss-20b
            if ':' in model:
                parts = model.split(':')
                if len(parts) >= 2:
                    # Replace colon with dash and combine
                    return f"{parts[0]}-{parts[1]}"
            return model
        else:
            # For other providers, return as-is
            return model

    def completion(self, model: str, prompt: str, **kwargs) -> str:
        """
        Generate single completion (non-chat).

        Args:
            model: Model name
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            Generated response text
        """
        if self.provider in ['ollama', 'openai', 'custom', 'groq', 'google', 'github', 'cerebras', 'openrouter', 'fireworks']:
            return self._openai_completion(model, prompt, **kwargs)
        else:
            # Convert to chat format for other providers
            messages = [{"role": "user", "content": prompt}]
            return self.chat_completion(model, messages, **kwargs)

    def _ollama_list_models(self) -> List[str]:
        """List Ollama models."""
        try:
            headers = {"X-API-Key": self.api_key} if self.api_key else {}
            response = requests.get(f"{self.ollama_url}/api/tags", headers=headers)
            response.raise_for_status()
            return [model['name'] for model in response.json()['models']]
        except Exception as e:
            print(f"Error listing Ollama models: {e}")
            return []

    def _ollama_get_running_models(self) -> List[str]:
        """Get running Ollama models."""
        try:
            headers = {"X-API-Key": self.api_key} if self.api_key else {}
            response = requests.get(f"{self.ollama_url}/api/ps", headers=headers)
            response.raise_for_status()
            return [model['name'] for model in response.json()['models']]
        except Exception as e:
            print(f"Error getting running models: {e}")
            return []

    def _openai_list_models(self) -> List[str]:
        """List OpenAI models."""
        try:
            models = self.openai_client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Error listing OpenAI models: {e}")
            return []

    def _anthropic_list_models(self) -> List[str]:
        """List Anthropic models."""
        # Anthropic doesn't have a list models API, return known models
        return ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']

    def _openai_chat_completion(self, model: str, messages: List[Dict], stage: Optional[str] = None, return_full_response: bool = False, openrouter_provider: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """OpenAI-style chat completion with retry support and logging."""
        start_time = time.time()
        
        # Explicitly disable streaming to prevent JSON parsing errors
        if 'stream' not in kwargs:
            kwargs['stream'] = False
        
        # Use requests directly for more reliable JSON parsing (bypasses OpenAI client library issues)
        use_requests_fallback = True  # Enabled by default for better compatibility
        
        @retry_with_backoff(
            max_retries=self.max_retries,
            initial_wait=self.initial_wait,
            backoff_factor=self.backoff_factor
        )
        def _call_api():
            if use_requests_fallback:
                
                import urllib.parse as urlparse
            
                base = self.base_url if self.base_url else "https://api.openai.com/v1"

      
                parsed_url = urlparse.urlparse(base)
                query_params = dict(urlparse.parse_qsl(parsed_url.query))

                
                path = parsed_url.path.rstrip('/')
                if '/v1' not in path:
                    path = f"{path}/v1"
                if '/chat/completions' not in path:
                    path = f"{path}/chat/completions"

                target_url = f"{parsed_url.scheme}://{parsed_url.netloc}{path}".replace('//', '/')
                target_url = target_url.replace(':/', '://') 

                headers = {
                    "Authorization": f"Bearer {self.api_key}" if self.api_key else "Bearer ollama",
                    "Content-Type": "application/json",
                }

                payload = {
                    "model": model,
                    "messages": messages,
                    **kwargs
                }
                if self.provider == 'openrouter' and openrouter_provider is not None:
                    payload['provider'] = openrouter_provider
                
                response = requests.post(target_url, headers=headers, json=payload, params=query_params, timeout=300)
                response.raise_for_status()
                data = response.json()

                # 3. 模擬 OpenAI 物件結構 (對應你 curl 的 JSON)
                class Message:
                    def __init__(self, msg_dict):
                        self.content = msg_dict.get('content', '')
                        self.role = msg_dict.get('role', 'assistant')
                        self.tool_calls = msg_dict.get('tool_calls', [])
                        # 如果你想保留 qwen3 的思考過程，可以加這一行
                        self.reasoning = msg_dict.get('reasoning', '')

                class Choice:
                    def __init__(self, choice_dict):
                        self.message = Message(choice_dict.get('message', {}))
                        self.finish_reason = choice_dict.get('finish_reason')

                class Usage:
                    def __init__(self, usage_dict):
                        self.prompt_tokens = usage_dict.get('prompt_tokens', 0)
                        self.completion_tokens = usage_dict.get('completion_tokens', 0)

                class ResponseWrapper:
                    def __init__(self, data_dict):
                        self.choices = [Choice(c) for c in data_dict.get('choices', [])]
                        self.usage = Usage(data_dict.get('usage', {}))
                
                return ResponseWrapper(data)
            else:                
                # Use OpenAI client (fallback behavior)
                completion = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                return completion
        
        try:
            completion = _call_api()
            first_choice = completion.choices[0]
            message = first_choice.message
            response_text = message.content or ""
            finish_reason = getattr(first_choice, 'finish_reason', None)

            def _normalize_tool_calls(raw_tool_calls):
                normalized = []
                if not raw_tool_calls:
                    return normalized
                for call in raw_tool_calls:
                    if isinstance(call, dict):
                        normalized.append({
                            'id': call.get('id', ''),
                            'type': call.get('type', 'function'),
                            'function': {
                                'name': (call.get('function', {}) or {}).get('name', ''),
                                'arguments': (call.get('function', {}) or {}).get('arguments', '{}')
                            }
                        })
                    else:
                        function_obj = getattr(call, 'function', None)
                        normalized.append({
                            'id': getattr(call, 'id', ''),
                            'type': getattr(call, 'type', 'function'),
                            'function': {
                                'name': getattr(function_obj, 'name', ''),
                                'arguments': getattr(function_obj, 'arguments', '{}')
                            }
                        })
                return normalized

            tool_calls = _normalize_tool_calls(getattr(message, 'tool_calls', None))
            execution_time = time.time() - start_time
            
            # Extract token usage
            request_tokens = completion.usage.prompt_tokens
            response_tokens = completion.usage.completion_tokens
            
            # Log usage if logging is enabled
            if self.enable_logging and self.logger:
                log_stage = stage or self.logger.stage
                logger = LLMLogger(stage=log_stage) if stage else self.logger
                
                logger.log_usage(
                    model=model,
                    request_tokens=request_tokens,
                    response_tokens=response_tokens,
                    execution_time=execution_time,
                    provider=self.provider
                )
            
            # Return full response if requested, otherwise just text for backward compatibility
            if return_full_response:
                return {
                    'text': response_text,
                    'tool_calls': tool_calls,
                    'finish_reason': finish_reason,
                    'usage': {
                        'prompt_tokens': request_tokens,
                        'completion_tokens': response_tokens,
                        'total_tokens': request_tokens + response_tokens
                    },
                    'model': model,
                    'provider': self.provider,
                    'execution_time': execution_time
                }
            else:
                return response_text
        except Exception as e:
            error_msg = str(e)
            print(f"Error in chat completion after retries: {error_msg}")
            
            if return_full_response:
                return {
                    'text': '',
                    'tool_calls': [],
                    'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                    'model': model,
                    'provider': self.provider,
                    'execution_time': time.time() - start_time,
                    'error': error_msg
                }
            else:
                return ""

    def _openai_completion(self, model: str, prompt: str, **kwargs) -> str:
        """OpenAI-style completion with retry support."""
        @retry_with_backoff(
            max_retries=self.max_retries,
            initial_wait=self.initial_wait,
            backoff_factor=self.backoff_factor
        )
        def _call_api():
            completion = self.openai_client.completions.create(
                model=model,
                prompt=prompt,
                **kwargs
            )
            return completion.choices[0].text
        
        try:
            return _call_api()
        except Exception as e:
            print(f"Error in completion after retries: {e}")
            return ""

    def _anthropic_chat_completion(self, model: str, messages: List[Dict], stage: Optional[str] = None, return_full_response: bool = False, **kwargs) -> str:
        """Anthropic chat completion with retry support and logging."""
        start_time = time.time()
        
        @retry_with_backoff(
            max_retries=self.max_retries,
            initial_wait=self.initial_wait,
            backoff_factor=self.backoff_factor
        )
        def _call_api():
            # Convert messages to Anthropic format
            system_message = ""
            conversation_messages = []

            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    conversation_messages.append(msg)

            # Ensure max_tokens is always a valid int and not None
            max_tokens = kwargs.get('max_tokens', 1000)
            if max_tokens is None:
                max_tokens = 1000
            try:
                max_tokens = int(max_tokens)
            except Exception:
                max_tokens = 1000
            message = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=kwargs.get('temperature', 0.7),
                system=system_message,
                messages=conversation_messages
            )
            return message
        
        try:
            message = _call_api()
            response_text = message.content[0].text
            execution_time = time.time() - start_time
            
            # Extract token usage
            request_tokens = message.usage.input_tokens
            response_tokens = message.usage.output_tokens
            
            # Log usage if logging is enabled
            if self.enable_logging and self.logger:
                log_stage = stage or self.logger.stage
                logger = LLMLogger(stage=log_stage) if stage else self.logger
                
                logger.log_usage(
                    model=model,
                    request_tokens=request_tokens,
                    response_tokens=response_tokens,
                    execution_time=execution_time,
                    provider=self.provider
                )
            
            # Return full response if requested, otherwise just text for backward compatibility
            if return_full_response:
                return {
                    'text': response_text,
                    'usage': {
                        'prompt_tokens': request_tokens,
                        'completion_tokens': response_tokens,
                        'total_tokens': request_tokens + response_tokens
                    },
                    'model': model,
                    'provider': self.provider,
                    'execution_time': execution_time
                }
            else:
                return response_text
        except Exception as e:
            print(f"Error in Anthropic chat completion after retries: {e}")
            if return_full_response:
                return {
                    'text': '',
                    'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                    'model': model,
                    'provider': self.provider,
                    'execution_time': time.time() - start_time,
                    'error': str(e)
                }
            else:
                return ""

    # Traditional requests methods for Ollama
    def ollama_generate(self, model: str, prompt: str, **kwargs) -> str:
        """Ollama generate using requests with retry support."""
        if self.provider != 'ollama':
            raise ValueError("This method is only for Ollama provider")

        @retry_with_backoff(
            max_retries=self.max_retries,
            initial_wait=self.initial_wait,
            backoff_factor=self.backoff_factor
        )
        def _call_api():
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                **kwargs
            }
            headers = {"X-API-Key": self.api_key} if self.api_key else {}
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["response"]
        
        try:
            return _call_api()
        except Exception as e:
            print(f"Error in Ollama generate after retries: {e}")
            return ""

    def ollama_chat(self, model: str, messages: List[Dict], **kwargs) -> str:
        """Ollama chat using requests with retry support."""
        if self.provider != 'ollama':
            raise ValueError("This method is only for Ollama provider")

        @retry_with_backoff(
            max_retries=self.max_retries,
            initial_wait=self.initial_wait,
            backoff_factor=self.backoff_factor
        )
        def _call_api():
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                **kwargs
            }
            headers = {"X-API-Key": self.api_key} if self.api_key else {}
            response = requests.post(f"{self.ollama_url}/api/chat", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["message"]["content"]
        
        try:
            return _call_api()
        except Exception as e:
            print(f"Error in Ollama chat after retries: {e}")
            return ""

    def _check_openai_limits(self) -> Dict[str, Any]:
        """Check rate limits for OpenAI-compatible APIs."""
        try:
            # Make a minimal request to get rate limit headers
            # Use a simple completion request that should be cheap
            test_messages = [{"role": "user", "content": "test"}]
            
            # Temporarily reduce retries for this check
            original_max_retries = self.max_retries
            self.max_retries = 0
            
            try:
                completion = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=test_messages,
                    max_tokens=1,  # Minimal response
                    temperature=0
                )
                
                # Extract rate limit information from response headers
                # Note: OpenAI doesn't always include these headers, depends on the provider
                response = completion._response if hasattr(completion, '_response') else None
                
                limits = {
                    'provider': self.provider,
                    'status': 'ok',
                    'rate_limit_remaining': None,
                    'rate_limit_reset': None,
                    'token_limit_remaining': None,
                    'token_limit_reset': None
                }
                
                if response and hasattr(response, 'headers'):
                    headers = response.headers
                    
                    # OpenAI-style rate limit headers
                    limits['rate_limit_remaining'] = self._extract_header_value(headers, [
                        'x-ratelimit-remaining-requests',
                        'x-ratelimit-remaining-tokens'
                    ])
                    
                    limits['rate_limit_reset'] = self._extract_header_value(headers, [
                        'x-ratelimit-reset-requests',
                        'x-ratelimit-reset-tokens'
                    ])
                    
                    # Some providers include token limits
                    limits['token_limit_remaining'] = self._extract_header_value(headers, [
                        'x-ratelimit-remaining-tokens'
                    ])
                    
                    limits['token_limit_reset'] = self._extract_header_value(headers, [
                        'x-ratelimit-reset-tokens'
                    ])
                
                # Determine status based on remaining limits
                if limits['rate_limit_remaining'] is not None:
                    remaining_pct = limits['rate_limit_remaining'] / 100.0  # Assume 100 is a reasonable baseline
                    if remaining_pct < 0.1:
                        limits['status'] = 'critical'
                    elif remaining_pct < 0.3:
                        limits['status'] = 'warning'
                
                return limits
                
            finally:
                self.max_retries = original_max_retries
                
        except Exception as e:
            return {
                'provider': self.provider,
                'status': 'error',
                'error': f'Failed to check limits: {str(e)}'
            }

    def _check_groq_limits(self) -> Dict[str, Any]:
        """Check rate limits for Groq API."""
        try:
            # Groq provides rate limit information via their API
            # Try to get rate limits from a test request
            test_messages = [{"role": "user", "content": "test"}]
            
            # Temporarily reduce retries for this check
            original_max_retries = self.max_retries
            self.max_retries = 0
            
            try:
                completion = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=test_messages,
                    max_tokens=1,
                    temperature=0
                )
                
                # Extract rate limit information from response headers
                response = completion._response if hasattr(completion, '_response') else None
                
                limits = {
                    'provider': 'groq',
                    'status': 'ok',
                    'rate_limit_remaining': None,
                    'rate_limit_reset': None,
                    'token_limit_remaining': None,
                    'token_limit_reset': None
                }
                
                if response and hasattr(response, 'headers'):
                    headers = response.headers
                    
                    # Groq uses different header names than OpenAI
                    # Try various header formats
                    limits['rate_limit_remaining'] = self._extract_header_value(headers, [
                        'x-ratelimit-remaining-requests',
                        'x-ratelimit-remaining',
                        'x-rate-limit-remaining-requests'
                    ])
                    
                    limits['rate_limit_reset'] = self._extract_header_value(headers, [
                        'x-ratelimit-reset-requests',
                        'x-ratelimit-reset',
                        'x-rate-limit-reset-requests'
                    ])
                    
                    limits['token_limit_remaining'] = self._extract_header_value(headers, [
                        'x-ratelimit-remaining-tokens',
                        'x-rate-limit-remaining-tokens'
                    ])
                    
                    limits['token_limit_reset'] = self._extract_header_value(headers, [
                        'x-ratelimit-reset-tokens',
                        'x-rate-limit-reset-tokens'
                    ])
                
                # If no headers found, provide default Groq limits info
                # Groq typically has generous limits, but let's be conservative
                if limits['rate_limit_remaining'] is None:
                    limits['rate_limit_remaining'] = 'unlimited'  # Groq has very high limits
                    limits['rate_limit_reset'] = 'N/A'
                    limits['token_limit_remaining'] = 'unlimited'
                    limits['token_limit_reset'] = 'N/A'
                
                return limits
                
            finally:
                self.max_retries = original_max_retries
                
        except Exception as e:
            # If rate limit check fails, return basic info
            return {
                'provider': 'groq',
                'status': 'ok',  # Assume OK if we can't check
                'rate_limit_remaining': 'unknown',
                'rate_limit_reset': 'unknown',
                'token_limit_remaining': 'unknown',
                'token_limit_reset': 'unknown',
                'error': f'Could not check limits: {str(e)}'
            }

    def _check_anthropic_limits(self) -> Dict[str, Any]:
        """Check rate limits for Anthropic API."""
        try:
            # Anthropic provides rate limit info in API responses
            # We'll make a minimal request to get current status
            test_messages = [{"role": "user", "content": "test"}]
            
            original_max_retries = self.max_retries
            self.max_retries = 0
            
            try:
                message = self.anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=1,
                    temperature=0,
                    messages=test_messages
                )
                
                limits = {
                    'provider': self.provider,
                    'status': 'ok',
                    'rate_limit_remaining': None,
                    'rate_limit_reset': None,
                    'token_limit_remaining': None,
                    'token_limit_reset': None
                }
                
                # Anthropic includes rate limit info in the response
                if hasattr(message, 'usage'):
                    # Note: Anthropic's rate limit info might be in different attributes
                    # This is a placeholder - actual implementation depends on Anthropic's API
                    pass
                
                return limits
                
            finally:
                self.max_retries = original_max_retries
                
        except Exception as e:
            return {
                'provider': self.provider,
                'status': 'error',
                'error': f'Failed to check limits: {str(e)}'
            }

    def _check_ollama_limits(self) -> Dict[str, Any]:
        """Check limits for Ollama (typically no limits)."""
        return {
            'provider': self.provider,
            'status': 'ok',
            'rate_limit_remaining': 'unlimited',
            'rate_limit_reset': None,
            'token_limit_remaining': 'unlimited',
            'token_limit_reset': None,
            'note': 'Ollama typically has no API rate limits'
        }

    def _extract_header_value(self, headers: Dict, header_names: List[str]) -> Optional[float]:
        """Extract numeric value from response headers."""
        for header_name in header_names:
            if header_name in headers:
                value = headers[header_name]
                try:
                    # Try to parse as float
                    return float(value)
                except (ValueError, TypeError):
                    # Some headers might be strings like "60s"
                    if isinstance(value, str) and value.endswith('s'):
                        try:
                            return float(value[:-1])  # Remove 's' suffix
                        except (ValueError, TypeError):
                            pass
        return None

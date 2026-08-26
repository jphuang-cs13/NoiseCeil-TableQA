"""
LLM Usage Logger

Logs API usage statistics including:
- Request tokens
- Response tokens
- Execution time
- Model used
- Stage (representation, retrieval, reasoning, etc.)

Usage:
    from llm.llm_logger import LLMLogger

    logger = LLMLogger(stage='representation')
    logger.log_usage(
        model='gpt-4',
        request_tokens=150,
        response_tokens=250,
        execution_time=2.5
    )
"""

import json
import os
import fcntl
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import csv


class LLMLogger:
    """Logger for LLM API usage statistics."""

    def __init__(self, stage: str = 'general', project_root: Optional[str] = None):
        """
        Initialize LLM logger.

        Args:
            stage: Stage name (representation, retrieval, reasoning, reasoning/zero-shot, etc.)
                   Can include path separators for nested stages (e.g., reasoning/zero-shot)
            project_root: Project root directory (defaults to parent of llm module)
        """
        self.stage = stage
        
        # Determine project root
        if project_root is None:
            # Get parent of the llm directory
            llm_dir = Path(__file__).parent
            project_root = llm_dir.parent
        
        self.project_root = Path(project_root)
        self.logs_dir = self.project_root / 'llm_logs'
        self.stage_dir = self.logs_dir / stage
        
        # Create directories if they don't exist
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the last part of the stage path for file naming
        # E.g., "reasoning/zero-shot" -> "zero-shot"
        file_prefix = stage.split('/')[-1] if '/' in stage else stage
        
        # File paths
        self.json_log_file = self.stage_dir / f"{file_prefix}_usage.jsonl"
        self.csv_log_file = self.stage_dir / f"{file_prefix}_usage.csv"
        self.summary_file = self.stage_dir / f"{file_prefix}_summary.json"
        
        # Initialize CSV header if file doesn't exist
        self._init_csv_header()

    def _init_csv_header(self):
        """Initialize CSV file with header if it doesn't exist."""
        if not self.csv_log_file.exists():
            with open(self.csv_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'model',
                    'provider',
                    'request_tokens',
                    'response_tokens',
                    'total_tokens',
                    'execution_time_sec',
                    'cost_estimate',
                    'notes'
                ])

    def log_usage(
        self,
        model: str,
        request_tokens: int,
        response_tokens: int,
        execution_time: float,
        provider: str = 'unknown',
        notes: str = '',
        cost_estimate: Optional[float] = None
    ) -> None:
        """
        Log LLM API usage.

        Args:
            model: Model name used
            request_tokens: Number of input tokens
            response_tokens: Number of output tokens
            execution_time: Execution time in seconds
            provider: Provider name (ollama, openai, anthropic, etc.)
            notes: Optional notes about the request
            cost_estimate: Optional estimated cost in USD
        """
        timestamp = datetime.now().isoformat()
        total_tokens = request_tokens + response_tokens

        # Log to JSONL
        log_entry = {
            'timestamp': timestamp,
            'model': model,
            'provider': provider,
            'request_tokens': request_tokens,
            'response_tokens': response_tokens,
            'total_tokens': total_tokens,
            'execution_time_sec': execution_time,
            'cost_estimate': cost_estimate,
            'notes': notes
        }

        with open(self.json_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Log to CSV
        with open(self.csv_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                model,
                provider,
                request_tokens,
                response_tokens,
                total_tokens,
                f"{execution_time:.4f}",
                cost_estimate if cost_estimate else '',
                notes
            ])

        # Update summary (pass request_tokens and response_tokens)
        self._update_summary(total_tokens, execution_time, cost_estimate, request_tokens, response_tokens)

    def _update_summary(self, total_tokens: int, execution_time: float, cost: Optional[float] = None, request_tokens: int = 0, response_tokens: int = 0) -> None:
        """Update aggregated summary statistics with file locking to prevent corruption."""
        max_retries = 5
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                # Open file with exclusive lock
                with open(self.summary_file, 'a+') as f:
                    # Acquire exclusive lock (will wait if another process has lock)
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    
                    try:
                        # Seek to beginning to read current content
                        f.seek(0)
                        content = f.read()
                        
                        if content.strip():
                            # Load existing summary
                            summary = json.loads(content)
                        else:
                            # Empty file, start fresh
                            summary = {}
                        
                        # Update summary
                        summary['total_requests'] = summary.get('total_requests', 0) + 1
                        summary['total_tokens'] = summary.get('total_tokens', 0) + total_tokens
                        summary['total_request_tokens'] = summary.get('total_request_tokens', 0) + request_tokens
                        summary['total_response_tokens'] = summary.get('total_response_tokens', 0) + response_tokens
                        summary['total_execution_time_sec'] = summary.get('total_execution_time_sec', 0) + execution_time

                        if cost is not None:
                            summary['total_cost_estimate'] = summary.get('total_cost_estimate', 0) + cost

                        summary['last_updated'] = datetime.now().isoformat()
                        summary['average_execution_time_sec'] = summary['total_execution_time_sec'] / summary['total_requests']
                        summary['average_tokens_per_request'] = summary['total_tokens'] / summary['total_requests']

                        # Write back (truncate first, then write)
                        f.seek(0)
                        f.truncate()
                        json.dump(summary, f, indent=2)
                        f.write('\n')  # Add trailing newline
                        f.flush()
                        os.fsync(f.fileno())  # Ensure written to disk
                        
                    finally:
                        # Release lock
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                # Success, exit retry loop
                break
                
            except (IOError, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    # Wait and retry
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    # Final attempt failed, log error
                    print(f"Warning: Failed to update summary after {max_retries} attempts: {e}")
                    # Try to at least fix corrupted file
                    try:
                        with open(self.summary_file, 'w') as f:
                            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                            json.dump({
                                'total_requests': 1,
                                'total_tokens': total_tokens,
                                'total_request_tokens': request_tokens,
                                'total_response_tokens': response_tokens,
                                'total_execution_time_sec': execution_time,
                                'last_updated': datetime.now().isoformat(),
                                'average_execution_time_sec': execution_time,
                                'average_tokens_per_request': float(total_tokens)
                            }, f, indent=2)
                            f.write('\n')
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except Exception as e2:
                        print(f"Error: Could not recover corrupted summary file: {e2}")

    def _load_summary(self) -> Dict[str, Any]:
        """Load existing summary or return empty dict."""
        if self.summary_file.exists():
            try:
                with open(self.summary_file, 'r') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # Shared lock for reading
                    try:
                        content = f.read().strip()
                        if content:
                            return json.loads(content)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (IOError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load summary file: {e}")
                return {}
        return {}

    def get_summary(self) -> Dict[str, Any]:
        """Get current summary statistics."""
        return self._load_summary()

    def get_all_logs(self) -> list[Dict[str, Any]]:
        """Get all logs for this stage as list of dicts."""
        logs = []
        if self.json_log_file.exists():
            with open(self.json_log_file, 'r') as f:
                for line in f:
                    logs.append(json.loads(line))
        return logs


class LLMLoggerManager:
    """Manager for multiple stage loggers."""

    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize logger manager.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root
        self.loggers: Dict[str, LLMLogger] = {}

    def get_logger(self, stage: str) -> LLMLogger:
        """
        Get or create logger for a stage.

        Args:
            stage: Stage name

        Returns:
            LLMLogger instance for the stage
        """
        if stage not in self.loggers:
            self.loggers[stage] = LLMLogger(stage=stage, project_root=self.project_root)
        return self.loggers[stage]

    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries for all stages that have logs."""
        summaries = {}
        logs_dir = Path(self.project_root or Path(__file__).parent.parent) / 'llm_logs'

        if logs_dir.exists():
            for stage_dir in logs_dir.iterdir():
                if stage_dir.is_dir():
                    summary_file = stage_dir / f"{stage_dir.name}_summary.json"
                    if summary_file.exists():
                        with open(summary_file, 'r') as f:
                            summaries[stage_dir.name] = json.load(f)

        return summaries

    def print_all_summaries(self) -> None:
        """Print summaries for all stages."""
        summaries = self.get_all_summaries()

        if not summaries:
            print("No LLM usage logs found.")
            return

        print("\n" + "=" * 80)
        print("LLM USAGE SUMMARY BY STAGE")
        print("=" * 80)

        total_requests = 0
        total_tokens = 0
        total_cost = 0.0

        for stage, summary in summaries.items():
            print(f"\n[{stage.upper()}]")
            print(f"  Requests: {summary.get('total_requests', 0)}")
            print(f"  Total Tokens: {summary.get('total_tokens', 0)}")
            print(f"  Total Execution Time: {summary.get('total_execution_time_sec', 0):.2f}s")
            print(f"  Avg Time per Request: {summary.get('average_execution_time_sec', 0):.4f}s")
            print(f"  Avg Tokens per Request: {summary.get('average_tokens_per_request', 0):.0f}")

            if summary.get('total_cost_estimate'):
                print(f"  Total Cost: ${summary.get('total_cost_estimate', 0):.4f}")

            total_requests += summary.get('total_requests', 0)
            total_tokens += summary.get('total_tokens', 0)
            total_cost += summary.get('total_cost_estimate', 0)

        print("\n" + "-" * 80)
        print(f"OVERALL TOTALS")
        print(f"  Total Requests: {total_requests}")
        print(f"  Total Tokens: {total_tokens}")
        if total_cost > 0:
            print(f"  Total Cost: ${total_cost:.4f}")
        print("=" * 80 + "\n")

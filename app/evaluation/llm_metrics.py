"""
LLM-based evaluation metrics.

This module implements evaluation metrics that leverage LLMs to make semantic judgments,
including LLM-based Exact Match (EM) that judges semantic equivalence between answers.
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from llm.llm_client import LLMClient

# Load environment variables
load_dotenv()

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

def _load_prompt(fname: str) -> Optional[str]:
    try:
        with open(_PROMPTS_DIR / fname, "r", encoding="utf-8") as _f:
            return _f.read()
    except Exception:
        return None


class LLMBasedMetrics:
    """
    Collection of evaluation metrics powered by LLMs.
    
    These metrics use an LLM to make more nuanced semantic judgments compared to
    simple string matching.
    """

    def __init__(self, 
                 provider: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: float = 0.0,
                 max_retries: int = 3):
        """
        Initialize LLM-based metrics evaluator.

        Args:
            provider: LLM provider (e.g., 'groq', 'openai', 'cerebras'). 
                     Auto-loads from LLM_PROVIDER env var if not provided.
            api_key: API key for the provider. Auto-loads from .env if not provided.
            base_url: Base URL for the provider. Auto-loads from .env if not provided.
            model: Model name to use. Auto-loads from LLM_MODEL env var if not provided.
            temperature: Temperature for LLM calls (default: 0.0 for deterministic output).
            max_retries: Maximum number of retries for API calls.
        """
        self.client = LLMClient(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            enable_logging=False  # Disable logging for metric calculations
        )
        self.model = model or self.client.model
        self.temperature = temperature
        self.max_retries = max_retries

    def llm_exact_match(self, gold_answer: str, user_answer: str, question: str = "", debug: bool = False) -> int:
        """
        Determine if two answers are semantically equivalent using an LLM.

        The LLM is instructed to judge whether the user answer and gold answer
        convey the same meaning, even if the wording differs.

        Args:
            gold_answer: The reference/correct answer.
            user_answer: The model-generated or user-provided answer.
            debug: If True, print debug information.

        Returns:
            1 if the answers are semantically equivalent, 0 otherwise.

        Raises:
            ValueError: If either answer is None or empty after stripping.
        """
        # Validate inputs
        gold_answer = str(gold_answer).strip() if gold_answer else ""
        user_answer = str(user_answer).strip() if user_answer else ""

        if not gold_answer and not user_answer:
            return 1

        if not gold_answer or not user_answer:
            return 0

        # Fast path: literal containment should count as equivalent.
        if gold_answer.lower() in user_answer.lower():
            return 1

        # Fast path: lexical overlap (bag-of-words) with light normalization.
        def _normalize(text: str) -> set:
            tokens = re.findall(r"[\w']+", text.lower())
            normalized = []
            for tok in tokens:
                if tok.endswith("'s") and len(tok) > 2:
                    tok = tok[:-2]
                if tok.startswith("'") and tok.endswith("'") and len(tok) > 2:
                    tok = tok[1:-1]
                normalized.append(tok)
            stopwords = {"the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "is", "are", "was", "were"}
            return {t for t in normalized if t and t not in stopwords}

        gold_tokens = _normalize(gold_answer)
        user_tokens = _normalize(user_answer)
        if gold_tokens and user_tokens:
            overlap = gold_tokens & user_tokens
            coverage = len(overlap) / len(gold_tokens)
            if coverage >= 0.6:
                return 1

        # Prepare the LLM prompt
        system_prompt = _load_prompt("llm_exact_match_system.txt") or (
            "You are an expert evaluator. Decide if the user answer correctly answers the question and aligns with the gold answer. Treat paraphrases, extra context, different order, and formatting (markdown/HTML/punctuation/case) as equivalent. If the gold answer text is clearly present or entailed in the user answer, and it answers the question without contradiction, respond YES. Respond with ONLY 'YES' or 'NO', nothing else."
        )

        question_part = f"Question: {question}\n\n" if question else ""
        user_template = _load_prompt("llm_exact_match_user.txt") or "{question_part}Gold Answer: {gold_answer}\n\nUser Answer: {user_answer}\n\nAre these semantically equivalent? Respond with only YES or NO."
        user_prompt = user_template.format(question_part=question_part, gold_answer=gold_answer, user_answer=user_answer)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.client.chat_completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=10  # Only need 'YES' or 'NO'
            )

            if debug:
                print(f"[DEBUG] Raw response: {repr(response)}")

            # Handle None or empty response
            if response is None:
                if debug:
                    print("[DEBUG] Response is None, falling back to string comparison")
                return 1 if gold_answer.lower() == user_answer.lower() else 0

            # Parse response - look for YES or NO anywhere in the response
            response_text = str(response).strip().upper()

            if debug:
                print(f"[DEBUG] Response text: [{response_text}]")

            if "YES" in response_text:
                return 1
            else:
                return 0

        except Exception as e:
            if debug:
                print(f"[DEBUG] Exception: {e}")
            print(f"Error in LLM-based exact match evaluation: {e}")
            # Fall back to string comparison on error
            return 1 if gold_answer.lower() == user_answer.lower() else 0

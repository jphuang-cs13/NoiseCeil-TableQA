"""
Token utilities for estimating and truncating table representations.

Heuristics:
- English-like text: ~1 token per 4 characters
- CJK (Chinese) text: ~1 token per 0.7 characters (0.6-0.8 range)

We use simple character-class based estimation to avoid heavyweight tokenizers.
"""

import re
from typing import Tuple

# Regex ranges for CJK characters and related blocks
# BMP ranges (cover most Traditional/Simplified usage):
_CJK_BMP_REGEX = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\u2E80-\u2EFF\u2F00-\u2FDF\u3000-\u303F\u31C0-\u31EF\uFE30-\uFE4F\uFF00-\uFFEF]"
)
# Supplementary planes (Ext B–G). Python regex needs \UXXXXXXXX escapes.
try:
    _CJK_SUP_REGEX = re.compile(r"[\U00020000-\U0003134F]")
except re.error:
    # Some environments may not support high-plane ranges; fallback to no-op
    _CJK_SUP_REGEX = re.compile(r"^$")


def estimate_tokens(text: str) -> Tuple[int, int, int, float]:
    """
    Estimate token count based on character composition.

    Returns (cjk_chars, non_cjk_chars, total_chars, tokens_estimate)
    """
    if not isinstance(text, str):
        s = str(text)
    else:
        s = text

    total_chars = len(s)
    if total_chars == 0:
        return 0, 0, 0, 0.0

    # Count CJK chars across BMP and supplementary planes
    cjk_chars = len(_CJK_BMP_REGEX.findall(s)) + len(_CJK_SUP_REGEX.findall(s))
    non_cjk_chars = total_chars - cjk_chars

    # Heuristic: tokens ≈ non-CJK / 4 + CJK / 0.7
    tokens_estimate = (non_cjk_chars / 4.0) + (cjk_chars / 0.7)
    return cjk_chars, non_cjk_chars, total_chars, tokens_estimate


def truncate_to_max_tokens(text: str, max_tokens: int = 8000) -> str:
    """
    Truncate the text so that the estimated tokens do not exceed max_tokens.

    Uses binary search over character length to find the largest prefix
    that fits under the token budget based on the heuristic estimator.
    """
    if not isinstance(text, str):
        s = str(text)
    else:
        s = text

    # Fast path: already within budget
    _, _, _, tokens = estimate_tokens(s)
    if tokens <= max_tokens:
        return s

    # Binary search for cutoff
    lo, hi = 0, len(s)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        _, _, _, t_est = estimate_tokens(s[:mid])
        if t_est <= max_tokens:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return s[:best]

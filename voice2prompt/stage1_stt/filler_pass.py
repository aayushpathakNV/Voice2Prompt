"""
Stage 1 — Rule-based filler pre-pass.

Runs on CPU in < 5 ms after transcription. Reduces Stage 2 input by ~25%,
saving ~60-90 ms of LLM generation time on the RTX 1000 Ada.

Strips:
  - Exact-match fillers: um, uh, umm, you know, i mean, basically, literally,
    sort of, kind of — always stripped regardless of position
  - Sentence-boundary fillers: right, okay, ok — only at sentence start/end
  - Adverbial "like": "it was, like, really fast" — only when comma-bounded
    or before an adjective, NOT when used as a verb ("I like Python")
  - False starts: repeated partial phrases within a 5-word window
  - STT punctuation artifacts and extra whitespace

Does NOT restructure content — that is Stage 2's responsibility.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Filler word lists
# ---------------------------------------------------------------------------

# Always stripped, any position. Multi-word entries must come before single-word.
_STANDALONE_FILLERS: tuple[str, ...] = (
    # multi-word first (processed before single-word to avoid partial matches)
    "you know what i mean",
    "you know what",
    "you know",
    "i mean",
    "like i said",
    "sort of like",
    "kind of like",
    "sort of",
    "kind of",
    # single-word
    "um",
    "uh",
    "umm",
    "uhh",
    "hmm",
    "basically",
    "literally",
    "obviously",
    "clearly",
    "essentially",
)

# Stripped only at sentence boundaries (start-of-sentence or before .?!).
# Deliberately excludes "so" and "well" — they are common non-filler sentence
# starters ("So the plan is..." / "Well, actually...").
_BOUNDARY_FILLERS: tuple[str, ...] = ("right", "okay", "ok")

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Adverbial "like": strip only when comma-bounded (",  like,  ") or when
# followed immediately by an adjective/adverb (not preceded by a verb that
# uses "like" as its main verb).
# We identify the adverbial case by requiring at least one surrounding comma
# OR the pattern ", like " without a following determiner/possessive.
_LIKE_COMMA_BOUNDED = re.compile(
    r",\s*like\s*,",
    re.IGNORECASE,
)
_LIKE_BEFORE_ADJ = re.compile(
    # "was/is/are/... like <adj>" — capture the verb, replace "like " with nothing.
    # Uses a capturing group instead of variable-width lookbehind (not supported in re).
    r"((?:was|is|are|were|feels?|seems?|looked?|sounds?)\s+)"
    r"like\s+"
    r"(?!(?:a|an|the|this|that|these|those|my|your|his|her|its|our|their)\b)",
    re.IGNORECASE,
)

# False start: phrase of 1–5 words immediately repeated (e.g. "I want to I want to")
_FALSE_START = re.compile(
    r"\b((?:\w+\s+){1,4}\w+)\s+\1\b",
    re.IGNORECASE,
)

# Cleanup patterns
_MULTI_SPACE = re.compile(r" {2,}")
_TRAILING_COMMA_BEFORE_PUNCT = re.compile(r",\s*(?=[.!?])")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")
_ORPHAN_LEADING_PUNCT = re.compile(r"^\s*[,;]\s*")

# ---------------------------------------------------------------------------
# Pre-compiled standalone filler patterns (built once at module load)
# ---------------------------------------------------------------------------

def _build_standalone_patterns() -> list[re.Pattern]:
    patterns = []
    for filler in _STANDALONE_FILLERS:
        # Match the filler with optional surrounding comma/space
        patterns.append(re.compile(
            rf"(?:,\s*)?\b{re.escape(filler)}\b(?:\s*,)?",
            re.IGNORECASE,
        ))
    return patterns

_STANDALONE_PATTERNS = _build_standalone_patterns()


def _build_boundary_patterns() -> list[tuple[re.Pattern, re.Pattern]]:
    """Return (start_pattern, end_pattern) pairs for each boundary filler."""
    pairs = []
    for filler in _BOUNDARY_FILLERS:
        f = re.escape(filler)
        # At sentence start: beginning of string or after .?! (with optional space/comma)
        start = re.compile(
            rf"(?:(?:^|(?<=[.!?])\s+)){f}\b[,]?\s*",
            re.IGNORECASE,
        )
        # At sentence end: before .?! or at end of string (with optional comma before)
        end = re.compile(
            rf"\s*,?\s*\b{f}\b(?=\s*[.!?]|$)",
            re.IGNORECASE,
        )
        pairs.append((start, end))
    return pairs

_BOUNDARY_PATTERNS = _build_boundary_patterns()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filler_pass(text: str, extra_fillers: list[str] | None = None) -> str:
    """
    Remove filler words and false starts from a raw STT transcript.

    Args:
        text:          Raw transcript string from Stage 1 STT.
        extra_fillers: Optional additional filler strings to strip (from config).
                       Supports multi-word phrases.

    Returns:
        Cleaned transcript string. Preserves all technical terms, numbers,
        names, and sentence structure.
    """
    if not text:
        return text

    # Handle extra_fillers from config: build and apply inline
    if extra_fillers:
        for filler in sorted(extra_fillers, key=len, reverse=True):
            text = re.sub(
                rf"(?:,\s*)?\b{re.escape(filler.lower())}\b(?:\s*,)?",
                " ",
                text,
                flags=re.IGNORECASE,
            )

    # Standalone fillers (pre-compiled, longest-first order)
    for pattern in _STANDALONE_PATTERNS:
        text = pattern.sub(" ", text)

    # Boundary fillers
    for start_pat, end_pat in _BOUNDARY_PATTERNS:
        text = start_pat.sub("", text)
        text = end_pat.sub("", text)

    # Adverbial "like" (comma-bounded and verb-preceded cases only)
    text = _LIKE_COMMA_BOUNDED.sub(",", text)
    text = _LIKE_BEFORE_ADJ.sub(r"\1", text)  # keep the verb, drop "like "

    # False starts (up to 3 passes to catch cascades)
    for _ in range(3):
        cleaned = _FALSE_START.sub(r"\1", text)
        if cleaned == text:
            break
        text = cleaned

    # Cleanup
    text = _TRAILING_COMMA_BEFORE_PUNCT.sub("", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _ORPHAN_LEADING_PUNCT.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)

    return text.strip()

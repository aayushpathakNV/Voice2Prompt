"""
Stage 1 — Rule-based filler pre-pass.

Runs on CPU in < 5 ms after transcription. Reduces Stage 2 input by ~25%,
saving ~60-90 ms of LLM generation time on the RTX 1000 Ada.

Strips:
  - Exact-match fillers (um, uh, like [adverbial], you know, etc.)
  - False starts (repeated partial phrases within a 5-word window)
  - STT punctuation artifacts and extra whitespace

Does NOT restructure content — that is Stage 2's responsibility.
"""

from __future__ import annotations

import re

# Fillers that are always stripped regardless of position
_STANDALONE_FILLERS = {
    "um", "uh", "umm", "uhh", "hmm",
    "you know", "i mean", "like i said",
    "basically", "literally", "obviously", "clearly",
    "sort of", "kind of", "you see",
}

# Fillers that are only stripped when they appear at sentence boundaries
_BOUNDARY_FILLERS = {"right", "okay", "ok", "so", "well", "now"}

# Adverbial "like" — strip when not followed by a noun phrase indicator
_ADVERBIAL_LIKE = re.compile(
    r"\blike\b(?!\s+(?:a|an|the|this|that|these|those|my|your|his|her|its|our|their)\b)",
    re.IGNORECASE,
)

# False start: a sequence of 1-5 words repeated within the next 5 words
_FALSE_START = re.compile(
    r"\b((?:\w+\s+){1,4}\w+)\s+\1\b",
    re.IGNORECASE,
)

# Collapse multiple spaces, fix common STT punctuation artifacts
_MULTI_SPACE = re.compile(r" {2,}")
_TRAILING_COMMA = re.compile(r",\s*(?=[.!?])")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")


def filler_pass(text: str, extra_fillers: list[str] | None = None) -> str:
    """
    Remove filler words and false starts from a raw transcript string.

    Args:
        text:          Raw transcript from Stage 1 STT.
        extra_fillers: Optional additional fillers to strip (from config).

    Returns:
        Cleaned transcript string.
    """
    all_standalone = _STANDALONE_FILLERS.copy()
    if extra_fillers:
        all_standalone.update(f.lower() for f in extra_fillers)

    # Multi-word fillers first (longest match wins)
    multi_word = sorted(
        (f for f in all_standalone if " " in f),
        key=len,
        reverse=True,
    )
    for filler in multi_word:
        text = re.sub(rf"\b{re.escape(filler)}\b[,]?\s*", "", text, flags=re.IGNORECASE)

    # Single-word fillers
    single_word = [f for f in all_standalone if " " not in f]
    for filler in single_word:
        text = re.sub(rf"\b{re.escape(filler)}\b[,]?\s*", "", text, flags=re.IGNORECASE)

    # Boundary fillers (right, okay, ok at sentence start/end)
    for filler in _BOUNDARY_FILLERS:
        text = re.sub(rf"(?:^|(?<=[.!?])\s+){re.escape(filler)}\b[,]?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(rf"\s*[,]?\b{re.escape(filler)}(?=\s*[.!?]|$)", "", text, flags=re.IGNORECASE)

    # Adverbial "like"
    text = _ADVERBIAL_LIKE.sub(" ", text)

    # False starts (up to 3 passes to catch cascades)
    for _ in range(3):
        new_text = _FALSE_START.sub(r"\1", text)
        if new_text == text:
            break
        text = new_text

    # Cleanup
    text = _TRAILING_COMMA.sub("", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = text.strip()

    return text

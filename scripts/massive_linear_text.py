"""Shared, dependency-free text normalization for the MASSIVE linear model."""

from __future__ import annotations

import unicodedata


def text_key(value: str) -> str:
    """Conservative duplicate key: casefold, NFKC, drop spacing/punctuation."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def model_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()

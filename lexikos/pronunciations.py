"""Shared parsing for bundled pronunciation-dictionary rows."""

from typing import Tuple


def split_pronunciation_variants(value: str) -> Tuple[str, ...]:
    """Split Lexikos ``~`` and CharsiuG2P comma-separated variants."""
    variants = tuple(
        variant.strip() for group in value.split(" ~ ") for variant in group.split(",")
    )
    if not variants or any(not variant for variant in variants):
        raise ValueError("pronunciation variants must be non-empty")
    return variants

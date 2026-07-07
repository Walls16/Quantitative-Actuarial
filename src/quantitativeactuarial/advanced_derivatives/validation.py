"""Shared validation helpers for advanced derivatives models."""

from __future__ import annotations


def validate_positive(name: str, value: float) -> None:
    """Raise ``ValueError`` when ``value`` is not strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def validate_nonnegative(name: str, value: float) -> None:
    """Raise ``ValueError`` when ``value`` is negative."""
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


def validate_between(name: str, value: float, lower: float, upper: float) -> None:
    """Raise ``ValueError`` when ``value`` is outside ``(lower, upper)``."""
    if not lower < value < upper:
        raise ValueError(f"{name} must be between {lower} and {upper}.")


def validate_positive_integer(name: str, value: int) -> None:
    """Raise ``ValueError`` when ``value`` is not a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def validate_option_type(option_type: str) -> None:
    """Raise ``ValueError`` unless ``option_type`` is ``call`` or ``put``."""
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

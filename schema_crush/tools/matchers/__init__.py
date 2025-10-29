"""matcher infrastructure for schema mapping."""

from .base import BaseMatcher
from .rule_matcher import RuleMatcher

__all__ = [
    "BaseMatcher",
    "RuleMatcher",
]
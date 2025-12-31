"""matcher infrastructure for schema mapping."""

from .base import BaseMatcher
from .rule_matcher import RuleMatcher
from .biobert_matcher import BioBERTMatcher

# magneto has dependency conflicts (Click==4.1 vs Click>=7.0)
# import lazily to avoid breaking other imports
try:
    from .magneto_matcher import MagnetoMatcher
except (ImportError, AssertionError) as e:
    MagnetoMatcher = None
    import warnings
    warnings.warn(f"MagnetoMatcher unavailable: {e}")

__all__ = [
    "BaseMatcher",
    "RuleMatcher",
    "BioBERTMatcher",
    "MagnetoMatcher",
]
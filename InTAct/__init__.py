"""
InTAct (Interval-based Task Activation Consolidation) Unlearning
"""

from .intact import UnlearnIntervalProtection, classification_forward_fn

__all__ = ['UnlearnIntervalProtection', 'classification_forward_fn']

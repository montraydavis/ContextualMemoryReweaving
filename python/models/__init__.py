"""Models package for ContextMemoryReweaving.

This package provides stable import paths used across tests, demos, and
documentation. Where possible, modules re-export implementations from
`core/` to avoid duplication.
"""

from .memory_entry import MemoryEntry
from .memory_buffer import LayeredMemoryBuffer
from .reconstruction import LayeredStateReconstructor

__all__ = [
    "MemoryEntry",
    "LayeredMemoryBuffer",
    "LayeredStateReconstructor",
]

"""
Neural Network Models for Contextual Memory Reweaving (CMR).

This module contains only the core neural network models and data models:
- Base transformer models with CMR capabilities
- Specialized model variants (e.g., Mistral integration)
- Relevance scoring neural networks
- Core data models
"""

# Core neural network models
from .base_transformer import CMRTransformer
from .mistral_integration import MistralCMRModel
from .relevance_scorer import RelevanceScorer

# Data models
from .memory_entry import MemoryEntry

__all__ = [
    # Neural network models
    'CMRTransformer',
    'MistralCMRModel',
    'RelevanceScorer',

    # Data models
    'MemoryEntry'
]
"""
Utility components for Contextual Memory Reweaving (CMR).

This module contains utility classes and helper functionality:
- Hook management for capturing intermediate states
- Memory hierarchy organization
- Retrieval caching utilities
"""

# Existing utilities
from .hooks import HookManager

# New utility classes (to be moved from models)
# from .memory_hierarchy import MemoryHierarchy
# from .retrieval_cache import RetrievalCache

__all__ = [
    # Existing utilities
    'HookManager',

    # Memory organization utilities will be added as they are moved
]
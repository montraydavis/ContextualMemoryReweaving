#!/usr/bin/env python3
"""
LayeredMemoryBuffer - Core Memory Management
Multi-layer memory buffer for storing and retrieving contextual states.
Implements efficient storage with automatic cleanup and relevance-based eviction.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import time
import numpy as np

# Import the MemoryEntry data model
from models.memory_entry import MemoryEntry


class LayeredMemoryBuffer:
    """
    Multi-layer memory buffer for storing and retrieving contextual states.
    Implements efficient storage with automatic cleanup and relevance-based eviction.
    """
    
    def __init__(self, 
                 max_entries_per_layer: int = 1000,
                 max_total_entries: int = 5000,
                 eviction_strategy: str = "lru_relevance",
                 cleanup_threshold: float = 0.8):
        """
        Initialize memory buffer.
        
        Args:
            max_entries_per_layer: Maximum entries per transformer layer
            max_total_entries: Maximum total entries across all layers
            eviction_strategy: Strategy for removing old entries ("lru", "relevance", "lru_relevance")
            cleanup_threshold: Trigger cleanup when buffer reaches this fraction of capacity
        """
        self.max_entries_per_layer = max_entries_per_layer
        self.max_total_entries = max_total_entries
        self.eviction_strategy = eviction_strategy
        self.cleanup_threshold = cleanup_threshold
        
        # Storage structures
        self.layer_buffers: Dict[int, List[MemoryEntry]] = defaultdict(list)
        self.entry_count = 0
        self.global_sequence_id = 0
        
        # Indexing for fast retrieval
        self.sequence_index: Dict[int, List[MemoryEntry]] = defaultdict(list)
        self.relevance_index: List[MemoryEntry] = []  # Sorted by relevance
        
        # Statistics
        self.stats = {
            'total_insertions': 0,
            'total_retrievals': 0,
            'total_evictions': 0,
            'cleanup_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        # Performance tracking
        self.access_patterns = defaultdict(int)
        self.layer_usage = defaultdict(int)

    def store_state(self,
                   hidden_state: torch.Tensor,
                   layer_idx: int,
                   position_idx: int,
                   relevance_score: float,
                   sequence_id: Optional[int] = None) -> bool:
        """
        Store a hidden state in the memory buffer.

        Args:
            hidden_state: Hidden state tensor to store
            layer_idx: Transformer layer index
            position_idx: Position within the sequence
            relevance_score: Relevance score for this state
            sequence_id: Sequence identifier (auto-generated if None)

        Returns:
            bool: True if stored successfully, False if rejected
        """
        # Auto-generate sequence ID if not provided
        if sequence_id is None:
            sequence_id = self.global_sequence_id

        # Create memory entry
        entry = MemoryEntry(
            hidden_state=hidden_state.detach().clone(),
            layer_idx=layer_idx,
            sequence_id=sequence_id,
            position_idx=position_idx,
            relevance_score=relevance_score,
            timestamp=time.time()
        )

        # Check if we need cleanup before insertion
        if self._should_cleanup():
            self._cleanup_buffer()

        # Check capacity constraints
        if not self._can_insert(layer_idx):
            # Try to evict lower relevance entries
            if not self._evict_for_insertion(layer_idx, relevance_score):
                return False  # Cannot make space

        # Insert the entry
        self._insert_entry(entry)
        self.stats['total_insertions'] += 1

        return True

    def retrieve_by_layer(self,
                         layer_idx: int,
                         k: Optional[int] = None,
                         min_relevance: float = 0.0) -> List[MemoryEntry]:
        """
        Retrieve stored states from specific layer.

        Args:
            layer_idx: Layer to retrieve from
            k: Maximum number of entries to return (None for all)
            min_relevance: Minimum relevance score threshold

        Returns:
            List of memory entries sorted by relevance (descending)
        """
        if layer_idx not in self.layer_buffers:
            return []

        # Filter by relevance threshold
        candidates = [
            entry for entry in self.layer_buffers[layer_idx]
            if entry.relevance_score >= min_relevance
        ]

        # Sort by relevance (descending)
        candidates.sort(key=lambda x: x.relevance_score, reverse=True)

        # Limit results if k specified
        if k is not None:
            candidates = candidates[:k]

        # Update access statistics
        for entry in candidates:
            entry.update_access()
        self.stats['total_retrievals'] += len(candidates)

        return candidates

    def retrieve_by_sequence(self,
                           sequence_id: int,
                           layer_indices: Optional[List[int]] = None) -> Dict[int, List[MemoryEntry]]:
        """
        Retrieve all stored states from a specific sequence.

        Args:
            sequence_id: Sequence identifier
            layer_indices: Specific layers to retrieve (None for all)

        Returns:
            Dictionary mapping layer_idx -> list of memory entries
        """
        if sequence_id not in self.sequence_index:
            return {}

        sequence_entries = self.sequence_index[sequence_id]
        result = defaultdict(list)

        for entry in sequence_entries:
            if layer_indices is None or entry.layer_idx in layer_indices:
                result[entry.layer_idx].append(entry)
                entry.update_access()

        # Sort each layer's entries by position
        for layer_entries in result.values():
            layer_entries.sort(key=lambda x: x.position_idx)

        self.stats['total_retrievals'] += sum(len(entries) for entries in result.values())

        return dict(result)

    def retrieve_top_k_relevant(self,
                              k: int,
                              layer_indices: Optional[List[int]] = None,
                              exclude_sequence: Optional[int] = None) -> List[MemoryEntry]:
        """
        Retrieve top-k most relevant entries across all layers.

        Args:
            k: Number of entries to retrieve
            layer_indices: Restrict to specific layers (None for all)
            exclude_sequence: Exclude entries from this sequence ID

        Returns:
            List of top-k most relevant memory entries
        """
        candidates = []

        for layer_idx, entries in self.layer_buffers.items():
            if layer_indices is None or layer_idx in layer_indices:
                for entry in entries:
                    if exclude_sequence is None or entry.sequence_id != exclude_sequence:
                        candidates.append(entry)

        # Sort by relevance and take top-k
        candidates.sort(key=lambda x: x.relevance_score, reverse=True)
        top_k = candidates[:k]

        # Update access statistics
        for entry in top_k:
            entry.update_access()
        self.stats['total_retrievals'] += len(top_k)

        return top_k

    def _can_insert(self, layer_idx: int) -> bool:
        """Check if we can insert into the specified layer."""
        layer_count = len(self.layer_buffers[layer_idx])
        return (layer_count < self.max_entries_per_layer and
                self.entry_count < self.max_total_entries)

    def _should_cleanup(self) -> bool:
        """Check if cleanup is needed."""
        return self.entry_count >= int(self.max_total_entries * self.cleanup_threshold)

    def _insert_entry(self, entry: MemoryEntry):
        """Insert entry into all relevant data structures."""
        # Add to layer buffer
        self.layer_buffers[entry.layer_idx].append(entry)

        # Add to sequence index
        self.sequence_index[entry.sequence_id].append(entry)

        # Add to relevance index (maintain sorted order)
        self._insert_into_relevance_index(entry)

        self.entry_count += 1

    def _insert_into_relevance_index(self, entry: MemoryEntry):
        """Insert entry into relevance index maintaining sort order."""
        # Binary search for insertion point
        left, right = 0, len(self.relevance_index)

        while left < right:
            mid = (left + right) // 2
            if self.relevance_index[mid].relevance_score > entry.relevance_score:
                left = mid + 1
            else:
                right = mid

        self.relevance_index.insert(left, entry)

    def _evict_for_insertion(self, layer_idx: int, new_relevance: float) -> bool:
        """Try to evict entries to make space for new insertion."""
        if self.eviction_strategy == "lru":
            return self._evict_lru(layer_idx)
        elif self.eviction_strategy == "relevance":
            return self._evict_low_relevance(layer_idx, new_relevance)
        elif self.eviction_strategy == "lru_relevance":
            return self._evict_lru_relevance(layer_idx, new_relevance)
        return False

    def _evict_lru(self, layer_idx: int) -> bool:
        """Evict least recently used entry from layer."""
        if not self.layer_buffers[layer_idx]:
            return False

        # Find LRU entry
        lru_entry = min(self.layer_buffers[layer_idx],
                       key=lambda x: x.last_access if x.last_access > 0 else x.timestamp)

        self._remove_entry(lru_entry)
        return True

    def _evict_low_relevance(self, layer_idx: int, new_relevance: float) -> bool:
        """Evict lowest relevance entry if it's lower than new entry."""
        if not self.layer_buffers[layer_idx]:
            return False

        # Find lowest relevance entry in this layer
        min_entry = min(self.layer_buffers[layer_idx], key=lambda x: x.relevance_score)

        if min_entry.relevance_score < new_relevance:
            self._remove_entry(min_entry)
            return True

        return False

    def _evict_lru_relevance(self, layer_idx: int, new_relevance: float) -> bool:
        """Combined LRU and relevance-based eviction."""
        if not self.layer_buffers[layer_idx]:
            return False

        # Score entries based on both recency and relevance
        def eviction_score(entry):
            recency_score = time.time() - (entry.last_access if entry.last_access > 0 else entry.timestamp)
            return entry.relevance_score - 0.1 * recency_score  # Weighted combination

        worst_entry = min(self.layer_buffers[layer_idx], key=eviction_score)

        if eviction_score(worst_entry) < new_relevance:
            self._remove_entry(worst_entry)
            return True

        return False

    def _remove_entry(self, entry: MemoryEntry):
        """Remove entry from all data structures."""
        # Remove from layer buffer
        self.layer_buffers[entry.layer_idx].remove(entry)

        # Remove from sequence index
        self.sequence_index[entry.sequence_id].remove(entry)
        if not self.sequence_index[entry.sequence_id]:
            del self.sequence_index[entry.sequence_id]

        # Remove from relevance index
        self.relevance_index.remove(entry)

        self.entry_count -= 1
        self.stats['total_evictions'] += 1

    def _cleanup_buffer(self):
        """Perform cleanup operations to free memory."""
        # Time-based cleanup first
        current_time = time.time()
        cleanup_age = 3600  # 1 hour threshold

        entries_to_remove = []
        for entry in list(self.relevance_index):
            if current_time - entry.timestamp > cleanup_age and entry.access_count == 0:
                entries_to_remove.append(entry)

        for entry in entries_to_remove:
            self._remove_entry(entry)

        # Capacity-based cleanup: if above threshold, evict least relevant until below threshold
        target_capacity = int(self.max_total_entries * self.cleanup_threshold)
        if self.entry_count > target_capacity:
            self.cleanup_least_relevant(target_entries=target_capacity)

        # Enforce per-layer capacity as well
        for layer_idx, entries in list(self.layer_buffers.items()):
            if len(entries) > self.max_entries_per_layer:
                # Remove least relevant entries in this layer until capacity met
                overflow = len(entries) - self.max_entries_per_layer
                # Sort ascending by relevance within the layer
                sorted_layer = sorted(entries, key=lambda e: e.relevance_score)
                for entry in sorted_layer[:overflow]:
                    self._remove_entry(entry)

        self.stats['cleanup_operations'] += 1

    def get_buffer_stats(self) -> Dict:
        """Get comprehensive buffer statistics."""
        layer_stats = {}
        for layer_idx, entries in self.layer_buffers.items():
            layer_stats[layer_idx] = {
                'count': len(entries),
                'avg_relevance': np.mean([e.relevance_score for e in entries]) if entries else 0,
                'avg_access_count': np.mean([e.access_count for e in entries]) if entries else 0
            }

        return {
            'total_entries': self.entry_count,
            'layer_distribution': layer_stats,
            'memory_utilization': self.entry_count / self.max_total_entries,
            'total_sequences': len(self.sequence_index),
            **self.stats
        }

    def get_total_entries(self) -> int:
        """Get total number of entries in the buffer."""
        return self.entry_count

    def get_entries_per_layer(self) -> Dict[int, int]:
        """Get number of entries per layer."""
        return {layer_idx: len(entries) for layer_idx, entries in self.layer_buffers.items()}

    def get_memory_size_mb(self) -> float:
        """Get approximate memory usage in MB."""
        # Rough estimate: each entry has a hidden state tensor
        # Assuming average hidden state size of 768 dimensions
        estimated_size_bytes = self.entry_count * 768 * 4  # 4 bytes per float32
        return estimated_size_bytes / (1024 * 1024)  # Convert to MB

    def cleanup_least_relevant(self, target_entries: int = 100):
        """Remove least relevant entries to free up space."""
        if self.entry_count <= target_entries:
            return

        # Sort entries by relevance score (ascending)
        sorted_entries = sorted(self.relevance_index, key=lambda e: e.relevance_score)

        # Remove the least relevant entries
        entries_to_remove = sorted_entries[:self.entry_count - target_entries]
        for entry in entries_to_remove:
            self._remove_entry(entry)

    def clear_sequence(self, sequence_id: int):
        """Remove all entries from a specific sequence."""
        if sequence_id in self.sequence_index:
            entries_to_remove = self.sequence_index[sequence_id].copy()
            for entry in entries_to_remove:
                self._remove_entry(entry)

    def clear_all(self):
        """Clear all stored memories."""
        self.layer_buffers.clear()
        self.sequence_index.clear()
        self.relevance_index.clear()
        self.entry_count = 0

        # Reset stats
        for key in self.stats:
            self.stats[key] = 0

    def get_statistics(self) -> Dict:
        """Alias for get_buffer_stats for compatibility."""
        return self.get_buffer_stats()

    def save_state(self, filepath: str):
        """Save memory buffer state to file."""
        import pickle
        state = {
            'layer_buffers': self.layer_buffers,
            'entry_count': self.entry_count,
            'global_sequence_id': self.global_sequence_id,
            'sequence_index': self.sequence_index,
            'relevance_index': self.relevance_index,
            'stats': self.stats,
            'max_entries_per_layer': self.max_entries_per_layer,
            'max_total_entries': self.max_total_entries,
            'eviction_strategy': self.eviction_strategy,
            'cleanup_threshold': self.cleanup_threshold,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """Load memory buffer state from file."""
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.layer_buffers = state['layer_buffers']
        self.entry_count = state['entry_count']
        self.global_sequence_id = state['global_sequence_id']
        self.sequence_index = state['sequence_index']
        self.relevance_index = state['relevance_index']
        self.stats = state['stats']
        self.max_entries_per_layer = state.get('max_entries_per_layer', self.max_entries_per_layer)
        self.max_total_entries = state.get('max_total_entries', self.max_total_entries)
        self.eviction_strategy = state.get('eviction_strategy', self.eviction_strategy)
        self.cleanup_threshold = state.get('cleanup_threshold', self.cleanup_threshold)

    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all memory entries across all layers."""
        all_entries = []
        for layer_entries in self.layer_buffers.values():
            all_entries.extend(layer_entries)
        return all_entries

    def optimize(self):
        """Run memory optimization: evict least relevant and rebuild indices."""
        # Target to keep 90% of current entries
        target_entries = max(1, int(self.entry_count * 0.9))
        self.cleanup_least_relevant(target_entries=target_entries)
        # Rebuild relevance index from remaining entries
        self.relevance_index = []
        for entries in self.layer_buffers.values():
            for entry in entries:
                self._insert_into_relevance_index(entry)

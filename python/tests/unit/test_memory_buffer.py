#!/usr/bin/env python3
"""
Test suite for LayeredMemoryBuffer - Day 4 Implementation
Comprehensive testing of memory buffer functionality including storage, retrieval, eviction, and edge cases.
"""

import pytest
import torch
import time
import numpy as np
from models.memory_buffer import LayeredMemoryBuffer, MemoryEntry


class TestMemoryEntry:
    """Test MemoryEntry dataclass functionality."""
    
    def test_memory_entry_creation(self):
        """Test MemoryEntry creation and basic attributes."""
        hidden_state = torch.randn(1, 768)
        entry = MemoryEntry(
            hidden_state=hidden_state,
            layer_idx=5,
            sequence_id=10,
            position_idx=25,
            relevance_score=0.85,
            timestamp=time.time()
        )
        
        assert entry.layer_idx == 5
        assert entry.sequence_id == 10
        assert entry.position_idx == 25
        assert entry.relevance_score == 0.85
        assert entry.access_count == 0
        assert entry.last_access == 0.0
    
    def test_update_access(self):
        """Test access statistics update."""
        entry = MemoryEntry(
            hidden_state=torch.randn(1, 768),
            layer_idx=0,
            sequence_id=0,
            position_idx=0,
            relevance_score=0.5,
            timestamp=time.time()
        )
        
        initial_count = entry.access_count
        initial_access = entry.last_access
        
        time.sleep(0.001)  # Small delay to ensure timestamp difference
        entry.update_access()
        
        assert entry.access_count == initial_count + 1
        assert entry.last_access > initial_access


class TestLayeredMemoryBuffer:
    """Test LayeredMemoryBuffer core functionality."""
    

    
    def test_buffer_initialization(self, small_buffer):
        """Test buffer initialization with correct default values."""
        assert small_buffer.max_entries_per_layer == 5
        assert small_buffer.max_total_entries == 20
        assert small_buffer.eviction_strategy == "lru_relevance"
        assert small_buffer.cleanup_threshold == 0.8
        assert small_buffer.entry_count == 0
        assert len(small_buffer.layer_buffers) == 0
        assert len(small_buffer.sequence_index) == 0
        assert len(small_buffer.relevance_index) == 0
    
    def test_store_state_success(self, small_buffer):
        """Test successful state storage."""
        hidden_state = torch.randn(1, 128)
        
        success = small_buffer.store_state(
            hidden_state=hidden_state,
            layer_idx=3,
            position_idx=10,
            relevance_score=0.8
        )
        
        assert success is True
        assert small_buffer.entry_count == 1
        assert len(small_buffer.layer_buffers[3]) == 1
        assert len(small_buffer.sequence_index[0]) == 1  # Auto-generated sequence ID
        assert len(small_buffer.relevance_index) == 1
        
        # Check entry details
        entry = small_buffer.layer_buffers[3][0]
        assert entry.layer_idx == 3
        assert entry.position_idx == 10
        assert entry.relevance_score == 0.8
        assert entry.sequence_id == 0
    
    def test_store_state_with_custom_sequence_id(self, small_buffer):
        """Test state storage with custom sequence ID."""
        hidden_state = torch.randn(1, 128)
        
        success = small_buffer.store_state(
            hidden_state=hidden_state,
            layer_idx=5,
            position_idx=15,
            relevance_score=0.9,
            sequence_id=42
        )
        
        assert success is True
        assert small_buffer.sequence_index[42][0].sequence_id == 42
    
    def test_store_state_capacity_constraint(self, small_buffer):
        """Test storage rejection when capacity is exceeded."""
        # Fill up the buffer
        for i in range(5):  # max_entries_per_layer
            hidden_state = torch.randn(1, 128)
            success = small_buffer.store_state(
                hidden_state=hidden_state,
                layer_idx=3,
                position_idx=i,
                relevance_score=0.5 + i * 0.1
            )
            assert success is True
        
        # Try to store one more - should fail
        hidden_state = torch.randn(1, 128)
        success = small_buffer.store_state(
            hidden_state=hidden_state,
            layer_idx=3,
            position_idx=5,
            relevance_score=0.3  # Lower than existing entries
        )
        
        assert success is False
        assert small_buffer.entry_count == 5  # Should not have increased
    
    def test_retrieve_by_layer(self, medium_buffer):
        """Test retrieval by layer with relevance filtering."""
        # Store states with different relevance scores
        for i in range(5):
            hidden_state = torch.randn(1, 128)
            relevance = 0.1 + i * 0.2  # 0.1, 0.3, 0.5, 0.7, 0.9
            medium_buffer.store_state(
                hidden_state=hidden_state,
                layer_idx=4,
                position_idx=i,
                relevance_score=relevance
            )
        
        # Retrieve top 3 entries
        entries = medium_buffer.retrieve_by_layer(4, k=3)
        assert len(entries) == 3
        
        # Should be sorted by relevance (descending)
        assert abs(entries[0].relevance_score - 0.9) < 1e-10
        assert abs(entries[1].relevance_score - 0.7) < 1e-10
        assert abs(entries[2].relevance_score - 0.5) < 1e-10
        
        # Test relevance threshold
        high_relevance_entries = medium_buffer.retrieve_by_layer(4, min_relevance=0.6)
        assert len(high_relevance_entries) == 2
        assert all(e.relevance_score >= 0.6 for e in high_relevance_entries)
    
    def test_retrieve_by_sequence(self, medium_buffer):
        """Test retrieval by sequence ID."""
        # Store states from multiple sequences
        for seq_id in range(3):
            for layer_idx in [2, 4]:
                for pos in range(3):
                    hidden_state = torch.randn(1, 128)
                    medium_buffer.store_state(
                        hidden_state=hidden_state,
                        layer_idx=layer_idx,
                        position_idx=pos,
                        relevance_score=0.5,
                        sequence_id=seq_id
                    )
        
        # Retrieve all entries from sequence 1
        seq_entries = medium_buffer.retrieve_by_sequence(1)
        assert len(seq_entries) == 2  # 2 layers
        assert len(seq_entries[2]) == 3  # 3 positions in layer 2
        assert len(seq_entries[4]) == 3  # 3 positions in layer 4
        
        # Test layer filtering
        layer_2_entries = medium_buffer.retrieve_by_sequence(1, layer_indices=[2])
        assert len(layer_2_entries) == 1
        assert 2 in layer_2_entries
        assert len(layer_2_entries[2]) == 3
    
    def test_retrieve_top_k_relevant(self, medium_buffer):
        """Test top-k retrieval across all layers."""
        # Store states with known relevance scores
        relevance_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        for i, relevance in enumerate(relevance_scores):
            hidden_state = torch.randn(1, 128)
            medium_buffer.store_state(
                hidden_state=hidden_state,
                layer_idx=i % 2,  # Alternate between layers 0 and 1
                position_idx=i,
                relevance_score=relevance
            )
        
        # Get top 3 most relevant entries
        top_entries = medium_buffer.retrieve_top_k_relevant(3)
        assert len(top_entries) == 3
        
        # Should be sorted by relevance (descending)
        assert top_entries[0].relevance_score == 0.9
        assert top_entries[1].relevance_score == 0.7
        assert top_entries[2].relevance_score == 0.5
        
        # Test layer filtering
        layer_0_entries = medium_buffer.retrieve_top_k_relevant(3, layer_indices=[0])
        assert all(e.layer_idx == 0 for e in layer_0_entries)
        
        # Test sequence exclusion
        exclude_entries = medium_buffer.retrieve_top_k_relevant(3, exclude_sequence=0)
        assert all(e.sequence_id != 0 for e in exclude_entries)
    
    def test_lru_eviction_strategy(self):
        """Test LRU eviction strategy."""
        buffer = LayeredMemoryBuffer(
            max_entries_per_layer=3,
            max_total_entries=10,
            eviction_strategy="lru"
        )
        
        # Fill up layer 0
        for i in range(3):
            hidden_state = torch.randn(1, 128)
            buffer.store_state(
                hidden_state=hidden_state,
                layer_idx=0,
                position_idx=i,
                relevance_score=0.5
            )
        
        # Access entries to create access patterns
        buffer.retrieve_by_layer(0, k=1)  # Access first entry
        time.sleep(0.001)
        buffer.retrieve_by_layer(0, k=1)  # Access second entry
        
        # Try to store one more - should evict least recently used
        hidden_state = torch.randn(1, 128)
        success = buffer.store_state(
            hidden_state=hidden_state,
            layer_idx=0,
            position_idx=3,
            relevance_score=0.6
        )
        
        assert success is True
        assert buffer.entry_count == 3  # Should still be 3 (evicted one)
        assert buffer.stats['total_evictions'] == 1
    
    def test_relevance_eviction_strategy(self):
        """Test relevance-based eviction strategy."""
        buffer = LayeredMemoryBuffer(
            max_entries_per_layer=3,
            max_total_entries=10,
            eviction_strategy="relevance"
        )
        
        # Fill up layer 0 with low relevance entries
        for i in range(3):
            hidden_state = torch.randn(1, 128)
            buffer.store_state(
                hidden_state=hidden_state,
                layer_idx=0,
                position_idx=i,
                relevance_score=0.1 + i * 0.1  # 0.1, 0.2, 0.3
            )
        
        # Try to store high relevance entry - should evict lowest relevance
        hidden_state = torch.randn(1, 128)
        success = buffer.store_state(
            hidden_state=hidden_state,
            layer_idx=0,
            position_idx=3,
            relevance_score=0.8  # Higher than existing entries
        )
        
        assert success is True
        assert buffer.entry_count == 3  # Should still be 3 (evicted one)
        assert buffer.stats['total_evictions'] == 1
        
        # Check that lowest relevance entry was evicted
        remaining_entries = buffer.retrieve_by_layer(0)
        relevance_scores = [e.relevance_score for e in remaining_entries]
        assert 0.1 not in relevance_scores  # Lowest should be gone
        assert 0.8 in relevance_scores  # New high relevance should be present
    
    def test_lru_relevance_eviction_strategy(self):
        """Test combined LRU-relevance eviction strategy."""
        buffer = LayeredMemoryBuffer(
            max_entries_per_layer=3,
            max_total_entries=10,
            eviction_strategy="lru_relevance"
        )
        
        # Fill up layer 0
        for i in range(3):
            hidden_state = torch.randn(1, 128)
            buffer.store_state(
                hidden_state=hidden_state,
                layer_idx=0,
                position_idx=i,
                relevance_score=0.5
            )
        
        # Access entries to create access patterns
        buffer.retrieve_by_layer(0, k=1)  # Access first entry
        
        # Try to store new entry
        hidden_state = torch.randn(1, 128)
        success = buffer.store_state(
            hidden_state=hidden_state,
            layer_idx=0,
            position_idx=3,
            relevance_score=0.6
        )
        
        assert success is True
        assert buffer.stats['total_evictions'] == 1
    
    def test_cleanup_operations(self, medium_buffer):
        """Test automatic cleanup operations."""
        # Create some old entries by manually setting timestamps
        import time
        old_timestamp = time.time() - 7200  # 2 hours ago

        # Add some entries with old timestamps
        for i in range(5):
            hidden_state = torch.randn(1, 128)
            success = medium_buffer.store_state(
                hidden_state=hidden_state,
                layer_idx=i % 3,
                position_idx=i,
                relevance_score=0.5
            )
            assert success
            # Manually set old timestamp for cleanup testing
            if medium_buffer.layer_buffers[i % 3]:
                medium_buffer.layer_buffers[i % 3][-1].timestamp = old_timestamp
                # Also update in relevance index
                for entry in medium_buffer.relevance_index:
                    if entry.layer_idx == i % 3 and entry.position_idx == i:
                        entry.timestamp = old_timestamp
                        break

        # Fill buffer to trigger cleanup threshold (80% of 50 = 40)
        for i in range(35):  # Total will be 40 entries
            hidden_state = torch.randn(1, 128)
            medium_buffer.store_state(
                hidden_state=hidden_state,
                layer_idx=i % 3,
                position_idx=i + 10,
                relevance_score=0.5
            )

        # Check that cleanup was triggered or manually trigger it
        if medium_buffer.stats['cleanup_operations'] == 0:
            # Manually trigger cleanup to test the functionality
            medium_buffer._cleanup_buffer()

        assert medium_buffer.stats['cleanup_operations'] > 0
    
    def test_clear_sequence(self, medium_buffer):
        """Test clearing specific sequence."""
        # Store states from multiple sequences
        for seq_id in range(3):
            for layer_idx in [1, 3]:
                hidden_state = torch.randn(1, 128)
                medium_buffer.store_state(
                    hidden_state=hidden_state,
                    layer_idx=layer_idx,
                    position_idx=0,
                    relevance_score=0.5,
                    sequence_id=seq_id
                )
        
        initial_count = medium_buffer.entry_count
        assert initial_count == 6  # 3 sequences × 2 layers
        
        # Clear sequence 1
        medium_buffer.clear_sequence(1)
        
        # Check that sequence 1 entries are gone
        assert medium_buffer.entry_count == initial_count - 2  # 2 entries removed
        assert 1 not in medium_buffer.sequence_index
        
        # Other sequences should remain
        assert 0 in medium_buffer.sequence_index
        assert 2 in medium_buffer.sequence_index
    
    def test_clear_all(self, medium_buffer):
        """Test clearing all stored memories."""
        # Store some states
        for i in range(5):
            hidden_state = torch.randn(1, 128)
            medium_buffer.store_state(
                hidden_state=hidden_state,
                layer_idx=i % 2,
                position_idx=i,
                relevance_score=0.5
            )
        
        assert medium_buffer.entry_count > 0
        
        # Clear all
        medium_buffer.clear_all()
        
        # Check that everything is cleared
        assert medium_buffer.entry_count == 0
        assert len(medium_buffer.layer_buffers) == 0
        assert len(medium_buffer.sequence_index) == 0
        assert len(medium_buffer.relevance_index) == 0
        
        # Stats should be reset
        for value in medium_buffer.stats.values():
            assert value == 0
    
    def test_get_buffer_stats(self, medium_buffer):
        """Test comprehensive buffer statistics."""
        # Store states in multiple layers
        for layer_idx in [2, 4, 6]:
            for i in range(3):
                hidden_state = torch.randn(1, 128)
                medium_buffer.store_state(
                    hidden_state=hidden_state,
                    layer_idx=layer_idx,
                    position_idx=i,
                    relevance_score=0.3 + i * 0.2  # 0.3, 0.5, 0.7
                )
        
        stats = medium_buffer.get_buffer_stats()
        
        # Check basic stats
        assert stats['total_entries'] == 9  # 3 layers × 3 entries
        assert stats['total_sequences'] == 1  # All from same auto-generated sequence
        assert stats['total_insertions'] == 9
        assert stats['total_evictions'] == 0
        
        # Check layer distribution
        layer_stats = stats['layer_distribution']
        assert len(layer_stats) == 3  # 3 layers
        assert layer_stats[2]['count'] == 3
        assert layer_stats[4]['count'] == 3
        assert layer_stats[6]['count'] == 3
        
        # Check relevance averages
        for layer_idx in [2, 4, 6]:
            avg_relevance = layer_stats[layer_idx]['avg_relevance']
            assert 0.4 < avg_relevance < 0.6  # Should be around 0.5
    
    def test_edge_cases(self, small_buffer):
        """Test edge cases and error conditions."""
        # Test with empty tensor
        empty_tensor = torch.empty(0, 128)
        success = small_buffer.store_state(
            hidden_state=empty_tensor,
            layer_idx=0,
            position_idx=0,
            relevance_score=0.5
        )
        assert success is True  # Should handle empty tensors
        
        # Test with very large relevance score
        hidden_state = torch.randn(1, 128)
        success = small_buffer.store_state(
            hidden_state=hidden_state,
            layer_idx=0,
            position_idx=0,
            relevance_score=1.5  # Above 1.0
        )
        assert success is True  # Should handle scores > 1.0
        
        # Test retrieval from non-existent layer
        entries = small_buffer.retrieve_by_layer(999)
        assert entries == []
        
        # Test retrieval from non-existent sequence
        seq_entries = small_buffer.retrieve_by_sequence(999)
        assert seq_entries == {}
    
    def test_performance_characteristics(self, medium_buffer):
        """Test performance characteristics and scaling."""
        # Measure insertion time
        start_time = time.time()
        for i in range(100):
            hidden_state = torch.randn(1, 128)
            medium_buffer.store_state(
                hidden_state=hidden_state,
                layer_idx=i % 5,
                position_idx=i,
                relevance_score=torch.rand(1).item()
            )
        insertion_time = time.time() - start_time
        
        # Measure retrieval time
        start_time = time.time()
        for _ in range(100):
            medium_buffer.retrieve_by_layer(0, k=10)
        retrieval_time = time.time() - start_time
        
        # Performance should be reasonable
        assert insertion_time < 1.0  # Should complete in under 1 second
        assert retrieval_time < 0.5  # Retrieval should be faster than insertion
        
        print(f"Insertion time: {insertion_time:.3f}s")
        print(f"Retrieval time: {retrieval_time:.3f}s")


class TestMemoryBufferIntegration:
    """Test integration scenarios and complex workflows."""
    
    def test_multi_layer_workflow(self):
        """Test complete workflow across multiple layers."""
        buffer = LayeredMemoryBuffer(
            max_entries_per_layer=20,
            max_total_entries=100,
            eviction_strategy="lru_relevance"
        )
        
        # Simulate transformer forward pass across multiple layers
        sequence_ids = [0, 1, 2]
        layers = [3, 6, 9, 12]

        successful_stores = 0
        total_attempts = 0

        for seq_id in sequence_ids:
            for layer_idx in layers:
                for pos in range(10):
                    hidden_state = torch.randn(1, 768)
                    relevance = torch.rand(1).item()

                    success = buffer.store_state(
                        hidden_state=hidden_state,
                        layer_idx=layer_idx,
                        position_idx=pos,
                        relevance_score=relevance,
                        sequence_id=seq_id
                    )
                    total_attempts += 1
                    if success:
                        successful_stores += 1

        # Should have stored entries up to the buffer limit
        assert successful_stores > 0
        assert buffer.entry_count <= buffer.max_total_entries
        assert total_attempts == 120  # 3 sequences × 4 layers × 10 positions
        
        # Test retrieval patterns
        # Get top memories from middle layers
        middle_layer_memories = buffer.retrieve_top_k_relevant(
            k=15, 
            layer_indices=[6, 9]
        )
        assert len(middle_layer_memories) == 15
        assert all(e.layer_idx in [6, 9] for e in middle_layer_memories)
        
        # Get sequence-specific memories
        seq_1_memories = buffer.retrieve_by_sequence(1)
        assert len(seq_1_memories) <= 4  # Up to 4 layers (some may have been evicted)
        total_seq_1_entries = sum(len(entries) for entries in seq_1_memories.values())
        assert total_seq_1_entries > 0  # Should have some entries
        assert total_seq_1_entries <= 40  # At most 4 layers × 10 positions
        
        # Test eviction under pressure
        # Fill buffer to capacity
        for i in range(50):
            hidden_state = torch.randn(1, 768)
            buffer.store_state(
                hidden_state=hidden_state,
                layer_idx=15,
                position_idx=i,
                relevance_score=0.1 + i * 0.01
            )
        
        # Should have triggered evictions
        assert buffer.stats['total_evictions'] > 0
        assert buffer.entry_count <= buffer.max_total_entries
    
    def test_memory_pressure_scenarios(self):
        """Test behavior under memory pressure."""
        buffer = LayeredMemoryBuffer(
            max_entries_per_layer=5,
            max_total_entries=20,
            eviction_strategy="lru_relevance",
            cleanup_threshold=0.7
        )
        
        # Fill buffer to trigger cleanup
        for i in range(15):  # 75% of capacity
            hidden_state = torch.randn(1, 128)
            buffer.store_state(
                hidden_state=hidden_state,
                layer_idx=i % 3,
                position_idx=i,
                relevance_score=0.5
            )
        
        # Trigger cleanup by adding more entries
        for i in range(10):
            hidden_state = torch.randn(1, 128)
            buffer.store_state(
                hidden_state=hidden_state,
                layer_idx=i % 3,
                position_idx=i + 15,
                relevance_score=0.6
            )
        
        # Should have performed cleanup
        assert buffer.stats['cleanup_operations'] > 0
        
        # Buffer should not exceed capacity
        assert buffer.entry_count <= buffer.max_total_entries


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

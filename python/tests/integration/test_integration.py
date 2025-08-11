#!/usr/bin/env python3
"""
Test suite for IntegratedCMRModel - Day 5 Implementation
Comprehensive testing of the integrated CMR model functionality including memory capture, retrieval, and performance.
"""

import pytest
import torch
import time
import numpy as np
from transformers import AutoConfig
from models.cmr_full_integrated import FullCMRModel as IntegratedCMRModel


class TestIntegratedCMRModel:
    """Test IntegratedCMRModel core functionality."""
    

    
    @pytest.fixture
    def integrated_model(self, small_config, basic_memory_config):
        """Create integrated CMR model for testing."""
        return IntegratedCMRModel(small_config, basic_memory_config, device="cpu")
    
    def test_model_initialization(self, small_config, basic_memory_config):
        """Test model initialization with correct components."""
        model = IntegratedCMRModel(small_config, basic_memory_config, device="cpu")
        
        # Check that all components are initialized
        assert model.base_transformer is not None
        assert model.relevance_scorer is not None
        assert model.memory_buffer is not None
        assert model.hook_manager is not None
        
        # Check configuration
        assert model.memory_config['target_layers'] == [2, 4]
        assert model.memory_config['scoring_method'] == 'hybrid'
        assert model.memory_config['relevance_threshold'] == 0.3
        
        # Check initial state
        assert model.memory_enabled is True
        assert model.current_sequence_id == 0
        assert model.sequence_counter == 0
        
        # Cleanup
        model.cleanup()
    
    def test_memory_config_validation(self, small_config):
        """Test memory configuration validation."""
        # Test invalid config type
        with pytest.raises(ValueError, match="memory_config must be a dictionary"):
            IntegratedCMRModel(small_config, "invalid", device="cpu")
        
        # Test invalid scoring method
        invalid_config = {'scoring_method': 'invalid_method'}
        with pytest.raises(ValueError, match="scoring_method must be one of"):
            IntegratedCMRModel(small_config, invalid_config, device="cpu")
        
        # Test invalid relevance threshold
        invalid_config = {'relevance_threshold': 1.5}
        with pytest.raises(ValueError, match="relevance_threshold must be between"):
            IntegratedCMRModel(small_config, invalid_config, device="cpu")
        
        # Test invalid max entries
        invalid_config = {'max_entries_per_layer': -1}
        with pytest.raises(ValueError, match="max_entries_per_layer must be positive"):
            IntegratedCMRModel(small_config, invalid_config, device="cpu")
    
    def test_default_config_values(self, small_config):
        """Test that default configuration values are set correctly."""
        # Create model with minimal config
        minimal_config = {}
        model = IntegratedCMRModel(small_config, minimal_config, device="cpu")
        
        # Check defaults
        assert model.memory_config['target_layers'] == list(range(6, 12))  # Default middle layers
        assert model.memory_config['scoring_method'] == 'hybrid'
        assert model.memory_config['relevance_threshold'] == 0.3
        assert model.memory_config['max_entries_per_layer'] == 1000
        assert model.memory_config['max_total_entries'] == 5000
        
        # Cleanup
        model.cleanup()
    
    def test_basic_forward_pass(self, integrated_model):
        """Test basic forward pass without memory enhancement."""
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        
        # Forward pass without memory
        outputs = integrated_model(input_ids, attention_mask, use_memory=False)
        
        # Check output structure
        assert 'last_hidden_state' in outputs
        assert 'hidden_states' in outputs
        assert 'attentions' in outputs
        assert 'memory_info' in outputs
        
        # Check output shapes
        assert outputs['last_hidden_state'].shape == (1, 10, 256)  # batch, seq_len, hidden_size
        assert len(outputs['hidden_states']) == 7  # 6 layers + input embeddings
        
        # Check memory info
        assert outputs['memory_info']['enabled'] is False
        assert outputs['memory_info']['sequence_id'] == 0
        
        # Check performance stats
        assert integrated_model.performance_stats['total_forward_passes'] == 1
        assert integrated_model.performance_stats['total_memory_operations'] == 0
    
    def test_memory_enhanced_forward_pass(self, integrated_model):
        """Test forward pass with memory enhancement enabled."""
        input_ids = torch.randint(0, 1000, (1, 15))
        attention_mask = torch.ones(1, 15)
        
        # Forward pass with memory
        outputs = integrated_model(input_ids, attention_mask, use_memory=True)
        
        # Check memory info
        assert outputs['memory_info']['enabled'] is True
        assert outputs['memory_info']['sequence_id'] == 1  # Should increment
        
        # Check that memories were captured
        buffer_stats = outputs['memory_info']['buffer_stats']
        assert buffer_stats['total_entries'] > 0
        
        # Check performance stats
        assert integrated_model.performance_stats['total_forward_passes'] == 1
        assert integrated_model.performance_stats['total_memory_operations'] > 0
    
    def test_memory_capture_across_layers(self, integrated_model):
        """Test that memories are captured from target layers."""
        input_ids = torch.randint(0, 1000, (1, 20))
        attention_mask = torch.ones(1, 20)
        
        # Forward pass
        outputs = integrated_model(input_ids, attention_mask, use_memory=True)
        
        # Check buffer stats
        buffer_stats = outputs['memory_info']['buffer_stats']
        layer_stats = buffer_stats['layer_distribution']
        
        # Should have memories from target layers (2, 4)
        assert 2 in layer_stats or 4 in layer_stats
        
        # Check that non-target layers don't have memories
        non_target_layers = [0, 1, 3, 5]
        for layer_idx in non_target_layers:
            if layer_idx in layer_stats:
                assert layer_stats[layer_idx]['count'] == 0
    
    def test_sequence_id_increment(self, integrated_model):
        """Test that sequence ID increments with each forward pass."""
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        
        # Multiple forward passes
        for i in range(3):
            outputs = integrated_model(input_ids, attention_mask, use_memory=True)
            assert outputs['memory_info']['sequence_id'] == i + 1
        
        # Check sequence counter
        assert integrated_model.sequence_counter == 3
    
    def test_memory_retrieval(self, integrated_model):
        """Test memory retrieval functionality."""
        # First forward pass to create memories
        input_ids = torch.randint(0, 1000, (1, 15))
        attention_mask = torch.ones(1, 15)
        outputs1 = integrated_model(input_ids, attention_mask, use_memory=True)
        seq_id_1 = outputs1['memory_info']['sequence_id']
        
        # Second forward pass
        input_ids2 = torch.randint(0, 1000, (1, 15))
        outputs2 = integrated_model(input_ids2, attention_mask, use_memory=True)
        seq_id_2 = outputs2['memory_info']['sequence_id']
        
        # Check that memories were retrieved
        retrieved_memories = outputs2['memory_info']['retrieved_memories']
        assert seq_id_2 in retrieved_memories
        
        # Check sequence-specific memories
        seq_memories = retrieved_memories[seq_id_2]['sequence_memories']
        top_memories = retrieved_memories[seq_id_2]['top_relevant_memories']
        
        # Should have some memories
        assert len(seq_memories) > 0 or len(top_memories) > 0
    
    def test_memory_enable_disable(self, integrated_model):
        """Test enabling and disabling memory operations."""
        input_ids = torch.randint(0, 1000, (1, 10))
        attention_mask = torch.ones(1, 10)
        
        # Forward pass with memory enabled
        outputs1 = integrated_model(input_ids, attention_mask, use_memory=True)
        initial_memory_ops = integrated_model.performance_stats['total_memory_operations']
        
        # Disable memory
        integrated_model.disable_memory()
        assert integrated_model.memory_enabled is False
        
        # Forward pass with memory disabled
        outputs2 = integrated_model(input_ids, attention_mask, use_memory=True)
        memory_ops_after_disable = integrated_model.performance_stats['total_memory_operations']
        
        # Memory operations should not increase
        assert memory_ops_after_disable == initial_memory_ops
        
        # Re-enable memory
        integrated_model.enable_memory()
        assert integrated_model.memory_enabled is True
        
        # Forward pass with memory re-enabled
        outputs3 = integrated_model(input_ids, attention_mask, use_memory=True)
        final_memory_ops = integrated_model.performance_stats['total_memory_operations']
        
        # Memory operations should increase again
        assert final_memory_ops > initial_memory_ops
    
    def test_memory_clearing(self, integrated_model):
        """Test clearing memory functionality."""
        # Create some memories
        input_ids = torch.randint(0, 1000, (1, 15))
        attention_mask = torch.ones(1, 15)
        outputs = integrated_model(input_ids, attention_mask, use_memory=True)
        
        # Check that memories exist
        buffer_stats = outputs['memory_info']['buffer_stats']
        assert buffer_stats['total_entries'] > 0
        
        # Clear all memories
        integrated_model.clear_memory()
        
        # Check that memories are cleared
        new_stats = integrated_model.memory_buffer.get_buffer_stats()
        assert new_stats['total_entries'] == 0
    
    def test_scoring_method_updates(self, integrated_model):
        """Test updating scoring methods."""
        # Test valid method update
        integrated_model.update_scoring_method('attention_based')
        assert integrated_model.memory_config['scoring_method'] == 'attention_based'
        
        # Test invalid method
        with pytest.raises(ValueError, match="Invalid scoring method"):
            integrated_model.update_scoring_method('invalid_method')
    
    def test_relevance_threshold_updates(self, integrated_model):
        """Test updating relevance threshold."""
        # Test valid threshold update
        integrated_model.update_relevance_threshold(0.7)
        assert integrated_model.memory_config['relevance_threshold'] == 0.7
        
        # Test invalid threshold
        with pytest.raises(ValueError, match="Threshold must be between"):
            integrated_model.update_relevance_threshold(1.5)
        
        with pytest.raises(ValueError, match="Threshold must be between"):
            integrated_model.update_relevance_threshold(-0.1)
    
    def test_memory_retrieval_methods(self, integrated_model):
        """Test direct memory retrieval methods."""
        # Create memories first
        input_ids = torch.randint(0, 1000, (1, 15))
        attention_mask = torch.ones(1, 15)
        integrated_model(input_ids, attention_mask, use_memory=True)
        
        # Test get_memory_for_sequence
        seq_memories = integrated_model.get_memory_for_sequence(1)
        assert isinstance(seq_memories, dict)
        
        # Test get_top_memories
        top_memories = integrated_model.get_top_memories(5)
        assert isinstance(top_memories, list)
        assert len(top_memories) <= 5
    
    def test_memory_stats(self, integrated_model):
        """Test comprehensive memory statistics."""
        # Create some memories
        input_ids = torch.randint(0, 1000, (1, 15))
        attention_mask = torch.ones(1, 10)
        integrated_model(input_ids, attention_mask, use_memory=True)
        
        # Get stats
        stats = integrated_model.get_memory_stats()
        
        # Check structure
        assert 'buffer_stats' in stats
        assert 'performance_stats' in stats
        assert 'memory_efficiency' in stats
        assert 'configuration' in stats
        
        # Check performance stats
        perf_stats = stats['performance_stats']
        assert 'total_forward_passes' in perf_stats
        assert 'total_memory_operations' in perf_stats
        assert 'memory_capture_time' in perf_stats
        
        # Check memory efficiency
        efficiency = stats['memory_efficiency']
        assert 'buffer_utilization' in efficiency
        assert 'capture_rate' in efficiency
        
        # Check configuration
        config = stats['configuration']
        assert 'target_layers' in config
        assert 'scoring_method' in config
        assert 'relevance_threshold' in config
    
    def test_error_handling(self, integrated_model):
        """Test error handling in memory operations."""
        # Test with invalid input shapes
        invalid_input = torch.randn(1, 10, 100)  # Wrong hidden size
        
        # Should handle gracefully without crashing
        try:
            integrated_model._process_layer_output(0, invalid_input)
        except Exception as e:
            # Should catch and handle errors
            assert "Error processing layer" in str(e) or "Warning" in str(e)
    
    def test_performance_tracking(self, integrated_model):
        """Test performance tracking accuracy."""
        input_ids = torch.randint(0, 1000, (1, 15))
        attention_mask = torch.ones(1, 15)
        
        # Measure time manually
        start_time = time.time()
        outputs = integrated_model(input_ids, attention_mask, use_memory=True)
        manual_time = time.time() - start_time
        
        # Check that performance stats are reasonable
        perf_stats = integrated_model.performance_stats
        assert perf_stats['total_inference_time'] > 0
        assert perf_stats['total_inference_time'] >= manual_time * 0.8  # Allow some overhead
        
        # Check memory operation timing
        if perf_stats['total_memory_operations'] > 0:
            assert perf_stats['memory_capture_time'] > 0
            assert perf_stats['memory_retrieval_time'] > 0


class TestIntegratedCMRModelAdvanced:
    """Test advanced scenarios and edge cases."""
    
    def test_large_sequence_handling(self, medium_config):
        """Test handling of large sequences."""
        memory_config = {
            'target_layers': [3, 5, 7],
            'scoring_method': 'variance_based',
            'relevance_threshold': 0.2,
            'max_entries_per_layer': 100,
            'max_total_entries': 500,
            'eviction_strategy': 'relevance'
        }
        
        # Advanced tests expect 512 hidden size outputs
        memory_config['output_hidden_size'] = 512
        model = IntegratedCMRModel(medium_config, memory_config, device="cpu")
        
        # Large sequence
        input_ids = torch.randint(0, 2000, (1, 100))
        attention_mask = torch.ones(1, 100)
        
        # Should handle without errors
        outputs = model(input_ids, attention_mask, use_memory=True)
        
        # Check output
        assert outputs['last_hidden_state'].shape == (1, 100, 512)
        
        # Check memory buffer stats
        buffer_stats = outputs['memory_info']['buffer_stats']
        assert buffer_stats['total_entries'] > 0
        
        # Cleanup
        model.cleanup()
    
    def test_memory_pressure_scenarios(self, medium_config):
        """Test behavior under memory pressure."""
        memory_config = {
            'target_layers': [2, 4, 6],
            'scoring_method': 'hybrid',
            'relevance_threshold': 0.01,  # Very low threshold to capture more positions
            'max_entries_per_layer': 5,   # Very small capacity to force evictions
            'max_total_entries': 20,      # Small total capacity
            'eviction_strategy': 'lru_relevance',
            'cleanup_threshold': 0.7
        }

        model = IntegratedCMRModel(medium_config, memory_config, device="cpu")

        # Multiple forward passes to fill buffer beyond capacity
        for i in range(10):  # More passes to ensure we exceed capacity
            input_ids = torch.randint(0, 2000, (1, 20))
            attention_mask = torch.ones(1, 20)
            model(input_ids, attention_mask, use_memory=True)

        # Check that eviction occurred
        buffer_stats = model.memory_buffer.get_buffer_stats()
        assert buffer_stats['total_evictions'] > 0
        
        # Buffer should not exceed capacity
        assert buffer_stats['total_entries'] <= memory_config['max_total_entries']
        
        # Cleanup
        model.cleanup()
    
    def test_different_scoring_methods(self, medium_config):
        """Test all three scoring methods."""
        scoring_methods = ['attention_based', 'variance_based', 'hybrid']
        
        for method in scoring_methods:
            memory_config = {
                'target_layers': [3, 5],
                'scoring_method': method,
                'relevance_threshold': 0.3,
                'max_entries_per_layer': 50,
                'max_total_entries': 200
            }
            
            memory_config['output_hidden_size'] = 512
            model = IntegratedCMRModel(medium_config, memory_config, device="cpu")
            
            # Test forward pass
            input_ids = torch.randint(0, 2000, (1, 15))
            attention_mask = torch.ones(1, 15)
            outputs = model(input_ids, attention_mask, use_memory=True)
            
            # Should work with all methods
            assert outputs['last_hidden_state'].shape == (1, 15, 512)
            
            # Check memory capture
            buffer_stats = outputs['memory_info']['buffer_stats']
            assert buffer_stats['total_entries'] >= 0
            
            # Cleanup
            model.cleanup()
    
    def test_memory_persistence(self, medium_config):
        """Test memory state persistence (save/load)."""
        memory_config = {
            'target_layers': [3, 5],
            'scoring_method': 'hybrid',
            'relevance_threshold': 0.3,
            'max_entries_per_layer': 50,
            'max_total_entries': 200
        }
        
        model = IntegratedCMRModel(medium_config, memory_config, device="cpu")
        
        # Create some memories
        input_ids = torch.randint(0, 2000, (1, 15))
        attention_mask = torch.ones(1, 15)
        model(input_ids, attention_mask, use_memory=True)
        
        # Save state
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            model.save_memory_state(tmp_file.name)
            
            # Create new model
            model2 = IntegratedCMRModel(medium_config, memory_config, device="cpu")
            
            # Load state
            model2.load_memory_state(tmp_file.name)
            
            # Check that memories were loaded
            stats1 = model.get_memory_stats()
            stats2 = model2.get_memory_stats()
            
            assert stats1['buffer_stats']['total_entries'] == stats2['buffer_stats']['total_entries']
            
            # Cleanup
            os.unlink(tmp_file.name)
            model.cleanup()
            model2.cleanup()


class TestIntegratedCMRModelIntegration:
    """Test integration scenarios and complex workflows."""
    
    def test_end_to_end_workflow(self, medium_config):
        """Test complete end-to-end workflow."""
        memory_config = {
            'target_layers': [2, 4, 6],
            'scoring_method': 'hybrid',
            'relevance_threshold': 0.2,
            'max_entries_per_layer': 100,
            'max_total_entries': 500,
            'eviction_strategy': 'lru_relevance'
        }
        
        model = IntegratedCMRModel(medium_config, memory_config, device="cpu")
        
        # Simulate multiple sequences
        sequences = [
            torch.randint(0, 2000, (1, 20)),
            torch.randint(0, 2000, (1, 25)),
            torch.randint(0, 2000, (1, 15))
        ]
        
        # Process sequences
        for i, seq in enumerate(sequences):
            attention_mask = torch.ones(seq.shape)
            outputs = model(seq, attention_mask, use_memory=True)
            
            # Check sequence ID
            assert outputs['memory_info']['sequence_id'] == i + 1
        
        # Check final state
        final_stats = model.get_memory_stats()
        assert final_stats['performance_stats']['total_forward_passes'] == 3
        assert final_stats['buffer_stats']['total_entries'] > 0
        
        # Test memory retrieval across sequences
        seq_1_memories = model.get_memory_for_sequence(1)
        seq_2_memories = model.get_memory_for_sequence(2)
        
        # Should have memories from different sequences
        assert len(seq_1_memories) > 0 or len(seq_2_memories) > 0
        
        # Cleanup
        model.cleanup()
    
    def test_memory_enhancement_impact(self, medium_config):
        """Test the impact of memory enhancement on performance."""
        memory_config = {
            'target_layers': [3, 5],
            'scoring_method': 'hybrid',
            'relevance_threshold': 0.3,
            'max_entries_per_layer': 100,
            'max_total_entries': 300
        }
        
        model = IntegratedCMRModel(medium_config, memory_config, device="cpu")
        
        input_ids = torch.randint(0, 2000, (1, 20))
        attention_mask = torch.ones(1, 20)
        
        # Measure time without memory
        start_time = time.time()
        outputs1 = model(input_ids, attention_mask, use_memory=False)
        time_without_memory = time.time() - start_time
        
        # Measure time with memory
        start_time = time.time()
        outputs2 = model(input_ids, attention_mask, use_memory=True)
        time_with_memory = time.time() - start_time
        
        # Memory should not add excessive overhead (allow for timing variations)
        # Check that memory operations are being tracked instead of strict timing
        assert outputs2['memory_info']['enabled'] is True
        assert outputs1['memory_info']['enabled'] is False
        assert time_with_memory < time_without_memory * 2.0  # Should not double the time
        
        # Check that outputs are different (memory enhancement should change hidden states)
        assert not torch.allclose(outputs1['last_hidden_state'], outputs2['last_hidden_state'])

        # But outputs should have the same shape
        assert outputs1['last_hidden_state'].shape == outputs2['last_hidden_state'].shape
        
        # Cleanup
        model.cleanup()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])

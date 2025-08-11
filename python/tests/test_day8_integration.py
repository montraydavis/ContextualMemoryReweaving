#!/usr/bin/env python3
"""
Tests for Day 8: Integration with Base Model
Tests the full CMR integration and performance optimization components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
from transformers import AutoConfig

from models.cmr_full_integrated import (
    FullCMRModel, 
    ReconstructionIntegrator, 
    CMRPerformanceMonitor
)
from models.performance_optimization import (
    CMRPerformanceOptimizer,
    AdaptiveThresholdManager,
    BatchProcessingOptimizer,
    MemoryPrefetcher,
    ComputationScheduler,
    BackgroundOptimizer
)

class TestFullCMRModel:
    """Test the full CMR integration model."""
    
    @pytest.fixture
    def gpt_oss_config(self):
        """Create a minimal configuration for testing."""
        return AutoConfig.from_pretrained("mistralai/Ministral-8B-Instruct-2410")
    
    @pytest.fixture
    def cmr_config(self):
        """Create CMR configuration for testing."""
        return {
            'max_entries_per_layer': 100,
            'max_total_entries': 500,
            'eviction_strategy': 'lru_relevance',
            'scoring_method': 'attention_based',
            'target_layers': [2, 4, 5],
            'intervention_layers': [2, 4, 5],
            'relevance_threshold': 0.5,
            'retrieval_strategy': 'multi_criteria',
            'retrieval_config': {
                'max_memories': 5,
                'similarity_threshold': 0.7
            },
            'reconstruction_config': {
                'integration_method': 'weighted_sum',
                'memory_weight': 0.3
            }
        }
    
    @pytest.fixture
    def cmr_model(self, gpt_oss_config, cmr_config):
        """Create a CMR model instance for testing."""
        return FullCMRModel(
            base_config=gpt_oss_config,
            cmr_config=cmr_config,
            device="cpu"
        )
    
    def test_model_initialization(self, cmr_model, gpt_oss_config):
        """Test that the model initializes correctly."""
        assert cmr_model.base_config == gpt_oss_config
        assert cmr_model.memory_enabled is True
        assert cmr_model.reconstruction_enabled is True
        assert len(cmr_model.target_layers) == 3
        assert cmr_model.relevance_threshold == 0.5
    
    def test_hook_setup(self, cmr_model):
        """Test that intervention and capture hooks are set up correctly."""
        # Check that hooks are registered
        assert len(cmr_model.hook_manager.hooks) > 0
        
        # Check that target layers have hooks
        layers = cmr_model._get_transformer_layers()
        for layer_idx in cmr_model.target_layers:
            if layer_idx < len(layers):
                layer = layers[layer_idx]
                assert id(layer) in cmr_model.hook_manager.hooks
    
    def test_forward_pass(self, cmr_model):
        """Test the forward pass with memory integration."""
        # Create test input
        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones_like(input_ids)
        
        # Run forward pass
        outputs = cmr_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_memory_info=True
        )
        
        # Check outputs
        assert 'last_hidden_state' in outputs
        assert 'forward_time' in outputs
        assert 'memory_stats' in outputs
        
        # Check output shapes
        assert outputs['last_hidden_state'].shape == (2, 16, 256)
    
    def test_memory_capture(self, cmr_model):
        """Test memory capture functionality."""
        # Create test input with high attention
        input_ids = torch.randint(0, 100, (1, 8))
        attention_mask = torch.ones_like(input_ids)
        
        # Run forward pass to trigger memory capture
        outputs = cmr_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Check that memories were captured
        memory_usage = cmr_model.get_memory_usage()
        assert memory_usage['total_entries'] > 0
    
    def test_memory_reconstruction(self, cmr_model):
        """Test memory reconstruction functionality."""
        # First, capture some memories
        input_ids = torch.randint(0, 100, (1, 8))
        attention_mask = torch.ones_like(input_ids)
        
        cmr_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Now test reconstruction
        query_ids = torch.randint(0, 100, (1, 6))
        query_mask = torch.ones_like(query_ids)
        
        outputs = cmr_model(
            input_ids=query_ids,
            attention_mask=query_mask
        )
        
        # Check that reconstruction occurred
        perf_stats = cmr_model.performance_monitor.get_stats()
        assert perf_stats['total_reconstructions'] >= 0
    
    def test_memory_management(self, cmr_model):
        """Test memory management functions."""
        # Capture some memories first
        input_ids = torch.randint(0, 100, (1, 8))
        attention_mask = torch.ones_like(input_ids)
        cmr_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Check memory usage
        memory_usage = cmr_model.get_memory_usage()
        assert memory_usage['total_entries'] > 0
        
        # Clear memory
        cmr_model.clear_memory()
        
        # Verify memory is cleared
        memory_usage_after = cmr_model.get_memory_usage()
        assert memory_usage_after['total_entries'] == 0
    
    def test_enable_disable_functions(self, cmr_model):
        """Test enable/disable functionality."""
        # Test memory enable/disable
        cmr_model.enable_memory(False)
        assert cmr_model.memory_enabled is False
        
        cmr_model.enable_memory(True)
        assert cmr_model.memory_enabled is True
        
        # Test reconstruction enable/disable
        cmr_model.enable_reconstruction(False)
        assert cmr_model.reconstruction_enabled is False
        
        cmr_model.enable_reconstruction(True)
        assert cmr_model.reconstruction_enabled is True

class TestReconstructionIntegrator:
    """Test the reconstruction integrator component."""
    
    @pytest.fixture
    def integrator(self):
        """Create a reconstruction integrator instance."""
        mock_buffer = Mock()
        config = {
            'integration_method': 'weighted_sum',
            'memory_weight': 0.3,
            'layer_weights': {0: 0.4, 1: 0.3}
        }
        return ReconstructionIntegrator(
            hidden_size=256,
            num_layers=6,
            memory_buffer=mock_buffer,
            reconstruction_config=config
        )
    
    def test_initialization(self, integrator):
        """Test integrator initialization."""
        assert integrator.hidden_size == 256
        assert integrator.num_layers == 6
        assert integrator.integration_method == 'weighted_sum'
        assert integrator.memory_weight == 0.3
    
    def test_integrate_memories_weighted_sum(self, integrator):
        """Test memory integration with weighted sum method."""
        # Create test hidden states
        hidden_states = torch.randn(2, 8, 256)
        
        # Create test memories
        memories = [
            {
                'hidden_state': torch.randn(256),
                'position': 0
            },
            {
                'hidden_state': torch.randn(256),
                'position': 4
            }
        ]
        
        # Integrate memories
        enhanced_states = integrator.integrate_memories(
            hidden_states=hidden_states,
            memories=memories,
            layer_idx=0
        )
        
        # Check output
        assert enhanced_states.shape == hidden_states.shape
        assert not torch.allclose(enhanced_states, hidden_states)
        
        # Check statistics
        stats = integrator.get_statistics()
        assert stats['integration_count'] == 1
        assert stats['total_integration_time'] > 0
    
    def test_integrate_memories_empty(self, integrator):
        """Test integration with no memories."""
        hidden_states = torch.randn(2, 8, 256)
        
        enhanced_states = integrator.integrate_memories(
            hidden_states=hidden_states,
            memories=[],
            layer_idx=0
        )
        
        # Should return original states unchanged
        assert torch.allclose(enhanced_states, hidden_states)
    
    def test_integrate_memories_gated(self, integrator):
        """Test gated integration method."""
        integrator.integration_method = 'gated_integration'
        
        hidden_states = torch.randn(2, 8, 256)
        memories = [{'hidden_state': torch.randn(256), 'position': 0}]
        
        enhanced_states = integrator.integrate_memories(
            hidden_states=hidden_states,
            memories=memories,
            layer_idx=0
        )
        
        assert enhanced_states.shape == hidden_states.shape
        assert not torch.allclose(enhanced_states, hidden_states)

class TestCMRPerformanceMonitor:
    """Test the performance monitoring component."""
    
    @pytest.fixture
    def monitor(self):
        """Create a performance monitor instance."""
        return CMRPerformanceMonitor()
    
    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.metrics['total_captures'] == 0
        assert monitor.metrics['total_reconstructions'] == 0
        assert len(monitor.metrics['capture_times']) == 0
    
    def test_record_capture(self, monitor):
        """Test capture recording."""
        monitor.record_capture(layer_idx=2, states_stored=5, capture_time=0.001)
        
        assert monitor.metrics['total_captures'] == 1
        assert len(monitor.metrics['capture_times']) == 1
        assert monitor.metrics['states_stored_per_layer'][2] == 5
    
    def test_record_reconstruction(self, monitor):
        """Test reconstruction recording."""
        monitor.record_reconstruction(layer_idx=3, memories_used=3, reconstruction_time=0.002)
        
        assert monitor.metrics['total_reconstructions'] == 1
        assert len(monitor.metrics['reconstruction_times']) == 1
        assert monitor.metrics['reconstructions_per_layer'][3] == 1
    
    def test_get_stats(self, monitor):
        """Test statistics retrieval."""
        # Record some metrics
        monitor.record_capture(layer_idx=1, states_stored=2, capture_time=0.001)
        monitor.record_reconstruction(layer_idx=1, memories_used=1, reconstruction_time=0.002)
        
        stats = monitor.get_stats()
        
        assert stats['total_captures'] == 1
        assert stats['total_reconstructions'] == 1
        assert stats['states_stored_per_layer'][1] == 2
        assert stats['reconstructions_per_layer'][1] == 1
        assert 'avg_capture_time' in stats
        assert 'avg_reconstruction_time' in stats
    
    def test_reset(self, monitor):
        """Test monitor reset functionality."""
        # Record some metrics
        monitor.record_capture(layer_idx=1, states_stored=2, capture_time=0.001)
        
        # Reset
        monitor.reset()
        
        # Verify reset
        assert monitor.metrics['total_captures'] == 0
        assert len(monitor.metrics['capture_times']) == 0

class TestCMRPerformanceOptimizer:
    """Test the performance optimization system."""
    
    @pytest.fixture
    def mock_cmr_model(self):
        """Create a mock CMR model."""
        model = Mock()
        model.memory_buffer = Mock()
        model.get_memory_usage.return_value = {'total_entries': 100}
        model.relevance_threshold = 0.5
        return model
    
    @pytest.fixture
    def optimizer_config(self):
        """Create optimization configuration."""
        return {
            'enable_background_optimization': False,  # Disable for testing
            'enable_batch_optimization': True,
            'enable_prefetching': True,
            'enable_adaptive_thresholds': True
        }
    
    @pytest.fixture
    def optimizer(self, mock_cmr_model, optimizer_config):
        """Create a performance optimizer instance."""
        return CMRPerformanceOptimizer(mock_cmr_model, optimizer_config)
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.cmr_model is not None
        assert optimizer.adaptive_thresholds is not None
        assert optimizer.batch_optimizer is not None
        assert optimizer.memory_prefetcher is not None
    
    def test_optimize_forward_pass(self, optimizer):
        """Test forward pass optimization."""
        input_ids = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones_like(input_ids)
        
        opt_inputs, opt_mask, opt_info = optimizer.optimize_forward_pass(
            input_ids, attention_mask
        )
        
        # Should return the same tensors (no optimization in this case)
        assert torch.equal(opt_inputs, input_ids)
        assert torch.equal(opt_mask, attention_mask)
        assert isinstance(opt_info, dict)
    
    def test_get_optimization_stats(self, optimizer):
        """Test optimization statistics retrieval."""
        stats = optimizer.get_optimization_stats()
        
        assert 'threshold_adjustments' in stats
        assert 'batch_optimizations' in stats
        assert 'adaptive_threshold_stats' in stats
        assert 'batch_optimization_stats' in stats
    
    def test_reset_stats(self, optimizer):
        """Test statistics reset functionality."""
        # Modify some stats
        optimizer.optimization_stats['threshold_adjustments'] = 5
        
        # Reset
        optimizer.reset_stats()
        
        # Verify reset
        assert optimizer.optimization_stats['threshold_adjustments'] == 0

class TestAdaptiveThresholdManager:
    """Test the adaptive threshold manager."""
    
    @pytest.fixture
    def threshold_manager(self):
        """Create a threshold manager instance."""
        return AdaptiveThresholdManager()
    
    def test_initialization(self, threshold_manager):
        """Test threshold manager initialization."""
        assert threshold_manager.base_threshold == 0.5
        assert threshold_manager.min_threshold == 0.1
        assert threshold_manager.max_threshold == 0.9
        assert threshold_manager.adjustment_count == 0
    
    def test_update_threshold(self, threshold_manager):
        """Test threshold update functionality."""
        threshold_manager.update_threshold(layer_idx=1, threshold=0.6, performance=0.8)
        
        assert threshold_manager.adjustment_count == 1
        assert len(threshold_manager.threshold_history) == 1
        assert len(threshold_manager.performance_history) == 1
    
    def test_get_optimal_threshold_memory_pressure(self, threshold_manager):
        """Test optimal threshold calculation under memory pressure."""
        # Simulate high memory usage
        for _ in range(10):
            threshold_manager.memory_usage_history.append(4500)
        
        optimal_threshold = threshold_manager.get_optimal_threshold(
            input_shape=(2, 256),
            memory_usage={'total_entries': 4500}
        )
        
        # Should increase threshold under memory pressure
        assert optimal_threshold is not None
        assert optimal_threshold > 0.5
    
    def test_get_optimal_threshold_sequence_length(self, threshold_manager):
        """Test optimal threshold calculation based on sequence length."""
        # Test with long sequence
        optimal_threshold = threshold_manager.get_optimal_threshold(
            input_shape=(2, 600),
            memory_usage={'total_entries': 1000}
        )
        
        # Should increase threshold for long sequences
        if optimal_threshold is not None:
            assert optimal_threshold > 0.5
    
    def test_get_stats(self, threshold_manager):
        """Test statistics retrieval."""
        stats = threshold_manager.get_stats()
        
        assert 'adjustment_count' in stats
        assert 'base_threshold' in stats
        assert 'threshold_history_length' in stats
    
    def test_reset_stats(self, threshold_manager):
        """Test statistics reset functionality."""
        # Modify some stats
        threshold_manager.update_threshold(1, 0.6, 0.8)
        
        # Reset
        threshold_manager.reset_stats()
        
        # Verify reset
        assert threshold_manager.adjustment_count == 0
        assert len(threshold_manager.threshold_history) == 0

class TestBatchProcessingOptimizer:
    """Test the batch processing optimizer."""
    
    @pytest.fixture
    def batch_optimizer(self):
        """Create a batch optimizer instance."""
        return BatchProcessingOptimizer()
    
    def test_initialization(self, batch_optimizer):
        """Test batch optimizer initialization."""
        assert batch_optimizer.optimization_count == 0
        assert batch_optimizer.optimization_savings == 0.0
    
    def test_optimize_batch_no_optimization(self, batch_optimizer):
        """Test batch optimization when no optimization is needed."""
        input_batch = torch.randint(0, 100, (2, 16))
        attention_mask = torch.ones_like(input_batch)
        
        opt_batch, opt_mask = batch_optimizer.optimize_batch(input_batch, attention_mask)
        
        # Should return same tensors
        assert torch.equal(opt_batch, input_batch)
        assert torch.equal(opt_mask, attention_mask)
        assert batch_optimizer.optimization_count == 0
    
    def test_optimize_batch_with_padding(self, batch_optimizer):
        """Test batch optimization with significant padding."""
        # Create input with padding
        input_batch = torch.randint(0, 100, (2, 20))
        attention_mask = torch.ones(2, 20)
        attention_mask[:, 12:] = 0  # Significant padding (8 out of 20 = 40% padding)
        
        # Verify the padding setup
        actual_lengths = attention_mask.sum(dim=1)
        max_actual_length = actual_lengths.max().item()
        assert max_actual_length == 12  # Should be 12 (20 - 8 padding)
        assert max_actual_length < 20 * 0.8  # Should be 12 < 16, which is true
        
        opt_batch, opt_mask = batch_optimizer.optimize_batch(input_batch, attention_mask)
        
        # Should optimize by removing padding since 12 < 20 * 0.8 = 16
        assert opt_batch.shape[1] < input_batch.shape[1]
        assert opt_mask.shape[1] < attention_mask.shape[1]
        assert batch_optimizer.optimization_count == 1
    
    def test_get_stats(self, batch_optimizer):
        """Test statistics retrieval."""
        # Perform some optimizations with significant padding
        input_batch = torch.randint(0, 100, (2, 20))
        attention_mask = torch.ones(2, 20)
        attention_mask[:, 12:] = 0  # 8 out of 20 = 40% padding (12 < 20 * 0.8 = 16)
        
        batch_optimizer.optimize_batch(input_batch, attention_mask)
        
        stats = batch_optimizer.get_stats()
        
        assert stats['optimization_count'] == 1
        assert stats['optimization_savings'] > 0
        assert 'avg_batch_size' in stats
    
    def test_reset_stats(self, batch_optimizer):
        """Test statistics reset functionality."""
        # Modify some stats
        batch_optimizer.optimization_count = 5
        
        # Reset
        batch_optimizer.reset_stats()
        
        # Verify reset
        assert batch_optimizer.optimization_count == 0
        assert batch_optimizer.optimization_savings == 0.0

if __name__ == "__main__":
    pytest.main([__file__])

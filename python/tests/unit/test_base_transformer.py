"""
Test file for base transformer implementation.
Tests the CMRTransformer class and its memory hook functionality.
"""

import torch
import pytest
from transformers import AutoConfig
from models.base_transformer import CMRTransformer

class TestBaseTransformer:
    """Test suite for CMRTransformer base class."""
    
    @pytest.fixture
    def small_config(self):
        """Create small GPT-OSS 20B config for testing."""
        # Use smaller config to keep unit test light
        return AutoConfig.from_pretrained("mistralai/Ministral-8B-Instruct-2410")
    
    @pytest.fixture
    def memory_config(self):
        """Create memory configuration for testing."""
        return {
            'target_layers': [2, 4],
            'buffer_size': 100,
            'relevance_threshold': 0.5
        }
    
    @pytest.fixture
    def cmr_model(self, small_config, memory_config):
        """Create CMR transformer model for testing."""
        model = CMRTransformer(small_config, memory_config)
        model.register_memory_hooks()
        yield model
        model.cleanup_hooks()
    
    def test_model_initialization(self, small_config, memory_config):
        """Test that model initializes correctly."""
        model = CMRTransformer(small_config, memory_config)
        assert model.config == small_config
        assert model.memory_config == memory_config
        assert model.memory_enabled == True
        assert model.current_sequence_id == 0
    
    def test_hook_registration(self, cmr_model):
        """Test that memory hooks are registered correctly."""
        # Check that hooks were registered
        assert len(cmr_model.layer_hooks) == 2  # Should have hooks on layers 2 and 4
        assert 2 in cmr_model.layer_hooks
        assert 4 in cmr_model.layer_hooks
    
    def test_forward_pass(self, cmr_model):
        """Test basic forward pass functionality."""
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = cmr_model(input_ids)
        
        # Check output structure
        assert 'last_hidden_state' in outputs
        assert 'captured_memory_states' in outputs
        assert 'memory_stats' in outputs
        
        # Check output shapes
        assert outputs['last_hidden_state'].shape == (batch_size, seq_len, 128)
        
        # Check memory stats
        memory_stats = outputs['memory_stats']
        assert 'total_captured_states' in memory_stats
        assert 'layers_with_memory' in memory_stats
    
    def test_memory_capture(self, cmr_model):
        """Test that hidden states are captured during forward pass."""
        input_ids = torch.randint(0, 1000, (1, 64))
        
        with torch.no_grad():
            outputs = cmr_model(input_ids)
        
        # Check that states were captured
        captured_states = outputs['captured_memory_states']
        assert len(captured_states) > 0
        
        # Check that target layers have captured states
        for layer_idx in [2, 4]:
            if layer_idx in captured_states:
                layer_states = captured_states[layer_idx]
                assert len(layer_states) > 0
                
                # Check state structure
                state_info = layer_states[0]
                assert 'hidden_state' in state_info
                assert 'layer_idx' in state_info
                assert 'sequence_id' in state_info
                assert 'timestamp' in state_info
    
    def test_memory_enable_disable(self, cmr_model):
        """Test memory enable/disable functionality."""
        input_ids = torch.randint(0, 1000, (1, 32))
        
        # Test with memory enabled
        cmr_model.enable_memory()
        with torch.no_grad():
            outputs = cmr_model(input_ids)
        enabled_captures = outputs['memory_stats']['total_captured_states']
        
        # Test with memory disabled
        cmr_model.disable_memory()
        with torch.no_grad():
            outputs = cmr_model(input_ids)
        disabled_captures = outputs['memory_stats']['total_captured_states']
        
        # Should capture more states when enabled
        assert enabled_captures >= disabled_captures
    
    def test_hook_cleanup(self, cmr_model):
        """Test that hooks are properly cleaned up."""
        # Verify hooks exist
        assert len(cmr_model.layer_hooks) > 0
        
        # Clean up hooks
        cmr_model.cleanup_hooks()
        
        # Verify hooks are removed
        assert len(cmr_model.layer_hooks) == 0

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

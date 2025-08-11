import torch
import pytest
from transformers import AutoModelForCausalLM, AutoConfig
from utils.hooks import HookManager
from models.base_transformer import CMRTransformer

class TestHookIntegration:
    """Test suite for hook system integration."""
    
    @pytest.fixture
    def llm_model(self):
        """Create small GPT-OSS 20B model for testing."""
        config = AutoConfig.from_pretrained("openai-community/gpt2")
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", config=config)
        # Add a helper method to get layers
        def get_layers():
            return model.transformer.h
        model.get_layers = get_layers
        return model
    
    @pytest.fixture
    def hook_manager(self):
        """Create hook manager instance."""
        return HookManager()
    
    def test_single_layer_hook(self, llm_model, hook_manager):
        """Test hooking a single transformer layer."""
        # Hook the middle layer
        target_layer = llm_model.get_layers()[3]
        hook_id = hook_manager.register_capture_hook(
            target_layer, "test_layer_3", layer_idx=3
        )
        
        # Forward pass
        input_ids = torch.randint(0, 1000, (2, 50))
        with torch.no_grad():
            outputs = llm_model(input_ids)
        
        # Verify capture
        captured = hook_manager.get_captured_data("test_layer_3")
        assert captured is not None
        assert captured['layer_idx'] == 3
        # GPT2 has 768 hidden dimensions
        assert captured['hidden_state'].shape == (2, 50, 768)
        
        hook_manager.remove_hooks()
    
    def test_multiple_layer_hooks(self, llm_model, hook_manager):
        """Test hooking multiple layers simultaneously."""
        target_layers = [1, 3, 5]
        hook_ids = hook_manager.register_layer_hooks(
            llm_model.get_layers(), target_layers, "multi_test"
        )
        
        # Forward pass
        input_ids = torch.randint(0, 1000, (1, 30))
        with torch.no_grad():
            outputs = llm_model(input_ids)
        
        # Verify all captures
        assert len(hook_ids) == 3
        for i, layer_idx in enumerate(target_layers):
            hook_id = f"multi_test_{layer_idx}"
            captured = hook_manager.get_captured_data(hook_id)
            assert captured is not None
            assert captured['layer_idx'] == layer_idx
            # GPT2 has 768 hidden dimensions
            assert captured['hidden_state'].shape == (1, 30, 768)
        
        hook_manager.remove_hooks()
    
    def test_cmr_transformer_integration(self):
        """Test hook integration with CMR transformer."""
        config = AutoConfig.from_pretrained("openai-community/gpt2")
        
        memory_config = {
            'target_layers': [2, 4],
            'buffer_size': 100
        }
        
        model = CMRTransformer(config, memory_config)
        model.register_memory_hooks()
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 40))
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Verify memory capture
        memory_stats = outputs['memory_stats']
        assert memory_stats['total_captured_states'] > 0
        assert 2 in memory_stats['layers_with_memory']
        assert 4 in memory_stats['layers_with_memory']
        
        model.cleanup_hooks()
    
    def test_hook_memory_usage(self, llm_model, hook_manager):
        """Test memory usage tracking."""
        # Register multiple hooks
        target_layers = [1, 2, 3]
        hook_ids = hook_manager.register_layer_hooks(
            llm_model.get_layers(), target_layers, "memory_test"
        )
        
        # Forward pass to capture data
        input_ids = torch.randint(0, 1000, (2, 25))
        with torch.no_grad():
            outputs = llm_model(input_ids)
        
        # Check memory usage
        memory_stats = hook_manager.get_memory_usage()
        assert memory_stats['total_captured_tensors'] == 3
        assert memory_stats['active_hooks'] == 3
        assert memory_stats['total_memory_bytes'] > 0
        assert memory_stats['total_memory_mb'] > 0
        
        hook_manager.remove_hooks()
    
    def test_custom_capture_function(self, llm_model, hook_manager):
        """Test custom capture function."""
        def custom_capture(module, input, output):
            # Custom logic: only capture if sequence length > 20
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output
            
            if hidden_state.shape[1] > 20:
                hook_manager.hook_data["custom_hook"] = {
                    'hidden_state': hidden_state.detach().clone(),
                    'layer_idx': 2,
                    'custom_flag': True
                }
        
        # Register custom hook
        target_layer = llm_model.get_layers()[2]
        hook_id = hook_manager.register_capture_hook(
            target_layer, "custom_hook", 
            capture_fn=custom_capture, layer_idx=2
        )
        
        # Test with short sequence (should not capture)
        short_input = torch.randint(0, 1000, (1, 15))
        with torch.no_grad():
            llm_model(short_input)
        
        # Should not have captured data
        captured = hook_manager.get_captured_data("custom_hook")
        assert captured is None
        
        # Test with long sequence (should capture)
        long_input = torch.randint(0, 1000, (1, 30))
        with torch.no_grad():
            llm_model(long_input)
        
        # Should have captured data
        captured = hook_manager.get_captured_data("custom_hook")
        assert captured is not None
        assert captured['custom_flag'] == True
        # GPT2 has 768 hidden dimensions
        assert captured['hidden_state'].shape == (1, 30, 768)
        
        hook_manager.remove_hooks()
    
    def test_hook_cleanup(self, llm_model, hook_manager):
        """Test proper hook cleanup."""
        # Register hooks
        target_layers = [1, 2]
        hook_ids = hook_manager.register_layer_hooks(
            llm_model.get_layers(), target_layers, "cleanup_test"
        )
        
        # Verify hooks are active
        assert len(hook_manager.hooks) > 0
        assert len(hook_manager.hook_configs) > 0
        
        # Cleanup specific hooks
        hook_manager.remove_hooks([hook_ids[0]])
        assert len(hook_manager.hook_configs) == 1
        
        # Cleanup all hooks
        hook_manager.remove_hooks()
        assert len(hook_manager.hooks) == 0
        assert len(hook_manager.hook_configs) == 0
    
    def test_data_clearing(self, llm_model, hook_manager):
        """Test captured data clearing."""
        # Register hook and capture data
        target_layer = llm_model.get_layers()[1]
        hook_id = hook_manager.register_capture_hook(
            target_layer, "clear_test", layer_idx=1
        )
        
        # Forward pass to capture data
        input_ids = torch.randint(0, 1000, (1, 20))
        with torch.no_grad():
            llm_model(input_ids)
        
        # Verify data is captured
        captured = hook_manager.get_captured_data("clear_test")
        assert captured is not None
        
        # Clear specific hook data
        hook_manager.clear_captured_data(["clear_test"])
        captured = hook_manager.get_captured_data("clear_test")
        assert captured is None
        
        # Clear all data
        hook_manager.clear_captured_data()
        all_data = hook_manager.get_all_captured_data()
        assert len(all_data) == 0
        
        hook_manager.remove_hooks()

## Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

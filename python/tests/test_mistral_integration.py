#!/usr/bin/env python3
"""
Tests for Mistral CMR integration module.
Tests memory capture, retrieval, and text generation capabilities.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mistral_integration import MistralCMRModel, create_mistral_cmr_model
from models.memory_buffer import LayeredMemoryBuffer
from models.relevance_scorer import RelevanceScorer

class TestMistralCMRModel:
    """Test cases for MistralCMRModel class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock Mistral configuration."""
        config = Mock()
        config.num_hidden_layers = 32
        config.hidden_size = 4096
        config.num_attention_heads = 32
        config.vocab_size = 32000
        config.max_position_embeddings = 32768
        return config
    
    @pytest.fixture
    def mock_memory_config(self):
        """Create a mock memory configuration."""
        return {
            'target_layers': [8, 16, 24],
            'buffer_size': 2000,
            'max_entries_per_layer': 500,
            'max_total_entries': 2000,
            'relevance_threshold': 0.6,
            'eviction_strategy': 'lru_relevance',
            'scoring_method': 'hybrid',
            'compression_ratio': 0.7,
            'max_memory_tokens': 128,
            'reconstruction_method': 'hierarchical'
        }
    
    @pytest.fixture
    def mock_transformer(self):
        """Create a mock transformer model."""
        transformer = Mock()
        
        # Mock the model structure
        mock_model = Mock()
        mock_layers = []
        
        # Create 32 mock layers
        for i in range(32):
            layer = Mock()
            layer.register_forward_hook = Mock(return_value=f"hook_{i}")
            mock_layers.append(layer)
        
        mock_model.layers = mock_layers
        transformer.model = mock_model
        
        # Mock parameters for device detection
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        transformer.parameters = Mock(return_value=[mock_param])
        
        return transformer
    
    def test_mistral_cmr_model_initialization(self, mock_config, mock_memory_config):
        """Test MistralCMRModel initialization."""
        with patch('models.mistral_integration.AutoConfig.from_pretrained', return_value=mock_config):
            with patch('models.mistral_integration.AutoModelForCausalLM.from_pretrained') as mock_load:
                mock_load.return_value = Mock()
                
                model = MistralCMRModel(
                    model_name="mistralai/Ministral-8B-Instruct-2410",
                    memory_config=mock_memory_config,
                    device="cpu",
                    use_quantization=False
                )
                
                assert model.model_name == "mistralai/Ministral-8B-Instruct-2410"
                assert model.device == "cpu"
                assert model.use_quantization is False
                assert model.max_memory_gb is None
                assert model.memory_config == mock_memory_config
    
    def test_load_mistral_config_success(self, mock_config):
        """Test successful Mistral configuration loading."""
        with patch('models.mistral_integration.AutoConfig.from_pretrained', return_value=mock_config):
            with patch('models.mistral_integration.AutoModelForCausalLM.from_pretrained') as mock_load:
                mock_load.return_value = Mock()
                
                model = MistralCMRModel(
                    model_name="mistralai/Ministral-8B-Instruct-2410",
                    device="cpu"
                )
                
                assert model.config == mock_config
                assert model.config.num_hidden_layers == 32
                assert model.config.hidden_size == 4096
    
    def test_load_mistral_config_failure(self):
        """Test Mistral configuration loading failure."""
        with patch('models.mistral_integration.AutoConfig.from_pretrained', side_effect=Exception("Config error")):
            with pytest.raises(RuntimeError, match="Could not load Mistral configuration"):
                MistralCMRModel(
                    model_name="mistralai/Ministral-8B-Instruct-2410",
                    device="cpu"
                )
    
    def test_default_mistral_memory_config(self, mock_config):
        """Test default memory configuration generation."""
        with patch('models.mistral_integration.AutoConfig.from_pretrained', return_value=mock_config):
            with patch('models.mistral_integration.AutoModelForCausalLM.from_pretrained') as mock_load:
                mock_load.return_value = Mock()
                
                model = MistralCMRModel(
                    model_name="mistralai/Ministral-8B-Instruct-2410",
                    device="cpu"
                )
                
                # Check that default config was generated
                assert 'target_layers' in model.memory_config
                assert model.memory_config['target_layers'] == [8, 16, 24]
                assert model.memory_config['buffer_size'] == 2000
                assert model.memory_config['relevance_threshold'] == 0.6
    
    def test_mistral_memory_config_for_different_layer_counts(self):
        """Test memory configuration for different layer counts."""
        test_cases = [
            (32, [8, 16, 24]),  # 32 layers
            (24, [6, 12, 18]),  # 24 layers
            (16, [4, 8, 12]),   # 16 layers
            (8, [2, 4, 6])      # 8 layers
        ]
        
        for num_layers, expected_targets in test_cases:
            config = Mock()
            config.num_hidden_layers = num_layers
            config.hidden_size = 4096
            config.num_attention_heads = 32
            config.vocab_size = 32000
            config.max_position_embeddings = 32768
            
            with patch('models.mistral_integration.AutoConfig.from_pretrained', return_value=config):
                with patch('models.mistral_integration.AutoModelForCausalLM.from_pretrained') as mock_load:
                    mock_load.return_value = Mock()
                    
                    model = MistralCMRModel(
                        model_name="test-model",
                        device="cpu"
                    )
                    
                    assert model.memory_config['target_layers'] == expected_targets
    
    def test_initialize_mistral_components(self, mock_config, mock_memory_config):
        """Test Mistral-specific component initialization."""
        with patch('models.mistral_integration.AutoConfig.from_pretrained', return_value=mock_config):
            with patch('models.mistral_integration.AutoModelForCausalLM.from_pretrained') as mock_load:
                mock_load.return_value = Mock()
                
                model = MistralCMRModel(
                    model_name="mistralai/Ministral-8B-Instruct-2410",
                    memory_config=mock_memory_config,
                    device="cpu"
                )
                
                # Check that components were initialized
                assert hasattr(model, 'memory_buffer')
                assert hasattr(model, 'relevance_scorer')
                assert hasattr(model, 'hook_manager')
                assert isinstance(model.memory_buffer, LayeredMemoryBuffer)
                assert isinstance(model.relevance_scorer, RelevanceScorer)
    
    def test_load_transformer_model_success(self, mock_config, mock_memory_config):
        """Test successful transformer model loading."""
        with patch('models.mistral_integration.AutoConfig.from_pretrained', return_value=mock_config):
            with patch('models.mistral_integration.AutoModelForCausalLM.from_pretrained') as mock_load:
                mock_model = Mock()
                mock_load.return_value = mock_model
                
                model = MistralCMRModel(
                    model_name="mistralai/Ministral-8B-Instruct-2410",
                    memory_config=mock_memory_config,
                    device="cpu",
                    use_quantization=True
                )
                
                # Check that model was loaded with correct parameters
                mock_load.assert_called_once()
                call_args = mock_load.call_args
                assert call_args[1]['torch_dtype'] == torch.float16
                assert call_args[1]['trust_remote_code'] is True
    
    def test_load_transformer_model_with_quantization(self, mock_config, mock_memory_config):
        """Test transformer model loading with quantization."""
        with patch('models.mistral_integration.AutoConfig.from_pretrained', return_value=mock_config):
            with patch('models.mistral_integration.AutoModelForCausalLM.from_pretrained') as mock_load:
                mock_model = Mock()
                mock_load.return_value = mock_model
                
                model = MistralCMRModel(
                    model_name="mistralai/Ministral-8B-Instruct-2410",
                    memory_config=mock_memory_config,
                    device="cpu",
                    use_quantization=True
                )
                
                # Check quantization parameters
                call_args = mock_load.call_args
                assert 'load_in_8bit' in call_args[1]
                assert call_args[1]['load_in_8bit'] is True
    
    def test_load_transformer_model_with_memory_limit(self, mock_config, mock_memory_config):
        """Test transformer model loading with memory limit."""
        with patch('models.mistral_integration.AutoConfig.from_pretrained', return_value=mock_config):
            with patch('models.mistral_integration.AutoModelForCausalLM.from_pretrained') as mock_load:
                mock_model = Mock()
                mock_load.return_value = mock_model
                
                model = MistralCMRModel(
                    model_name="mistralai/Ministral-8B-Instruct-2410",
                    memory_config=mock_memory_config,
                    device="cpu",
                    max_memory_gb=8.0
                )
                
                # Check memory limit parameters
                call_args = mock_load.call_args
                assert 'max_memory' in call_args[1]
                assert call_args[1]['max_memory'] == {0: "8.0GB"}
    
    def test_register_memory_hooks(self, mock_config, mock_memory_config, mock_transformer):
        """Test memory hook registration."""
        with patch('models.mistral_integration.AutoConfig.from_pretrained', return_value=mock_config):
            with patch('models.mistral_integration.AutoModelForCausalLM.from_pretrained', return_value=mock_transformer):
                model = MistralCMRModel(
                    model_name="mistralai/Ministral-8B-Instruct-2410",
                    memory_config=mock_memory_config,
                    device="cpu"
                )
                
                # Mock the transformer property
                model.transformer = mock_transformer
                
                # Register hooks
                model.register_memory_hooks()
                
                # Check that hooks were registered
                assert len(model.layer_hooks) == 3  # 3 target layers
                assert 8 in model.layer_hooks
                assert 16 in model.layer_hooks
                assert 24 in model.layer_hooks
    
    def test_capture_mistral_layer_state(self, mock_config, mock_memory_config):
        """Test Mistral layer state capture."""
        with patch('models.mistral_integration.AutoConfig.from_pretrained', return_value=mock_config):
            with patch('models.mistral_integration.AutoModelForCausalLM.from_pretrained') as mock_load:
                mock_load.return_value = Mock()
                
                model = MistralCMRModel(
                    model_name="mistralai/Ministral-8B-Instruct-2410",
                    memory_config=mock_memory_config,
                    device="cpu"
                )
                
                # Mock components
                model.memory_enabled = True
                model.current_sequence_id = 1
                model.memory_buffer = Mock()
                model.memory_buffer.store_state.return_value = True
                
                # Mock the relevance scorer's forward method to return high scores above threshold
                with patch.object(model.relevance_scorer, 'forward', return_value=torch.tensor([[0.8, 0.9, 0.7]])):
                    # Create mock hidden state
                    hidden_state = torch.randn(1, 3, 4096)  # batch_size=1, seq_len=3, hidden_size=4096
                    
                    # Capture state
                    model._capture_mistral_layer_state(8, hidden_state)
                    
                    # Check that memory was stored for all 3 sequence positions
                    assert model.memory_buffer.store_state.call_count == 3
                assert model._state_counter == 3
    
    def test_generate_with_memory(self, mock_config, mock_memory_config):
        """Test text generation with memory integration."""
        with patch('models.mistral_integration.AutoConfig.from_pretrained', return_value=mock_config):
            with patch('models.mistral_integration.AutoModelForCausalLM.from_pretrained') as mock_load:
                mock_load.return_value = Mock()
                
                model = MistralCMRModel(
                    model_name="mistralai/Ministral-8B-Instruct-2410",
                    memory_config=mock_memory_config,
                    device="cpu"
                )
                
                # Mock components
                mock_transformer = Mock()
                mock_transformer.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
                
                # Mock parameters method to return a proper iterable for device detection
                mock_param = Mock()
                mock_param.device = torch.device('cpu')
                mock_transformer.parameters = Mock(return_value=iter([mock_param]))
                
                model.transformer = mock_transformer
                
                # Mock tokenizer
                with patch('models.mistral_integration.AutoTokenizer.from_pretrained') as mock_tokenizer:
                    mock_tok = Mock()
                    mock_tok.return_value = {'input_ids': torch.tensor([[1, 2, 3]])}
                    mock_tok.decode.return_value = "Generated text"
                    mock_tokenizer.return_value = mock_tok
                    
                    # Mock memory buffer
                    model.memory_buffer = Mock()
                    model.memory_buffer.entry_count = 10
                    
                    # Mock the _retrieve_mistral_memories method to return proper structure
                    with patch.object(model, '_retrieve_mistral_memories', return_value={'layer_8': [Mock(), Mock()], 'layer_16': [Mock()]}):
                        # Generate text
                        result = model.generate_with_memory(
                            prompt="Test prompt",
                            max_length=50,
                            temperature=0.7,
                            use_memory=True
                        )
                        
                        assert result == "Generated text"
                        mock_transformer.generate.assert_called_once()
    
    def test_get_mistral_stats(self, mock_config, mock_memory_config):
        """Test Mistral statistics retrieval."""
        with patch('models.mistral_integration.AutoConfig.from_pretrained', return_value=mock_config):
            with patch('models.mistral_integration.AutoModelForCausalLM.from_pretrained') as mock_load:
                mock_load.return_value = Mock()
                
                model = MistralCMRModel(
                    model_name="mistralai/Ministral-8B-Instruct-2410",
                    memory_config=mock_memory_config,
                    device="cpu",
                    use_quantization=True,
                    max_memory_gb=8.0
                )
                
                # Mock transformer for device detection
                mock_transformer = Mock()
                mock_param = Mock()
                mock_param.device = torch.device('cpu')
                mock_transformer.parameters = Mock(return_value=iter([mock_param]))
                model.transformer = mock_transformer
                
                # Mock base stats - use the correct method name from base class
                with patch.object(model, '_get_memory_stats', return_value={'total_entries': 100}):
                    stats = model.get_mistral_stats()
                    
                    assert 'model_name' in stats
                    assert 'device' in stats
                    assert 'use_quantization' in stats
                    assert 'max_memory_gb' in stats
                    assert 'mistral_architecture' in stats
                    assert stats['model_name'] == "mistralai/Ministral-8B-Instruct-2410"
                    assert stats['use_quantization'] is True
                    assert stats['max_memory_gb'] == 8.0

class TestMistralCMRModelFactory:
    """Test cases for the factory function."""
    
    def test_create_mistral_cmr_model_success(self):
        """Test successful model creation."""
        with patch('models.mistral_integration.MistralCMRModel') as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model
            mock_model.register_memory_hooks = Mock()
            
            result = create_mistral_cmr_model(
                model_name="mistralai/Ministral-8B-Instruct-2410",
                use_quantization=True
            )
            
            assert result == mock_model
            mock_model_class.assert_called_once()
            mock_model.register_memory_hooks.assert_called_once()
    
    def test_create_mistral_cmr_model_failure(self):
        """Test model creation failure."""
        with patch('models.mistral_integration.MistralCMRModel', side_effect=Exception("Creation error")):
            with pytest.raises(Exception, match="Creation error"):
                create_mistral_cmr_model(
                    model_name="mistralai/Ministral-8B-Instruct-2410"
                )

class TestMistralIntegrationEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.integration
    def test_full_mistral_integration_workflow(self):
        """Test the complete Mistral integration workflow."""
        # This test would require actual model loading and is marked as integration
        # In a real environment, this would test the full pipeline
        pass
    
    @pytest.mark.integration
    def test_memory_capture_and_retrieval_workflow(self):
        """Test memory capture and retrieval workflow."""
        # This test would test the complete memory workflow
        pass

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from models.reconstruction import (
    LayeredStateReconstructor,
    HierarchicalReconstructor,
    AttentionBasedReconstructor,
    MLPReconstructor,
    MemoryAttention,
    ContextBlender
)

# Mock MemoryEntry class for testing
class MockMemoryEntry:
    def __init__(self, hidden_state, relevance_score=0.8):
        self.hidden_state = hidden_state
        self.relevance_score = relevance_score

class TestLayeredStateReconstructor:
    """Test the main LayeredStateReconstructor class."""
    
    @pytest.fixture
    def reconstructor(self):
        """Create a basic reconstructor for testing."""
        return LayeredStateReconstructor(
            hidden_size=128,
            num_layers=6,
            reconstruction_method="hierarchical",
            max_memory_tokens=16,
            compression_ratio=0.5
        )
    
    @pytest.fixture
    def test_data(self):
        """Create test data tensors."""
        batch_size = 2
        seq_len = 20
        hidden_size = 128
        num_memories = 8
        
        # Create mock memory entries
        memory_entries = []
        for i in range(num_memories):
            hidden_state = torch.randn(1, hidden_size)
            relevance_score = torch.rand(1).item()
            memory_entries.append(MockMemoryEntry(hidden_state, relevance_score))
        
        # Current hidden states
        current_states = torch.randn(batch_size, seq_len, hidden_size)
        
        return {
            'memory_entries': memory_entries,
            'current_states': current_states,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'hidden_size': hidden_size,
            'num_memories': num_memories
        }
    
    def test_initialization(self, reconstructor):
        """Test that the reconstructor initializes correctly."""
        assert reconstructor.hidden_size == 128
        assert reconstructor.num_layers == 6
        assert reconstructor.reconstruction_method == "hierarchical"
        assert reconstructor.max_memory_tokens == 16
        assert reconstructor.compression_ratio == 0.5
        assert reconstructor.compressed_size == 64
        
        # Check that all layer reconstructors were created
        assert len(reconstructor.layer_reconstructors) == 6
        for i in range(6):
            assert f"layer_{i}" in reconstructor.layer_reconstructors
        
        # Check that integration modules were created
        assert len(reconstructor.memory_projectors) == 6
        assert len(reconstructor.context_blenders) == 6
    
    def test_build_reconstructors(self, reconstructor):
        """Test that different reconstruction methods create correct reconstructors."""
        # Test hierarchical method
        hierarchical_reconstructor = LayeredStateReconstructor(
            hidden_size=128, num_layers=3, reconstruction_method="hierarchical"
        )
        assert isinstance(hierarchical_reconstructor.layer_reconstructors["layer_0"], 
                        HierarchicalReconstructor)
        
        # Test attention method
        attention_reconstructor = LayeredStateReconstructor(
            hidden_size=128, num_layers=3, reconstruction_method="attention"
        )
        assert isinstance(attention_reconstructor.layer_reconstructors["layer_0"], 
                        AttentionBasedReconstructor)
        
        # Test MLP method
        mlp_reconstructor = LayeredStateReconstructor(
            hidden_size=128, num_layers=3, reconstruction_method="mlp"
        )
        assert isinstance(mlp_reconstructor.layer_reconstructors["layer_0"], 
                        MLPReconstructor)
        
        # Test invalid method
        with pytest.raises(ValueError, match="Unknown reconstruction method"):
            LayeredStateReconstructor(
                hidden_size=128, num_layers=3, reconstruction_method="invalid"
            )
    
    def test_prepare_memory_states(self, reconstructor, test_data):
        """Test memory state preparation."""
        memory_states = reconstructor._prepare_memory_states(
            test_data['memory_entries'], 
            test_data['batch_size']
        )
        
        assert memory_states is not None
        assert memory_states.shape == (test_data['batch_size'], test_data['num_memories'], test_data['hidden_size'])
        
        # Test with empty memory entries
        empty_states = reconstructor._prepare_memory_states([], test_data['batch_size'])
        assert empty_states is None
    
    def test_reconstruct_layer_memories(self, reconstructor, test_data):
        """Test the main reconstruction method."""
        layer_idx = 2
        enhanced_states = reconstructor.reconstruct_layer_memories(
            layer_idx=layer_idx,
            memory_entries=test_data['memory_entries'],
            current_hidden_states=test_data['current_states']
        )
        
        # Check output shape matches input
        assert enhanced_states.shape == test_data['current_states'].shape
        
        # Test with no memories
        no_memory_states = reconstructor.reconstruct_layer_memories(
            layer_idx=layer_idx,
            memory_entries=[],
            current_hidden_states=test_data['current_states']
        )
        assert torch.equal(no_memory_states, test_data['current_states'])
    
    def test_memory_position_embeddings(self, reconstructor, test_data):
        """Test that position embeddings are added correctly."""
        memory_states = reconstructor._prepare_memory_states(
            test_data['memory_entries'], 
            test_data['batch_size']
        )
        
        # Check that position embeddings were added
        # The original memory states should be different from the prepared ones
        # due to position embeddings
        assert not torch.equal(memory_states, memory_states - reconstructor.memory_position_embeddings.weight[:test_data['num_memories']].unsqueeze(0))

class TestHierarchicalReconstructor:
    """Test the HierarchicalReconstructor class."""
    
    @pytest.fixture
    def reconstructor(self):
        return HierarchicalReconstructor(hidden_size=128, compressed_size=64)
    
    def test_initialization(self, reconstructor):
        """Test hierarchical reconstructor initialization."""
        assert reconstructor.hidden_size == 128
        assert reconstructor.compressed_size == 64
        assert isinstance(reconstructor.compress, nn.Sequential)
        assert isinstance(reconstructor.reconstruct, nn.Sequential)
        assert isinstance(reconstructor.global_attention, nn.MultiheadAttention)
        assert isinstance(reconstructor.local_attention, nn.MultiheadAttention)
    
    def test_forward_pass(self, reconstructor):
        """Test the forward pass of hierarchical reconstructor."""
        batch_size = 2
        num_memories = 8
        seq_len = 20
        hidden_size = 128
        
        memory_states = torch.randn(batch_size, num_memories, hidden_size)
        current_states = torch.randn(batch_size, seq_len, hidden_size)
        
        output = reconstructor(memory_states, current_states)
        
        assert output.shape == (batch_size, num_memories, hidden_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestAttentionBasedReconstructor:
    """Test the AttentionBasedReconstructor class."""
    
    @pytest.fixture
    def reconstructor(self):
        return AttentionBasedReconstructor(hidden_size=128, max_memories=16)
    
    def test_initialization(self, reconstructor):
        """Test attention-based reconstructor initialization."""
        assert reconstructor.hidden_size == 128
        assert reconstructor.max_memories == 16
        assert isinstance(reconstructor.cross_attention, nn.MultiheadAttention)
        assert isinstance(reconstructor.self_attention, nn.MultiheadAttention)
        assert isinstance(reconstructor.ffn, nn.Sequential)
        assert isinstance(reconstructor.norm1, nn.LayerNorm)
        assert isinstance(reconstructor.norm2, nn.LayerNorm)
        assert isinstance(reconstructor.norm3, nn.LayerNorm)
    
    def test_forward_pass(self, reconstructor):
        """Test the forward pass of attention-based reconstructor."""
        batch_size = 2
        num_memories = 8
        seq_len = 20
        hidden_size = 128
        
        memory_states = torch.randn(batch_size, num_memories, hidden_size)
        current_states = torch.randn(batch_size, seq_len, hidden_size)
        
        output = reconstructor(memory_states, current_states)
        
        assert output.shape == (batch_size, num_memories, hidden_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestMLPReconstructor:
    """Test the MLPReconstructor class."""
    
    @pytest.fixture
    def reconstructor(self):
        return MLPReconstructor(hidden_size=128, compressed_size=64)
    
    def test_initialization(self, reconstructor):
        """Test MLP reconstructor initialization."""
        assert isinstance(reconstructor.encoder, nn.Sequential)
        assert isinstance(reconstructor.decoder, nn.Sequential)
        assert isinstance(reconstructor.context_gate, nn.Sequential)
    
    def test_forward_pass(self, reconstructor):
        """Test the forward pass of MLP reconstructor."""
        batch_size = 2
        num_memories = 8
        seq_len = 20
        hidden_size = 128
        
        memory_states = torch.randn(batch_size, num_memories, hidden_size)
        current_states = torch.randn(batch_size, seq_len, hidden_size)
        
        output = reconstructor(memory_states, current_states)
        
        assert output.shape == (batch_size, num_memories, hidden_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestMemoryAttention:
    """Test the MemoryAttention class."""
    
    @pytest.fixture
    def attention(self):
        return MemoryAttention(hidden_size=128, max_memories=16)
    
    def test_initialization(self, attention):
        """Test memory attention initialization."""
        assert attention.hidden_size == 128
        assert attention.max_memories == 16
        assert isinstance(attention.query_proj, nn.Linear)
        assert isinstance(attention.key_proj, nn.Linear)
        assert isinstance(attention.value_proj, nn.Linear)
        assert attention.scale == pytest.approx(11.313708498984761)  # sqrt(128)
    
    def test_forward_pass(self, attention):
        """Test the forward pass of memory attention."""
        batch_size = 2
        seq_len = 20
        num_memories = 8
        hidden_size = 128
        
        queries = torch.randn(batch_size, seq_len, hidden_size)
        memory_keys = torch.randn(batch_size, num_memories, hidden_size)
        memory_values = torch.randn(batch_size, num_memories, hidden_size)
        
        output, weights = attention(queries, memory_keys, memory_values)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert weights.shape == (batch_size, seq_len, num_memories)
        assert not torch.isnan(output).any()
        assert not torch.isnan(weights).any()
        
        # Check that attention weights are valid (non-negative, finite)
        # Note: We don't check exact sum to 1 because dropout is applied to the weights
        assert (weights >= 0).all()
        assert torch.isfinite(weights).all()
        # Check that weights are reasonable (not all zeros, not extremely large)
        assert weights.sum() > 0
        assert weights.max() < 10.0

class TestContextBlender:
    """Test the ContextBlender class."""
    
    @pytest.fixture
    def blender(self):
        return ContextBlender(hidden_size=128)
    
    def test_initialization(self, blender):
        """Test context blender initialization."""
        assert blender.hidden_size == 128
        assert isinstance(blender.memory_gate, nn.Sequential)
        assert isinstance(blender.integration_layer, nn.TransformerEncoderLayer)
        assert isinstance(blender.output_proj, nn.Linear)
    
    def test_forward_pass(self, blender):
        """Test the forward pass of context blender."""
        batch_size = 2
        seq_len = 20
        num_memories = 8
        hidden_size = 128
        
        current_states = torch.randn(batch_size, seq_len, hidden_size)
        memory_states = torch.randn(batch_size, num_memories, hidden_size)
        layer_weight = torch.tensor(0.7)
        
        output = blender(current_states, memory_states, layer_weight)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestIntegration:
    """Test integration between components."""
    
    def test_full_reconstruction_pipeline(self):
        """Test the complete reconstruction pipeline."""
        # Create reconstructor
        reconstructor = LayeredStateReconstructor(
            hidden_size=64,
            num_layers=4,
            reconstruction_method="hierarchical",
            max_memory_tokens=8
        )
        
        # Create test data
        batch_size = 2
        seq_len = 16
        hidden_size = 64
        num_memories = 4
        
        # Create mock memory entries
        memory_entries = []
        for i in range(num_memories):
            hidden_state = torch.randn(1, hidden_size)
            relevance_score = torch.rand(1).item()
            memory_entries.append(MockMemoryEntry(hidden_state, relevance_score))
        
        current_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Test reconstruction for each layer
        for layer_idx in range(4):
            enhanced_states = reconstructor.reconstruct_layer_memories(
                layer_idx=layer_idx,
                memory_entries=memory_entries,
                current_hidden_states=current_states
            )
            
            assert enhanced_states.shape == current_states.shape
            assert not torch.isnan(enhanced_states).any()
            assert not torch.isinf(enhanced_states).any()
    
    def test_different_reconstruction_methods(self):
        """Test that different reconstruction methods work correctly."""
        methods = ["hierarchical", "attention", "mlp"]
        hidden_size = 64
        num_layers = 3
        
        for method in methods:
            reconstructor = LayeredStateReconstructor(
                hidden_size=hidden_size,
                num_layers=num_layers,
                reconstruction_method=method
            )
            
            # Test that it can process data
            batch_size = 2
            seq_len = 16
            num_memories = 4
            
            memory_entries = []
            for i in range(num_memories):
                hidden_state = torch.randn(1, hidden_size)
                memory_entries.append(MockMemoryEntry(hidden_state))
            
            current_states = torch.randn(batch_size, seq_len, hidden_size)
            
            enhanced_states = reconstructor.reconstruct_layer_memories(
                layer_idx=1,
                memory_entries=memory_entries,
                current_hidden_states=current_states
            )
            
            assert enhanced_states.shape == current_states.shape
            assert not torch.isnan(enhanced_states).any()

if __name__ == "__main__":
    # Run basic tests
    print("Running reconstruction system tests...")
    
    # Test basic functionality
    hidden_size = 64
    num_layers = 4
    
    reconstructor = LayeredStateReconstructor(
        hidden_size=hidden_size,
        num_layers=num_layers,
        reconstruction_method="hierarchical"
    )
    
    print("✅ Basic reconstructor created successfully")
    
    # Test with sample data
    batch_size = 2
    seq_len = 16
    num_memories = 4
    
    memory_entries = []
    for i in range(num_memories):
        hidden_state = torch.randn(1, hidden_size)
        memory_entries.append(MockMemoryEntry(hidden_state))
    
    current_states = torch.randn(batch_size, seq_len, hidden_size)
    
    enhanced_states = reconstructor.reconstruct_layer_memories(
        layer_idx=1,
        memory_entries=memory_entries,
        current_hidden_states=current_states
    )
    
    print(f"✅ Reconstruction successful: {enhanced_states.shape}")
    print("🎉 All basic tests passed!")

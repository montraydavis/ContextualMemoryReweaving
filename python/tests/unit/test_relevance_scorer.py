import pytest
import torch
import torch.nn as nn
from models.relevance_scorer import RelevanceScorer

class TestRelevanceScorer:
    """Test suite for RelevanceScorer class."""
    

    
    def test_initialization(self, hidden_size):
        """Test RelevanceScorer initialization with different methods."""
        # Test attention-based initialization
        scorer = RelevanceScorer(hidden_size, "attention_based")
        assert scorer.scoring_method == "attention_based"
        assert hasattr(scorer, 'score_head')
        assert scorer.score_head.in_features == hidden_size
        assert scorer.score_head.out_features == 1
        
        # Test variance-based initialization
        scorer = RelevanceScorer(hidden_size, "variance_based")
        assert scorer.scoring_method == "variance_based"
        assert not hasattr(scorer, 'score_head')
        
        # Test hybrid initialization
        scorer = RelevanceScorer(hidden_size, "hybrid")
        assert scorer.scoring_method == "hybrid"
        assert hasattr(scorer, 'attention_head')
        assert hasattr(scorer, 'variance_weight')
        
        # Test invalid method
        with pytest.raises(ValueError, match="Unknown scoring method"):
            RelevanceScorer(hidden_size, "invalid_method")
    
    def test_attention_based_scoring(self, hidden_size, hidden_states, attention_mask):
        """Test attention-based scoring method."""
        scorer = RelevanceScorer(hidden_size, "attention_based")
        
        # Test forward pass
        scores = scorer(hidden_states, attention_mask)
        assert scores.shape == (hidden_states.shape[0], hidden_states.shape[1])
        assert torch.all(scores >= 0) and torch.all(scores <= 1)
        
        # Test that masked positions have zero scores
        assert torch.all(scores[0, 45:] == 0)
        
        # Test that scores sum to 1 per sequence (softmax)
        assert torch.allclose(scores.sum(dim=1), torch.ones(scores.shape[0]), atol=1e-6)
        
        # Test without mask
        scores_no_mask = scorer(hidden_states)
        assert scores_no_mask.shape == hidden_states.shape[:2]
        assert torch.allclose(scores_no_mask.sum(dim=1), torch.ones(scores_no_mask.shape[0]), atol=1e-6)
    
    def test_variance_based_scoring(self, hidden_size, hidden_states, attention_mask):
        """Test variance-based scoring method."""
        scorer = RelevanceScorer(hidden_size, "variance_based")
        
        # Test forward pass
        scores = scorer(hidden_states, attention_mask)
        assert scores.shape == (hidden_states.shape[0], hidden_states.shape[1])
        assert torch.all(scores >= 0) and torch.all(scores <= 1)
        
        # Test that masked positions have zero scores
        assert torch.all(scores[0, 45:] == 0)
        
        # Test that non-masked positions have non-negative scores
        # Note: variance-based scoring can produce zero scores for positions with low variance
        assert torch.all(scores[0, :45] >= 0)

        # Test without mask
        scores_no_mask = scorer(hidden_states)
        assert scores_no_mask.shape == hidden_states.shape[:2]
        assert torch.all(scores_no_mask >= 0)
    
    def test_hybrid_scoring(self, hidden_size, hidden_states, attention_mask):
        """Test hybrid scoring method."""
        scorer = RelevanceScorer(hidden_size, "hybrid")
        
        # Test forward pass
        scores = scorer(hidden_states, attention_mask)
        assert scores.shape == (hidden_states.shape[0], hidden_states.shape[1])
        assert torch.all(scores >= 0) and torch.all(scores <= 1)
        
        # Test that masked positions have zero scores
        assert torch.all(scores[0, 45:] == 0)
        
        # Test without mask
        scores_no_mask = scorer(hidden_states)
        assert scores_no_mask.shape == hidden_states.shape[:2]
        assert torch.all(scores_no_mask > 0)
    
    def test_input_validation(self, hidden_size):
        """Test input validation in forward method."""
        scorer = RelevanceScorer(hidden_size, "attention_based")
        
        # Test invalid tensor type
        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            scorer("not_a_tensor")
        
        # Test wrong tensor dimensions
        with pytest.raises(ValueError, match="must be 3D tensor"):
            scorer(torch.randn(10, 20))
        
        # Test wrong hidden size
        with pytest.raises(ValueError, match=f"last dimension must be {hidden_size}"):
            scorer(torch.randn(2, 10, hidden_size + 1))
        
        # Test incompatible attention mask
        with pytest.raises(ValueError, match="incompatible"):
            scorer(torch.randn(2, 10, hidden_size), torch.randn(3, 10))
    
    def test_get_top_k_positions(self, hidden_size, hidden_states, attention_mask):
        """Test top-k position selection."""
        scorer = RelevanceScorer(hidden_size, "attention_based")
        scores = scorer(hidden_states, attention_mask)
        
        # Test top-5 selection
        top_positions = scorer.get_top_k_positions(scores, k=5, attention_mask=attention_mask)
        assert len(top_positions) == 5
        assert all(isinstance(pos, tuple) and len(pos) == 2 for pos in top_positions)
        
        # Test that all positions are valid
        for batch_idx, seq_idx in top_positions:
            assert 0 <= batch_idx < hidden_states.shape[0]
            assert 0 <= seq_idx < hidden_states.shape[1]
            # Check that masked positions are not included
            if batch_idx == 0 and seq_idx >= 45:
                pytest.fail(f"Masked position included: ({batch_idx}, {seq_idx})")
        
        # Test without mask
        top_positions_no_mask = scorer.get_top_k_positions(scores, k=3)
        assert len(top_positions_no_mask) == 3
        
        # Test edge cases
        assert scorer.get_top_k_positions(scores, k=0) == []
        assert len(scorer.get_top_k_positions(scores, k=1000)) <= scores.numel()
    
    def test_get_top_k_validation(self, hidden_size, hidden_states):
        """Test validation in get_top_k_positions."""
        scorer = RelevanceScorer(hidden_size, "attention_based")
        scores = scorer(hidden_states)
        
        # Test invalid k (negative values)
        with pytest.raises(ValueError, match="must be a non-negative integer"):
            scorer.get_top_k_positions(scores, k=-1)
        
        # Test invalid scores tensor
        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            scorer.get_top_k_positions("not_tensor", k=5)
        with pytest.raises(ValueError, match="must be 2D tensor"):
            scorer.get_top_k_positions(torch.randn(10, 20, 30), k=5)
    
    def test_get_scoring_stats(self, hidden_size, hidden_states, attention_mask):
        """Test scoring statistics computation."""
        scorer = RelevanceScorer(hidden_size, "attention_based")
        scores = scorer(hidden_states, attention_mask)
        
        # Test with mask
        stats = scorer.get_scoring_stats(scores, attention_mask)
        assert 'mean_score' in stats
        assert 'std_score' in stats
        assert 'min_score' in stats
        assert 'max_score' in stats
        assert 'total_positions' in stats
        
        # Test that stats are reasonable
        assert 0 <= stats['min_score'] <= stats['max_score'] <= 1
        assert stats['total_positions'] > 0
        
        # Test without mask
        stats_no_mask = scorer.get_scoring_stats(scores)
        assert stats_no_mask['total_positions'] == scores.numel()
    
    def test_update_scoring_method(self, hidden_size):
        """Test dynamic scoring method updates."""
        scorer = RelevanceScorer(hidden_size, "attention_based")
        assert scorer.scoring_method == "attention_based"
        assert hasattr(scorer, 'score_head')
        
        # Update to variance-based
        scorer.update_scoring_method("variance_based")
        assert scorer.scoring_method == "variance_based"
        assert not hasattr(scorer, 'score_head')
        
        # Update to hybrid
        scorer.update_scoring_method("hybrid")
        assert scorer.scoring_method == "hybrid"
        assert hasattr(scorer, 'attention_head')
        assert hasattr(scorer, 'variance_weight')
        
        # Update back to attention-based
        scorer.update_scoring_method("attention_based")
        assert scorer.scoring_method == "attention_based"
        assert hasattr(scorer, 'score_head')
        
        # Test invalid method
        with pytest.raises(ValueError, match="Unknown scoring method"):
            scorer.update_scoring_method("invalid")
    
    def test_edge_cases(self, hidden_size):
        """Test edge cases and boundary conditions."""
        scorer = RelevanceScorer(hidden_size, "variance_based")
        
        # Test with very small hidden states
        small_states = torch.randn(1, 1, hidden_size) * 1e-6
        scores = scorer(small_states)
        assert scores.shape == (1, 1)
        assert torch.all(scores >= 0) and torch.all(scores <= 1)
        
        # Test with all-zero hidden states
        zero_states = torch.zeros(2, 3, hidden_size)
        scores = scorer(zero_states)
        assert scores.shape == (2, 3)
        # Should assign uniform scores when all variances are zero
        assert torch.allclose(scores[0], scores[0][0])
        assert torch.allclose(scores[1], scores[1][0])
    
    def test_device_handling(self, hidden_size):
        """Test that tensors work on different devices if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        scorer = RelevanceScorer(hidden_size, "attention_based")
        device = torch.device("cuda")
        
        # Move scorer to GPU
        scorer = scorer.to(device)
        
        # Create tensors on GPU
        hidden_states = torch.randn(2, 10, hidden_size, device=device)
        attention_mask = torch.ones(2, 10, device=device)
        
        # Test forward pass
        scores = scorer(hidden_states, attention_mask)
        assert scores.device == device
        
        # Test top-k selection
        top_positions = scorer.get_top_k_positions(scores, k=3)
        assert len(top_positions) == 3
    
    def test_gradient_flow(self, hidden_size):
        """Test that gradients flow properly through learnable parameters."""
        scorer = RelevanceScorer(hidden_size, "hybrid")
        hidden_states = torch.randn(2, 10, hidden_size, requires_grad=True)
        
        # Forward pass
        scores = scorer(hidden_states, attention_mask=None)
        
        # Compute loss (dummy loss for testing)
        loss = scores.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert hidden_states.grad is not None
        assert scorer.attention_head.weight.grad is not None
        assert scorer.attention_head.bias.grad is not None
        assert scorer.variance_weight.grad is not None
        
        # Check gradient shapes
        assert scorer.attention_head.weight.grad.shape == scorer.attention_head.weight.shape
        assert scorer.attention_head.bias.grad.shape == scorer.attention_head.bias.shape
        assert scorer.variance_weight.grad.shape == scorer.variance_weight.shape

## Example usage and testing
if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3
"""
Demonstration script for RelevanceScorer functionality.
Shows different scoring methods, visualizations, and integration examples.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from models.relevance_scorer import RelevanceScorer
from utils.hooks import HookManager
from models.base_transformer import CMRTransformer
from transformers import AutoConfig
import time

def create_sample_data(batch_size=2, seq_len=50, hidden_size=768):
    """Create sample hidden states for demonstration."""
    # Generate realistic hidden states
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Create attention mask (simulate padding)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[0, 45:] = 0  # Mask last 5 tokens of first sequence
    attention_mask[1, 40:] = 0  # Mask last 10 tokens of second sequence
    
    return hidden_states, attention_mask

def demonstrate_scoring_methods(hidden_states, attention_mask):
    """Demonstrate all three scoring methods."""
    print("=" * 60)
    print("RELEVANCE SCORING METHODS DEMONSTRATION")
    print("=" * 60)
    
    hidden_size = hidden_states.size(-1)
    
    # Test attention-based scoring
    print("\n1. ATTENTION-BASED SCORING")
    print("-" * 30)
    attention_scorer = RelevanceScorer(hidden_size, "attention_based")
    
    start_time = time.time()
    attention_scores = attention_scorer(hidden_states, attention_mask)
    attention_time = time.time() - start_time
    
    print(f"Shape: {attention_scores.shape}")
    print(f"Computation time: {attention_time:.4f}s")
    print(f"Score range: [{attention_scores.min():.4f}, {attention_scores.max():.4f}]")
    print(f"Sum per sequence: {attention_scores.sum(dim=1).tolist()}")
    
    # Test variance-based scoring
    print("\n2. VARIANCE-BASED SCORING")
    print("-" * 30)
    variance_scorer = RelevanceScorer(hidden_size, "variance_based")
    
    start_time = time.time()
    variance_scores = variance_scorer(hidden_states, attention_mask)
    variance_time = time.time() - start_time
    
    print(f"Shape: {variance_scores.shape}")
    print(f"Computation time: {variance_time:.4f}s")
    print(f"Score range: [{variance_scores.min():.4f}, {variance_scores.max():.4f}]")
    
    # Test hybrid scoring
    print("\n3. HYBRID SCORING")
    print("-" * 30)
    hybrid_scorer = RelevanceScorer(hidden_size, "hybrid")
    
    start_time = time.time()
    hybrid_scores = hybrid_scorer(hidden_states, attention_mask)
    hybrid_time = time.time() - start_time
    
    print(f"Shape: {hybrid_scores.shape}")
    print(f"Computation time: {hybrid_time:.4f}s")
    print(f"Score range: [{hybrid_scores.min():.4f}, {hybrid_scores.max():.4f}]")
    print(f"Variance weight: {torch.sigmoid(hybrid_scorer.variance_weight).item():.4f}")
    
    return {
        'attention': (attention_scores, attention_time),
        'variance': (variance_scores, variance_time),
        'hybrid': (hybrid_scores, hybrid_time)
    }

def demonstrate_top_k_selection(scores_dict, attention_mask):
    """Demonstrate top-k position selection."""
    print("\n" + "=" * 60)
    print("TOP-K POSITION SELECTION")
    print("=" * 60)
    
    for method_name, (scores, _) in scores_dict.items():
        print(f"\n{method_name.upper()} SCORING - Top 10 positions:")
        print("-" * 40)
        
        # Get top 10 positions
        top_positions = scores_dict[method_name][0].get_top_k_positions(scores, k=10, attention_mask=attention_mask)
        
        print(f"Top 10 positions (batch_idx, seq_idx):")
        for i, (batch_idx, seq_idx) in enumerate(top_positions):
            score = scores[batch_idx, seq_idx].item()
            print(f"  {i+1:2d}. ({batch_idx:2d}, {seq_idx:2d}) -> Score: {score:.4f}")
        
        # Verify no masked positions are included
        masked_positions = []
        for batch_idx, seq_idx in top_positions:
            if attention_mask[batch_idx, seq_idx] == 0:
                masked_positions.append((batch_idx, seq_idx))
        
        if masked_positions:
            print(f"  WARNING: {len(masked_positions)} masked positions included!")
        else:
            print("  ✓ All positions are valid (no masked positions)")

def demonstrate_scoring_stats(scores_dict, attention_mask):
    """Demonstrate scoring statistics."""
    print("\n" + "=" * 60)
    print("SCORING STATISTICS")
    print("=" * 60)
    
    # Use attention-based scorer for stats (any scorer would work)
    scorer = RelevanceScorer(scores_dict['attention'][0].size(-1), "attention_based")
    
    for method_name, (scores, _) in scores_dict.items():
        print(f"\n{method_name.upper()} SCORING STATISTICS:")
        print("-" * 35)
        
        stats = scorer.get_scoring_stats(scores, attention_mask)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

def visualize_scores(scores_dict, attention_mask, save_plots=True):
    """Create visualizations of the relevance scores."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Relevance Scoring Visualization', fontsize=16)
    
    # Plot 1: Attention-based scores
    ax1 = axes[0, 0]
    attention_scores = scores_dict['attention'][0]
    im1 = ax1.imshow(attention_scores.detach().numpy(), cmap='viridis', aspect='auto')
    ax1.set_title('Attention-Based Scoring')
    ax1.set_xlabel('Sequence Position')
    ax1.set_ylabel('Batch Index')
    plt.colorbar(im1, ax=ax1)
    
    # Plot 2: Variance-based scores
    ax2 = axes[0, 1]
    variance_scores = scores_dict['variance'][0]
    im2 = ax2.imshow(variance_scores.detach().numpy(), cmap='plasma', aspect='auto')
    ax2.set_title('Variance-Based Scoring')
    ax2.set_xlabel('Sequence Position')
    ax2.set_ylabel('Batch Index')
    plt.colorbar(im2, ax=ax2)
    
    # Plot 3: Hybrid scores
    ax3 = axes[1, 0]
    hybrid_scores = scores_dict['hybrid'][0]
    im3 = ax3.imshow(hybrid_scores.detach().numpy(), cmap='magma', aspect='auto')
    ax3.set_title('Hybrid Scoring')
    ax3.set_xlabel('Sequence Position')
    ax3.set_ylabel('Batch Index')
    plt.colorbar(im3, ax=ax3)
    
    # Plot 4: Score comparison across methods
    ax4 = axes[1, 1]
    batch_idx, seq_idx = 0, 25  # Compare at middle position of first sequence
    
    methods = list(scores_dict.keys())
    comparison_scores = [scores_dict[method][0][batch_idx, seq_idx].item() for method in methods]
    
    bars = ax4.bar(methods, comparison_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax4.set_title(f'Score Comparison at Position ({batch_idx}, {seq_idx})')
    ax4.set_ylabel('Relevance Score')
    ax4.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, comparison_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('relevance_scoring_visualization.png', dpi=300, bbox_inches='tight')
        print("  ✓ Visualization saved as 'relevance_scoring_visualization.png'")
    
    plt.show()

def demonstrate_integration_with_hooks():
    """Demonstrate integration with HookManager."""
    print("\n" + "=" * 60)
    print("INTEGRATION WITH HOOK MANAGER")
    print("=" * 60)
    
    # Create a simple transformer layer
    layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
    
    # Create hook manager and relevance scorer
    hook_manager = HookManager()
    relevance_scorer = RelevanceScorer(128, "attention_based")
    
    # Register hook
    hook_id = hook_manager.register_capture_hook(layer, "demo_hook", layer_idx=0)
    
    # Create input
    x = torch.randn(2, 10, 128)  # batch_size=2, seq_len=10, hidden_size=128
    
    # Forward pass
    with torch.no_grad():
        output = layer(x)
    
    # Get captured data
    captured_data = hook_manager.get_captured_data(hook_id)
    if captured_data:
        hidden_state = captured_data['hidden_state']
        print(f"Captured hidden state shape: {hidden_state.shape}")
        
        # Apply relevance scoring
        scores = relevance_scorer(hidden_state)
        print(f"Relevance scores shape: {scores.shape}")
        print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # Get top positions
        top_positions = relevance_scorer.get_top_k_positions(scores, k=5)
        print(f"Top 5 positions: {top_positions}")
    
    # Cleanup
    hook_manager.remove_hooks()

def demonstrate_integration_with_cmr_transformer():
    """Demonstrate integration with CMRTransformer."""
    print("\n" + "=" * 60)
    print("INTEGRATION WITH CMR TRANSFORMER")
    print("=" * 60)
    
    # Create small config for demonstration
    config = AutoConfig.from_pretrained("openai-community/gpt2")
    
    memory_config = {
        'target_layers': [1, 2],
        'buffer_size': 100,
        'relevance_threshold': 0.5
    }
    
    # Create model
    model = CMRTransformer(config, memory_config)
    model.register_memory_hooks()
    
    # Create input
    input_ids = torch.randint(0, 1000, (2, 20))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Check memory capture
    memory_stats = outputs['memory_stats']
    print(f"Memory capture successful: {memory_stats['total_captured_states'] > 0}")
    print(f"Layers with memory: {memory_stats['layers_with_memory']}")
    print(f"Memory usage: {memory_stats['memory_usage_mb']:.2f} MB")
    
    # Cleanup
    model.cleanup_hooks()

def performance_benchmark(scores_dict, hidden_states, attention_mask, num_runs=100):
    """Benchmark performance of different scoring methods."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    hidden_size = hidden_states.size(-1)
    
    for method_name in scores_dict.keys():
        print(f"\n{method_name.upper()} SCORING:")
        print("-" * 25)
        
        # Create scorer
        scorer = RelevanceScorer(hidden_size, method_name)
        
        # Warmup
        for _ in range(10):
            _ = scorer(hidden_states, attention_mask)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = scorer(hidden_states, attention_mask)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = num_runs / total_time
        
        print(f"  Total time ({num_runs} runs): {total_time:.4f}s")
        print(f"  Average time per run: {avg_time:.6f}s")
        print(f"  Throughput: {throughput:.1f} runs/second")

def main():
    """Main demonstration function."""
    print("Contextual Memory Reweaving - Day 3: Relevance Scoring Foundation")
    print("=" * 70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create sample data
    print("\nCreating sample data...")
    hidden_states, attention_mask = create_sample_data()
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Demonstrate scoring methods
    scores_dict = demonstrate_scoring_methods(hidden_states, attention_mask)
    
    # Demonstrate top-k selection
    demonstrate_top_k_selection(scores_dict, attention_mask)
    
    # Demonstrate scoring statistics
    demonstrate_scoring_stats(scores_dict, attention_mask)
    
    # Create visualizations
    try:
        visualize_scores(scores_dict, attention_mask)
    except ImportError:
        print("  ⚠️  matplotlib not available, skipping visualizations")
    
    # Demonstrate integration
    demonstrate_integration_with_hooks()
    demonstrate_integration_with_cmr_transformer()
    
    # Performance benchmark
    performance_benchmark(scores_dict, hidden_states, attention_mask)
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("✓ Three scoring methods (attention, variance, hybrid)")
    print("✓ Top-k position selection")
    print("✓ Scoring statistics and analysis")
    print("✓ Integration with HookManager and CMRTransformer")
    print("✓ Performance benchmarking")
    print("✓ Comprehensive error handling and validation")

if __name__ == "__main__":
    main()

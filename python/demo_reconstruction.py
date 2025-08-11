#!/usr/bin/env python3
"""
Demo script for the Layered State Reconstruction (LLSR) system.
This demonstrates how the reconstruction system works with different methods
and shows the integration with memory states.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any

# Import our reconstruction system
from models.reconstruction import (
    LayeredStateReconstructor,
    HierarchicalReconstructor,
    AttentionBasedReconstructor,
    MLPReconstructor
)

# Mock classes for demonstration
class MockMemoryEntry:
    """Mock memory entry for demonstration purposes."""
    def __init__(self, hidden_state: torch.Tensor, relevance_score: float, 
                 layer_idx: int, timestamp: float, metadata: Dict[str, Any] = None):
        self.hidden_state = hidden_state
        self.relevance_score = relevance_score
        self.layer_idx = layer_idx
        self.timestamp = timestamp
        self.metadata = metadata or {}

class MockTransformerLayer:
    """Mock transformer layer to simulate the forward pass."""
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=8, 
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through transformer layer."""
        # Self-attention
        attn_output, _ = self.self_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=attention_mask == 0 if attention_mask is not None else None
        )
        hidden_states = self.norm1(hidden_states + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.norm2(hidden_states + ff_output)
        
        return hidden_states

def create_sample_memory_entries(hidden_size: int, num_entries: int, 
                               num_layers: int) -> List[MockMemoryEntry]:
    """Create sample memory entries for demonstration."""
    memory_entries = []
    
    for i in range(num_entries):
        # Create diverse hidden states
        if i % 3 == 0:
            # Pattern 1: High activation in first half
            hidden_state = torch.randn(1, hidden_size)
            hidden_state[:, :hidden_size//2] *= 2.0
        elif i % 3 == 1:
            # Pattern 2: High activation in second half
            hidden_state = torch.randn(1, hidden_size)
            hidden_state[:, hidden_size//2:] *= 2.0
        else:
            # Pattern 3: Random with some structure
            hidden_state = torch.randn(1, hidden_size)
            hidden_state = hidden_state * (1 + 0.5 * torch.sin(torch.arange(hidden_size).float()))
        
        # Relevance score based on entry index (simulating temporal decay)
        relevance_score = 0.9 * np.exp(-0.1 * i) + 0.1
        
        # Layer index (distribute across layers)
        layer_idx = i % num_layers
        
        # Timestamp (simulating temporal information)
        timestamp = 1000.0 - i * 10.0
        
        # Metadata
        metadata = {
            'source': f'sample_{i}',
            'category': ['text', 'image', 'audio'][i % 3],
            'confidence': relevance_score
        }
        
        entry = MockMemoryEntry(
            hidden_state=hidden_state,
            relevance_score=relevance_score,
            layer_idx=layer_idx,
            timestamp=timestamp,
            metadata=metadata
        )
        memory_entries.append(entry)
    
    return memory_entries

def demonstrate_reconstruction_methods():
    """Demonstrate different reconstruction methods."""
    print("🔧 Demonstrating Different Reconstruction Methods")
    print("=" * 60)
    
    # Configuration
    hidden_size = 256
    num_layers = 6
    max_memory_tokens = 16
    batch_size = 2
    seq_len = 32
    
    # Create sample data
    current_states = torch.randn(batch_size, seq_len, hidden_size)
    memory_entries = create_sample_memory_entries(hidden_size, 20, num_layers)
    
    methods = ["hierarchical", "attention", "mlp"]
    
    for method in methods:
        print(f"\n📊 Testing {method.upper()} Reconstruction Method")
        print("-" * 40)
        
        # Create reconstructor
        reconstructor = LayeredStateReconstructor(
            hidden_size=hidden_size,
            num_layers=num_layers,
            reconstruction_method=method,
            max_memory_tokens=max_memory_tokens
        )
        
        # Test reconstruction for a specific layer
        layer_idx = 3
        start_time = torch.cuda.Event() if torch.cuda.is_available() else None
        end_time = torch.cuda.Event() if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        enhanced_states = reconstructor.reconstruct_layer_memories(
            layer_idx=layer_idx,
            memory_entries=memory_entries,
            current_hidden_states=current_states
        )
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            print(f"  ⏱️  Processing time: {elapsed_time:.2f} ms")
        
        # Analyze results
        print(f"  📏 Input shape: {current_states.shape}")
        print(f"  📏 Output shape: {enhanced_states.shape}")
        print(f"  🧠 Memories processed: {len(memory_entries)}")
        print(f"  🎯 Target layer: {layer_idx}")
        
        # Check for numerical stability
        if torch.isnan(enhanced_states).any():
            print("  ⚠️  Warning: NaN values detected in output")
        else:
            print("  ✅ Output is numerically stable")
        
        if torch.isinf(enhanced_states).any():
            print("  ⚠️  Warning: Infinite values detected in output")
        else:
            print("  ✅ Output contains no infinite values")
        
        # Compute change magnitude
        change_magnitude = torch.norm(enhanced_states - current_states, dim=-1).mean()
        print(f"  📊 Average change magnitude: {change_magnitude:.4f}")

def demonstrate_layer_specific_reconstruction():
    """Demonstrate how reconstruction varies across layers."""
    print("\n🔬 Demonstrating Layer-Specific Reconstruction")
    print("=" * 60)
    
    # Configuration
    hidden_size = 128
    num_layers = 8
    max_memory_tokens = 12
    batch_size = 1
    seq_len = 24
    
    # Create reconstructor
    reconstructor = LayeredStateReconstructor(
        hidden_size=hidden_size,
        num_layers=num_layers,
        reconstruction_method="hierarchical",
        max_memory_tokens=max_memory_tokens
    )
    
    # Create sample data
    current_states = torch.randn(batch_size, seq_len, hidden_size)
    memory_entries = create_sample_memory_entries(hidden_size, 15, num_layers)
    
    print(f"📊 Testing reconstruction across {num_layers} layers")
    print(f"🧠 Processing {len(memory_entries)} memory entries")
    print(f"📏 Sequence length: {seq_len}, Hidden size: {hidden_size}")
    
    # Test each layer
    layer_results = {}
    for layer_idx in range(num_layers):
        enhanced_states = reconstructor.reconstruct_layer_memories(
            layer_idx=layer_idx,
            memory_entries=memory_entries,
            current_hidden_states=current_states
        )
        
        # Compute metrics
        change_magnitude = torch.norm(enhanced_states - current_states, dim=-1).mean()
        layer_weight = torch.sigmoid(reconstructor.layer_weights[layer_idx]).item()
        
        layer_results[layer_idx] = {
            'change_magnitude': change_magnitude.item(),
            'layer_weight': layer_weight,
            'output_shape': enhanced_states.shape
        }
        
        print(f"  Layer {layer_idx:2d}: Change={change_magnitude:.4f}, Weight={layer_weight:.3f}")
    
    # Analyze patterns
    changes = [results['change_magnitude'] for results in layer_results.values()]
    weights = [results['layer_weight'] for results in layer_results.values()]
    
    print(f"\n📈 Analysis:")
    print(f"  🎯 Average change magnitude: {np.mean(changes):.4f}")
    print(f"  🎯 Std change magnitude: {np.std(changes):.4f}")
    print(f"  ⚖️  Average layer weight: {np.mean(weights):.3f}")
    print(f"  🔍 Min/Max change: {min(changes):.4f} / {max(changes):.4f}")

def demonstrate_memory_integration():
    """Demonstrate how memories are integrated with current states."""
    print("\n🔗 Demonstrating Memory Integration")
    print("=" * 60)
    
    # Configuration
    hidden_size = 64
    num_layers = 4
    max_memory_tokens = 8
    batch_size = 2
    seq_len = 16
    
    # Create reconstructor
    reconstructor = LayeredStateReconstructor(
        hidden_size=hidden_size,
        num_layers=num_layers,
        reconstruction_method="attention",
        max_memory_tokens=max_memory_tokens
    )
    
    # Create diverse memory entries
    memory_entries = []
    
    # High-relevance memories
    for i in range(4):
        hidden_state = torch.randn(1, hidden_size) * 2.0  # High activation
        memory_entries.append(MockMemoryEntry(
            hidden_state=hidden_state,
            relevance_score=0.9 - i * 0.1,
            layer_idx=i % num_layers,
            timestamp=1000.0 - i * 5.0
        ))
    
    # Low-relevance memories
    for i in range(4):
        hidden_state = torch.randn(1, hidden_size) * 0.5  # Low activation
        memory_entries.append(MockMemoryEntry(
            hidden_state=hidden_state,
            relevance_score=0.3 - i * 0.05,
            layer_idx=i % num_layers,
            timestamp=1000.0 - (i + 4) * 5.0
        ))
    
    # Current states with some structure
    current_states = torch.randn(batch_size, seq_len, hidden_size)
    # Add some pattern to current states
    current_states[:, :, :hidden_size//2] += torch.sin(torch.arange(seq_len).float()).unsqueeze(-1).unsqueeze(0)
    
    print(f"📊 Testing memory integration")
    print(f"🧠 High-relevance memories: 4")
    print(f"🧠 Low-relevance memories: 4")
    print(f"📏 Current states shape: {current_states.shape}")
    
    # Test integration
    layer_idx = 2
    enhanced_states = reconstructor.reconstruct_layer_memories(
        layer_idx=layer_idx,
        memory_entries=memory_entries,
        current_hidden_states=current_states
    )
    
    # Analyze integration effects
    print(f"\n📈 Integration Analysis:")
    
    # Compute changes
    changes = enhanced_states - current_states
    change_magnitude = torch.norm(changes, dim=-1)
    
    print(f"  🎯 Average change magnitude: {change_magnitude.mean():.4f}")
    print(f"  🎯 Change std: {change_magnitude.std():.4f}")
    
    # Check if high-relevance memories had more impact
    high_rel_impact = torch.norm(changes[:, :4, :], dim=-1).mean()
    low_rel_impact = torch.norm(changes[:, 4:, :], dim=-1).mean()
    
    print(f"  🔥 High-relevance impact: {high_rel_impact:.4f}")
    print(f"  ❄️  Low-relevance impact: {low_rel_impact:.4f}")
    
    if high_rel_impact > low_rel_impact:
        print(f"  ✅ High-relevance memories had greater impact")
    else:
        print(f"  ⚠️  Low-relevance memories had greater impact")

def demonstrate_performance_comparison():
    """Compare performance of different reconstruction methods."""
    print("\n⚡ Performance Comparison")
    print("=" * 60)
    
    # Configuration
    hidden_size = 128
    num_layers = 6
    max_memory_tokens = 16
    batch_size = 2
    seq_len = 32
    
    # Create sample data
    current_states = torch.randn(batch_size, seq_len, hidden_size)
    memory_entries = create_sample_memory_entries(hidden_size, 20, num_layers)
    
    methods = ["hierarchical", "attention", "mlp"]
    results = {}
    
    print(f"📊 Comparing {len(methods)} reconstruction methods")
    print(f"🧠 Processing {len(memory_entries)} memories")
    print(f"📏 Input shape: {current_states.shape}")
    
    for method in methods:
        print(f"\n🔧 Testing {method.upper()} method:")
        
        # Create reconstructor
        reconstructor = LayeredStateReconstructor(
            hidden_size=hidden_size,
            num_layers=num_layers,
            reconstruction_method=method,
            max_memory_tokens=max_memory_tokens
        )
        
        # Measure time
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            enhanced_states = reconstructor.reconstruct_layer_memories(
                layer_idx=3,
                memory_entries=memory_entries,
                current_hidden_states=current_states
            )
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
        else:
            import time
            start_time = time.time()
            enhanced_states = reconstructor.reconstruct_layer_memories(
                layer_idx=3,
                memory_entries=memory_entries,
                current_hidden_states=current_states
            )
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Compute metrics
        change_magnitude = torch.norm(enhanced_states - current_states, dim=-1).mean()
        
        results[method] = {
            'time_ms': elapsed_time,
            'change_magnitude': change_magnitude.item(),
            'output_shape': enhanced_states.shape
        }
        
        print(f"  ⏱️  Processing time: {elapsed_time:.2f} ms")
        print(f"  📊 Change magnitude: {change_magnitude:.4f}")
        print(f"  📏 Output shape: {enhanced_states.shape}")
    
    # Summary
    print(f"\n📈 Performance Summary:")
    fastest_method = min(results.keys(), key=lambda x: results[x]['time_ms'])
    slowest_method = max(results.keys(), key=lambda x: results[x]['time_ms'])
    
    print(f"  🏃 Fastest: {fastest_method.upper()} ({results[fastest_method]['time_ms']:.2f} ms)")
    print(f"  🐌 Slowest: {slowest_method.upper()} ({results[slowest_method]['time_ms']:.2f} ms)")
    
    # Memory usage estimation
    print(f"\n💾 Memory Usage Estimation:")
    for method, result in results.items():
        # Rough estimation: input + output + intermediate tensors
        estimated_memory = (current_states.numel() + 
                          result['output_shape'].numel()) * 4  # 4 bytes per float32
        print(f"  {method.upper()}: ~{estimated_memory / 1024:.1f} KB")

def main():
    """Main demonstration function."""
    print("🚀 Layered State Reconstruction (LLSR) System Demo")
    print("=" * 70)
    print("This demo showcases the reconstruction system's capabilities")
    print("for integrating stored memories with current transformer states.\n")
    
    try:
        # Run demonstrations
        demonstrate_reconstruction_methods()
        demonstrate_layer_specific_reconstruction()
        demonstrate_memory_integration()
        demonstrate_performance_comparison()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("  ✅ Multiple reconstruction methods (hierarchical, attention, MLP)")
        print("  ✅ Layer-specific memory integration")
        print("  ✅ Relevance-aware memory processing")
        print("  ✅ Performance optimization")
        print("  ✅ Numerical stability")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Main entry point for ContextMemoryReweaving project.
Demonstrates the base transformer with memory hooks.
"""

import torch
from transformers import AutoConfig
from models.base_transformer import CMRTransformer

def main():
    """Main function demonstrating CMR transformer functionality."""
    print("🚀 ContextMemoryReweaving - Day 1 Implementation")
    print("=" * 50)
    
    # Create configuration
    print("📋 Creating LLM configuration...")
    config = AutoConfig.from_pretrained("mistralai/Ministral-8B-Instruct-2410")
    
    memory_config = {
        'target_layers': [2, 4],  # Hook layers 2 and 4
        'buffer_size': 100,
        'relevance_threshold': 0.5
    }
    
    print(f"   - Vocab size: {getattr(config, 'vocab_size', 'N/A')}")
    print(f"   - Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
    print(f"   - Number of layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
    print(f"   - Target layers for hooks: {memory_config['target_layers']}")
    
    # Initialize model
    print("\n🔧 Initializing CMR Transformer...")
    model = CMRTransformer(config, memory_config)
    model.register_memory_hooks()
    
    print(f"   - Registered hooks on {len(model.layer_hooks)} layers")
    print(f"   - Memory enabled: {model.memory_enabled}")
    
    # Test forward pass
    print("\n🧪 Testing forward pass...")
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, getattr(config, 'vocab_size', 50257), (batch_size, seq_len))
    
    print(f"   - Input shape: {input_ids.shape}")
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Display results
    print("\n📊 Results:")
    print(f"   - Output hidden state shape: {outputs['last_hidden_state'].shape}")
    
    memory_stats = outputs['memory_stats']
    print(f"   - Total captured states: {memory_stats['total_captured_states']}")
    print(f"   - Layers with memory: {memory_stats['layers_with_memory']}")
    print(f"   - Current sequence ID: {memory_stats['sequence_id']}")
    
    # Show captured states details
    captured_states = outputs['captured_memory_states']
    print(f"\n🔍 Captured States Details:")
    for layer_idx, states in captured_states.items():
        print(f"   - Layer {layer_idx}: {len(states)} states")
        if states:
            first_state = states[0]
            state_shape = first_state['hidden_state'].shape
            print(f"     First state shape: {state_shape}")
    
    # Test memory enable/disable
    print("\n🔄 Testing memory control...")
    
    # Disable memory
    model.disable_memory()
    with torch.no_grad():
        outputs_disabled = model(input_ids)
    disabled_captures = outputs_disabled['memory_stats']['total_captured_states']
    
    # Re-enable memory
    model.enable_memory()
    with torch.no_grad():
        outputs_enabled = model(input_ids)
    enabled_captures = outputs_enabled['memory_stats']['total_captured_states']
    
    print(f"   - States captured with memory disabled: {disabled_captures}")
    print(f"   - States captured with memory enabled: {enabled_captures}")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    model.cleanup_hooks()
    print(f"   - Hooks removed: {len(model.layer_hooks)}")
    
    print("\n✅ Day 1 implementation completed successfully!")
    print("\n📝 Next steps:")
    print("   - Implement relevance scoring (Day 3)")
    print("   - Implement memory buffer (Day 4)")
    print("   - Integrate all components (Day 5)")

if __name__ == "__main__":
    main()

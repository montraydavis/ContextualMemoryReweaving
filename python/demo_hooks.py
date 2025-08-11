#!/usr/bin/env python3
"""
Demonstration script for the advanced hook system (Day 2 implementation).
Shows how to use HookManager for sophisticated memory capture with Mistral.
"""

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from utils.hooks import HookManager
from models.base_transformer import CMRTransformer

def demo_basic_hooks():
    """Demonstrate basic hook functionality."""
    print("🔧 Demo 1: Basic Hook Management")
    print("=" * 50)
    
    # Create hook manager
    hook_manager = HookManager()
    
    # Create a simple transformer layer for demonstration
    layer = torch.nn.TransformerEncoderLayer(d_model=128, nhead=4)
    
    # Register a hook
    hook_id = hook_manager.register_capture_hook(
        layer, "demo_hook", layer_idx=0
    )
    print(f"✅ Registered hook: {hook_id}")
    
    # Test forward pass
    x = torch.randn(2, 30, 128)  # batch_size=2, seq_len=30, hidden_size=128
    output = layer(x)
    
    # Check captured data
    captured = hook_manager.get_captured_data("demo_hook")
    print(f"📊 Captured data shape: {captured['hidden_state'].shape}")
    print(f"📊 Layer index: {captured['layer_idx']}")
    
    # Check memory usage
    memory_stats = hook_manager.get_memory_usage()
    print(f"💾 Memory usage: {memory_stats['total_memory_mb']:.2f} MB")
    
    # Cleanup
    hook_manager.remove_hooks()
    print("🧹 Hooks cleaned up\n")

def demo_multiple_layer_hooks():
    """Demonstrate hooking multiple layers."""
    print("🔧 Demo 2: Multiple Layer Hooks")
    print("=" * 50)
    
    # Create small Mistral model for testing
    print("📥 Loading Mistral model for testing...")
    model_name = "mistralai/Mistral-7B-v0.1"  # Using smaller Mistral variant for testing
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 to reduce memory usage
            device_map="auto",  # Automatically handle device placement
            load_in_8bit=True,  # Use 8-bit quantization for memory efficiency
        )
        print(f"✅ Mistral model loaded successfully: {model_name}")
        
        # Create hook manager
        hook_manager = HookManager()
        
        # Hook multiple layers (Mistral has 32 layers, we'll hook a few)
        target_layers = [8, 16, 24]  # Distributed across the model
        hook_ids = hook_manager.register_layer_hooks(
            model.model.layers, target_layers, "mistral_demo"
        )
        print(f"✅ Registered hooks on layers: {target_layers}")
        print(f"📋 Hook IDs: {hook_ids}")
        
        # Test forward pass with a simple prompt
        prompt = "Hello, this is a test of the Mistral model with hooks."
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Check all captures
        for layer_idx in target_layers:
            hook_id = f"mistral_demo_{layer_idx}"
            captured = hook_manager.get_captured_data(hook_id)
            if captured:
                print(f"📊 Layer {layer_idx}: {captured['hidden_state'].shape}")
            else:
                print(f"📊 Layer {layer_idx}: No capture (hook may not have triggered)")
        
        # Memory usage
        memory_stats = hook_manager.get_memory_usage()
        print(f"💾 Total captured tensors: {memory_stats['total_captured_tensors']}")
        print(f"💾 Total memory: {memory_stats['total_memory_mb']:.2f} MB")
        
        # Cleanup
        hook_manager.remove_hooks()
        print("🧹 Hooks cleaned up")
        
        # Free model memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"⚠️  Could not load full Mistral model: {e}")
        print("🔄 Falling back to mock model for demonstration...")
        
        # Fallback: Create mock model structure
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(d_model=128, nhead=4) 
            for _ in range(32)  # Mistral has 32 layers
        ])
        
        # Create hook manager
        hook_manager = HookManager()
        
        # Hook multiple layers
        target_layers = [8, 16, 24]
        hook_ids = hook_manager.register_layer_hooks(
            model.model.layers, target_layers, "mistral_mock_demo"
        )
        print(f"✅ Registered hooks on layers: {target_layers}")
        print(f"📋 Hook IDs: {hook_ids}")
        
        # Test forward pass
        x = torch.randn(1, 25, 128)
        for layer in model.model.layers:
            x = layer(x)
        
        # Check all captures
        for layer_idx in target_layers:
            hook_id = f"mistral_mock_demo_{layer_idx}"
            captured = hook_manager.get_captured_data(hook_id)
            print(f"📊 Layer {layer_idx}: {captured['hidden_state'].shape}")
        
        # Memory usage
        memory_stats = hook_manager.get_memory_usage()
        print(f"💾 Total captured tensors: {memory_stats['total_captured_tensors']}")
        print(f"💾 Total memory: {memory_stats['total_memory_mb']:.2f} MB")
        
        # Cleanup
        hook_manager.remove_hooks()
        print("🧹 Hooks cleaned up")
    
    print()

def demo_custom_capture():
    """Demonstrate custom capture functions."""
    print("🔧 Demo 3: Custom Capture Functions")
    print("=" * 50)
    
    # Create hook manager
    hook_manager = HookManager()
    
    # Create a simple layer
    layer = torch.nn.Linear(128, 128)
    
    # Custom capture function: only capture if input has high variance
    def variance_based_capture(module, input, output):
        input_tensor = input[0]  # input is a tuple
        variance = torch.var(input_tensor)
        
        if variance > 0.5:  # Only capture high-variance inputs
            hook_manager.hook_data["variance_hook"] = {
                'hidden_state': output.detach().clone(),
                'input_variance': variance.item(),
                'capture_reason': 'high_variance'
            }
    
    # Register custom hook
    hook_id = hook_manager.register_capture_hook(
        layer, "variance_hook", 
        capture_fn=variance_based_capture
    )
    print(f"✅ Registered custom hook: {hook_id}")
    
    # Test with low variance input (should not capture)
    low_var_input = torch.ones(1, 10, 128) * 0.1
    low_var_output = layer(low_var_input)
    
    captured = hook_manager.get_captured_data("variance_hook")
    print(f"📊 Low variance input captured: {captured is not None}")
    
    # Test with high variance input (should capture)
    high_var_input = torch.randn(1, 10, 128)
    high_var_output = layer(high_var_input)
    
    captured = hook_manager.get_captured_data("variance_hook")
    if captured:
        print(f"📊 High variance input captured: {captured['capture_reason']}")
        print(f"📊 Input variance: {captured['input_variance']:.3f}")
    
    # Cleanup
    hook_manager.remove_hooks()
    print("🧹 Hooks cleaned up\n")

def demo_cmr_integration():
    """Demonstrate integration with CMR transformer."""
    print("🔧 Demo 4: CMR Transformer Integration with Mistral")
    print("=" * 60)
    
    # Create Mistral configuration
    print("📥 Loading Mistral configuration...")
    try:
        config = AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1")
        print(f"✅ Mistral config loaded: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
        
        memory_config = {
            'target_layers': [8, 16, 24],  # Distributed across Mistral's 32 layers
            'buffer_size': 100
        }
        
        print(f"🎯 Target layers for memory capture: {memory_config['target_layers']}")
        
        # Note: For full Mistral integration, you'd need to modify CMRTransformer
        # to work with Mistral's architecture. For now, we'll show the concept.
        print("ℹ️  Note: Full Mistral integration requires CMRTransformer modifications")
        print("   This demo shows the configuration and target layer setup")
        
        # Show what the memory config would look like
        print(f"\n📋 Memory Configuration:")
        print(f"   • Target layers: {memory_config['target_layers']}")
        print(f"   • Buffer size: {memory_config['buffer_size']}")
        print(f"   • Model layers: {config.num_hidden_layers}")
        print(f"   • Hidden size: {config.hidden_size}")
        print(f"   • Attention heads: {config.num_attention_heads}")
        
        # Demonstrate layer selection logic
        total_layers = config.num_hidden_layers
        early_layers = [i for i in range(0, total_layers//3)]
        middle_layers = [i for i in range(total_layers//3, 2*total_layers//3)]
        late_layers = [i for i in range(2*total_layers//3, total_layers)]
        
        print(f"\n🔍 Layer Distribution Analysis:")
        print(f"   • Early layers (0-{len(early_layers)-1}): {early_layers[:5]}...")
        print(f"   • Middle layers ({len(early_layers)}-{len(early_layers)+len(middle_layers)-1}): {middle_layers[:5]}...")
        print(f"   • Late layers ({len(early_layers)+len(middle_layers)}-{total_layers-1}): {late_layers[:5]}...")
        
        # Recommend optimal layer selection for Mistral
        recommended_layers = [
            early_layers[len(early_layers)//2],      # Middle of early layers
            middle_layers[len(middle_layers)//2],    # Middle of middle layers  
            late_layers[len(late_layers)//2]         # Middle of late layers
        ]
        
        print(f"\n💡 Recommended target layers for Mistral:")
        print(f"   • {recommended_layers} (distributed across model depth)")
        print(f"   • This provides good coverage of different abstraction levels")
        
    except Exception as e:
        print(f"⚠️  Could not load Mistral config: {e}")
        print("🔄 Using fallback configuration...")
        
        # Fallback config
        config = AutoConfig.from_pretrained("openai-community/gpt2")
        memory_config = {
            'target_layers': [2, 4],
            'buffer_size': 50
        }
        
        print(f"✅ Fallback config loaded: {config.num_hidden_layers} layers")
        
        # Create CMR transformer with fallback
        model = CMRTransformer(config, memory_config)
        model.register_memory_hooks()
        print(f"✅ CMR Transformer created with hooks on layers: {memory_config['target_layers']}")
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 20))
        print(f"📝 Input shape: {input_ids.shape}")
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Display results
        memory_stats = outputs['memory_stats']
        print(f"📊 Total captured states: {memory_stats['total_captured_states']}")
        print(f"📊 Layers with memory: {memory_stats['layers_with_memory']}")
        
        # Show captured states details
        captured_states = outputs['captured_memory_states']
        for layer_idx, states in captured_states.items():
            print(f"🔍 Layer {layer_idx}: {len(states)} states")
            if states:
                first_state = states[0]
                state_shape = first_state['hidden_state'].shape
                print(f"   First state shape: {state_shape}")
        
        # Test memory control
        print("\n🔄 Testing memory control...")
        model.disable_memory()
        with torch.no_grad():
            outputs_disabled = model(input_ids)
        disabled_captures = outputs_disabled['memory_stats']['total_captured_states']
        
        model.enable_memory()
        with torch.no_grad():
            outputs_enabled = model(input_ids)
        enabled_captures = outputs_enabled['memory_stats']['total_captured_states']
        
        print(f"📊 States with memory disabled: {disabled_captures}")
        print(f"📊 States with memory enabled: {enabled_captures}")
        
        # Cleanup
        model.cleanup_hooks()
        print("🧹 CMR hooks cleaned up")
    
    print()

def main():
    """Run all demonstrations."""
    print("🚀 Contextual Memory Reweaving - Day 2 Hook System Demo with Mistral")
    print("=" * 80)
    print("This demonstration showcases the advanced hook system implementation")
    print("that provides sophisticated memory capture capabilities with Mistral models.\n")
    
    try:
        # Run all demos
        demo_basic_hooks()
        demo_multiple_layer_hooks()
        demo_custom_capture()
        demo_cmr_integration()
        
        print("✅ All demonstrations completed successfully!")
        print("\n📝 Key Features Demonstrated:")
        print("   • Centralized hook management")
        print("   • Multi-layer hook registration with Mistral")
        print("   • Custom capture functions")
        print("   • Memory usage tracking")
        print("   • Mistral architecture analysis")
        print("   • Optimal layer selection for memory capture")
        print("   • Proper cleanup and resource management")
        
        print("\n🔮 Next Steps for Full Mistral Integration:")
        print("   • Modify CMRTransformer to work with Mistral's architecture")
        print("   • Implement Mistral-specific layer hooking")
        print("   • Add support for Mistral's sliding window attention")
        print("   • Optimize memory capture for Mistral's 32-layer structure")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("This might be due to missing dependencies or import issues.")
        print("\n💡 To use full Mistral models, ensure you have:")
        print("   • Sufficient GPU memory (24GB+ for 8B models)")
        print("   • transformers>=4.35.0")
        print("   • torch>=2.0.0")
        print("   • accelerate for device mapping")

if __name__ == "__main__":
    main()

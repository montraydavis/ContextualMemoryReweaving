import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional, Any, Tuple
from collections import defaultdict
import weakref

class HookManager:
    """
    Centralized hook management system for transformer layers.
    Handles registration, removal, and coordination of multiple hooks.
    """
    
    def __init__(self):
        self.hooks = {}  # module_id -> list of (hook_id, hook_handle) tuples
        self.hook_data = defaultdict(dict)  # Storage for hook outputs
        self.hook_configs = {}  # Hook configuration storage
        
    def register_capture_hook(self, 
                            module: nn.Module, 
                            hook_id: str,
                            capture_fn: Optional[Callable] = None,
                            layer_idx: Optional[int] = None) -> str:
        """
        Register a forward hook for capturing intermediate states.
        
        Args:
            module: PyTorch module to hook
            hook_id: Unique identifier for this hook
            capture_fn: Custom capture function (optional)
            layer_idx: Layer index for organization
            
        Returns:
            Hook handle ID for later removal
            
        Raises:
            ValueError: If hook_id already exists or module is None
            RuntimeError: If hook registration fails
        """
        if module is None:
            raise ValueError("Module cannot be None")
        
        if hook_id in self.hook_configs:
            raise ValueError(f"Hook ID '{hook_id}' already exists")
        
        def default_capture_fn(module, input, output):
            """Default capture function for hidden states."""
            try:
                # Handle different output formats
                if isinstance(output, tuple):
                    hidden_state = output[0]  # Usually first element
                else:
                    hidden_state = output
                    
                # Store captured data
                self.hook_data[hook_id] = {
                    'hidden_state': hidden_state.detach().clone(),
                    'layer_idx': layer_idx,
                    'module_name': module.__class__.__name__,
                    'capture_timestamp': len(self.hook_data)
                }
            except Exception as e:
                # Log error but don't crash the hook
                print(f"Warning: Hook {hook_id} capture failed: {e}")
                
        # Use provided capture function or default
        capture_function = capture_fn if capture_fn else default_capture_fn
        
        try:
            # Register the hook
            hook_handle = module.register_forward_hook(capture_function)
            
            # Store hook information with proper tracking
            module_id = id(module)
            if module_id not in self.hooks:
                self.hooks[module_id] = []
            
            # Store as tuple for proper tracking
            self.hooks[module_id].append((hook_id, hook_handle))
            
            # Store configuration
            self.hook_configs[hook_id] = {
                'module_id': module_id,
                'layer_idx': layer_idx,
                'module_ref': weakref.ref(module)
            }
            
            return hook_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to register hook {hook_id}: {e}")
    
    def register_layer_hooks(self, 
                           transformer_layers: nn.ModuleList,
                           target_layers: List[int],
                           hook_prefix: str = "layer") -> List[str]:
        """
        Register hooks on multiple transformer layers.
        
        Args:
            transformer_layers: ModuleList of transformer layers
            target_layers: List of layer indices to hook
            hook_prefix: Prefix for hook IDs
            
        Returns:
            List of registered hook IDs
            
        Raises:
            ValueError: If target_layers contains invalid indices
        """
        if not isinstance(transformer_layers, nn.ModuleList):
            raise ValueError("transformer_layers must be a ModuleList")
        
        if not target_layers:
            return []
        
        # Validate layer indices
        max_layer = len(transformer_layers) - 1
        invalid_layers = [idx for idx in target_layers if idx < 0 or idx > max_layer]
        if invalid_layers:
            raise ValueError(f"Invalid layer indices: {invalid_layers}. Valid range: 0-{max_layer}")
        
        registered_hooks = []
        
        for layer_idx in target_layers:
            try:
                hook_id = f"{hook_prefix}_{layer_idx}"
                
                # Create layer-specific capture function
                def create_layer_capture(l_idx):
                    def layer_capture_fn(module, input, output):
                        try:
                            if isinstance(output, tuple):
                                hidden_state = output[0]
                            else:
                                hidden_state = output
                                
                            self.hook_data[f"{hook_prefix}_{l_idx}"] = {
                                'hidden_state': hidden_state.detach().clone(),
                                'layer_idx': l_idx,
                                'sequence_position': hidden_state.shape[1],  # seq_len
                                'batch_size': hidden_state.shape[0]
                            }
                        except Exception as e:
                            print(f"Warning: Layer hook {hook_prefix}_{l_idx} capture failed: {e}")
                    return layer_capture_fn
                
                # Register hook
                hook_handle = transformer_layers[layer_idx].register_forward_hook(
                    create_layer_capture(layer_idx)
                )
                
                # Store hook info with proper tracking
                module_id = id(transformer_layers[layer_idx])
                if module_id not in self.hooks:
                    self.hooks[module_id] = []
                
                self.hooks[module_id].append((hook_id, hook_handle))
                
                # Store configuration
                self.hook_configs[hook_id] = {
                    'module_id': module_id,
                    'layer_idx': layer_idx,
                    'module_ref': weakref.ref(transformer_layers[layer_idx])
                }
                
                registered_hooks.append(hook_id)
                
            except Exception as e:
                print(f"Warning: Failed to register hook for layer {layer_idx}: {e}")
                continue
        
        return registered_hooks
    
    def get_captured_data(self, hook_id: str) -> Optional[Dict]:
        """Retrieve captured data for specific hook."""
        if not isinstance(hook_id, str):
            raise ValueError("hook_id must be a string")
        return self.hook_data.get(hook_id)
    
    def get_all_captured_data(self) -> Dict:
        """Retrieve all captured data."""
        return dict(self.hook_data)
    
    def clear_captured_data(self, hook_ids: Optional[List[str]] = None):
        """Clear captured data for specified hooks or all hooks."""
        if hook_ids is None:
            self.hook_data.clear()
        else:
            if not isinstance(hook_ids, list):
                raise ValueError("hook_ids must be a list or None")
            for hook_id in hook_ids:
                if not isinstance(hook_id, str):
                    print(f"Warning: Skipping invalid hook_id: {hook_id}")
                    continue
                self.hook_data.pop(hook_id, None)
    
    def remove_hooks(self, hook_ids: Optional[List[str]] = None):
        """Remove specified hooks or all hooks."""
        if hook_ids is None:
            # Remove all hooks
            for module_hooks in self.hooks.values():
                for _, hook_handle in module_hooks:
                    try:
                        hook_handle.remove()
                    except Exception as e:
                        print(f"Warning: Failed to remove hook: {e}")
            self.hooks.clear()
            self.hook_configs.clear()
        else:
            # Remove specific hooks
            if not isinstance(hook_ids, list):
                raise ValueError("hook_ids must be a list or None")
                
            for hook_id in hook_ids:
                if not isinstance(hook_id, str):
                    print(f"Warning: Skipping invalid hook_id: {hook_id}")
                    continue
                    
                if hook_id not in self.hook_configs:
                    print(f"Warning: Hook {hook_id} not found, skipping removal")
                    continue
                
                config = self.hook_configs[hook_id]
                module_id = config['module_id']
                
                # Find and remove the specific hook
                if module_id in self.hooks:
                    # Find the specific hook to remove
                    hooks_to_remove = []
                    for i, (stored_hook_id, hook_handle) in enumerate(self.hooks[module_id]):
                        if stored_hook_id == hook_id:
                            try:
                                hook_handle.remove()
                                hooks_to_remove.append(i)
                            except Exception as e:
                                print(f"Warning: Failed to remove hook {hook_id}: {e}")
                    
                    # Remove hooks in reverse order to maintain indices
                    for i in reversed(hooks_to_remove):
                        del self.hooks[module_id][i]
                    
                    # Clean up empty module entries
                    if not self.hooks[module_id]:
                        del self.hooks[module_id]
                
                # Remove configuration
                del self.hook_configs[hook_id]
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage statistics."""
        total_tensors = 0
        total_memory = 0
        
        for hook_data in self.hook_data.values():
            if 'hidden_state' in hook_data:
                tensor = hook_data['hidden_state']
                total_tensors += 1
                total_memory += tensor.numel() * tensor.element_size()
        
        return {
            'total_captured_tensors': total_tensors,
            'total_memory_bytes': total_memory,
            'total_memory_mb': total_memory / (1024 * 1024),
            'active_hooks': len(self.hooks),
            'total_hook_configs': len(self.hook_configs)
        }
    
    def cleanup_invalid_references(self):
        """Clean up invalid weak references to deleted modules."""
        invalid_hooks = []
        
        for hook_id, config in self.hook_configs.items():
            module_ref = config.get('module_ref')
            if module_ref is not None:
                module = module_ref()
                if module is None:  # Module was deleted
                    invalid_hooks.append(hook_id)
        
        # Remove invalid hooks
        if invalid_hooks:
            print(f"Cleaning up {len(invalid_hooks)} invalid hook references")
            self.remove_hooks(invalid_hooks)

## Example hook testing
if __name__ == "__main__":
    # Test hook manager
    hook_manager = HookManager()
    
    # Create dummy transformer layer
    layer = nn.TransformerEncoderLayer(d_model=768, nhead=12)
    
    # Register hook
    hook_id = hook_manager.register_capture_hook(layer, "test_hook", layer_idx=0)
    
    # Test forward pass
    x = torch.randn(2, 50, 768)  # batch_size=2, seq_len=50, hidden_size=768
    output = layer(x)
    
    # Check captured data
    captured = hook_manager.get_captured_data("test_hook")
    print(f"Captured data shape: {captured['hidden_state'].shape}")
    print(f"Memory usage: {hook_manager.get_memory_usage()}")
    
    # Cleanup
    hook_manager.remove_hooks()

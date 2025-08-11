import time
from collections import defaultdict, deque
from types import SimpleNamespace
from typing import Dict, List

import torch
import torch.nn as nn


class CMRTransformer(nn.Module):
    """Lightweight transformer with memory hook support for tests/demos.

    This intentionally small implementation provides:
    - A tiny Transformer stack (hidden size 128) for quick CPU tests
    - Registerable forward hooks on specific layers (by index)
    - Memory capture buffers and basic stats reporting
    """

    def __init__(self, config, memory_config=None):
        super().__init__()
        self.config = config
        self.memory_config = memory_config or {}

        # Core state
        self.memory_enabled = True
        self.current_sequence_id = 0
        self.layer_hooks: Dict[int, torch.utils.hooks.RemovableHandle] = {}
        self.captured_memory_states: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.memory_config.get('buffer_size', 100))
        )

        # Model dims (keep small for tests)
        self.hidden_size = 128
        self.vocab_size = getattr(config, 'vocab_size', 1000)
        num_layers = (
            getattr(config, 'num_hidden_layers', None)
            or getattr(config, 'n_layer', None)
            or 12
        )

        # Simple embedding + stack of TransformerEncoderLayers
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)
        # Use batch_first=True so tensors are (B, T, C)
        layers: List[nn.Module] = [
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=4,
                dim_feedforward=self.hidden_size * 4,
                batch_first=True,
                activation='gelu'
            )
            for _ in range(num_layers)
        ]
        # Expose as GPT-like container with `.h`
        self.transformer = SimpleNamespace(h=nn.ModuleList(layers))

    # ---- Memory control ----
    def enable_memory(self):
        self.memory_enabled = True

    def disable_memory(self):
        self.memory_enabled = False

    # ---- Hook management ----
    def register_memory_hooks(self):
        """Register forward hooks on specified transformer layers.

        Hooks capture hidden states into per-layer ring buffers.
        """
        target_layers = self.memory_config.get('target_layers', [2, 4])

        # Clean any existing hooks first to avoid duplicates
        self.cleanup_hooks()

        for layer_idx in target_layers:
            if 0 <= layer_idx < len(self.transformer.h):
                layer = self.transformer.h[layer_idx]

                def make_hook(idx):
                    def hook_fn(module, inputs, output):
                        if not self.memory_enabled:
                            return
                        # Output may be Tensor or tuple; handle both
                        hidden = output[0] if isinstance(output, tuple) else output
                        # Store detached clone to avoid holding graph
                        self.captured_memory_states[idx].append({
                            'hidden_state': hidden.detach().clone(),
                            'layer_idx': idx,
                            'sequence_id': self.current_sequence_id,
                            'timestamp': time.time(),
                        })
                    return hook_fn

                handle = layer.register_forward_hook(make_hook(layer_idx))
                self.layer_hooks[layer_idx] = handle

    def cleanup_hooks(self):
        """Remove all registered hooks and clear the registry."""
        to_remove = list(self.layer_hooks.values())
        for handle in to_remove:
            try:
                handle.remove()
            except Exception:
                pass
        self.layer_hooks.clear()

    # ---- Forward ----
    def forward(self, input_ids: torch.LongTensor):
        """Run a forward pass through the tiny transformer.

        Returns a dict with last_hidden_state, captured_memory_states, memory_stats.
        """
        # Simple token embedding
        x = self.embed(input_ids)

        # Sequentially pass through layers
        for layer in self.transformer.h:
            x = layer(x)

        # Update sequence counter after pass
        self.current_sequence_id += 1

        # Prepare outputs
        last_hidden_state = x  # (B, T, 128)

        # Compute memory stats (cumulative)
        total_captured = sum(len(v) for v in self.captured_memory_states.values())
        layers_with = sorted([idx for idx, v in self.captured_memory_states.items() if len(v) > 0])
        memory_stats = {
            'total_captured_states': int(total_captured),
            'layers_with_memory': layers_with,
        }

        return {
            'last_hidden_state': last_hidden_state,
            'captured_memory_states': self.captured_memory_states,
            'memory_stats': memory_stats,
        }

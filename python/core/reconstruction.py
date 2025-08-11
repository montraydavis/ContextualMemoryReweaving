import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import math

# Updated imports for new structure
from models.memory_entry import MemoryEntry
from .memory_buffer import LayeredMemoryBuffer


class LayeredStateReconstructor(nn.Module):
    """
    Layered Latent State Reconstruction (LLSR) mechanism.
    Reconstructs stored memory states into optimized contextual embeddings
    for integration into current forward pass.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        reconstruction_method: str = "hierarchical",
        compressed_size: Optional[int] = None,
        max_memory_tokens: int = 512,
        compression_ratio: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Normalize method names to match tests ("attention" vs "attention_based")
        if reconstruction_method == "attention_based":
            reconstruction_method = "attention"
        self.reconstruction_method = reconstruction_method
        self.compressed_size = compressed_size or int(hidden_size * compression_ratio)
        self.max_memory_tokens = max_memory_tokens
        self.compression_ratio = compression_ratio

        # Per-layer reconstructors, projectors and blenders expected by tests
        self.layer_reconstructors = nn.ModuleDict()
        self.memory_projectors = nn.ModuleDict()
        self.context_blenders = nn.ModuleDict()
        self._build_reconstructors()

        # Positional embeddings for memories
        self.memory_position_embeddings = nn.Embedding(max_memory_tokens, hidden_size)

        # Attention for selecting relevant memories in streaming mode
        self.memory_attention = MemoryAttention(hidden_size, max_memory_tokens)

        # Layer-specific reconstruction weights
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

        # Reconstruction quality metrics
        self.reconstruction_stats = {
            "total_reconstructions": 0,
            "avg_reconstruction_quality": 0.0,
            "memory_utilization": 0.0,
        }

    def _build_reconstructors(self) -> None:
        for layer_idx in range(self.num_layers):
            key = f"layer_{layer_idx}"
            if self.reconstruction_method == "hierarchical":
                self.layer_reconstructors[key] = HierarchicalReconstructor(
                    self.hidden_size, self.compressed_size
                )
            elif self.reconstruction_method == "attention":
                self.layer_reconstructors[key] = AttentionBasedReconstructor(
                    self.hidden_size, self.max_memory_tokens
                )
            elif self.reconstruction_method == "mlp":
                self.layer_reconstructors[key] = MLPReconstructor(
                    self.hidden_size, self.compressed_size
                )
            else:
                raise ValueError(f"Unknown reconstruction method: {self.reconstruction_method}")

            # Identity projector (kept for API completeness)
            self.memory_projectors[key] = nn.Linear(self.hidden_size, self.hidden_size)
            # Per-layer context blender
            self.context_blenders[key] = ContextBlender(self.hidden_size)
    
    def forward(
        self,
        current_hidden_states: torch.Tensor,
        memory_buffer: LayeredMemoryBuffer,
        layer_idx: int,
        sequence_position: int = 0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Reconstruct and integrate memory states with current hidden states.
        
        Args:
            current_hidden_states: Current layer hidden states [batch_size, seq_len, hidden_size]
            memory_buffer: Memory buffer containing stored states
            layer_idx: Current transformer layer index
            sequence_position: Current position in sequence
            
        Returns:
            Tuple of (enhanced_hidden_states, reconstruction_info)
        """
        batch_size, seq_len, hidden_size = current_hidden_states.shape
        
        # Retrieve relevant memories for this layer
        relevant_memories = memory_buffer.retrieve_by_layer(
            layer_idx=layer_idx,
            k=min(self.max_memory_tokens, 50),  # Limit for efficiency
            min_relevance=0.3
        )
        
        if not relevant_memories:
            # No relevant memories, return original states
            return current_hidden_states, {'num_memories_used': 0, 'reconstruction_quality': 0.0}
        
        # Extract memory tensors
        memory_states = torch.stack([mem.hidden_state for mem in relevant_memories])  # [num_memories, hidden_size]
        memory_states = memory_states.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_memories, hidden_size]
        
        # Apply memory attention to select most relevant memories
        attended_memories, attention_weights = self.memory_attention(
            current_hidden_states, memory_states
        )
        
        # Reconstruct memory states using selected method
        reconstructed_memories = self.layer_reconstructors[f"layer_{layer_idx}"](
            attended_memories, current_hidden_states
        )
        
        # Blend reconstructed memories with current states
        layer_weight = torch.sigmoid(self.layer_weights[layer_idx])
        enhanced_states = self.context_blenders[f"layer_{layer_idx}"](
            current_hidden_states, reconstructed_memories, layer_weight
        )
        
        # Apply layer-specific weighting
        final_states = layer_weight * enhanced_states + (1 - layer_weight) * current_hidden_states
        
        # Update reconstruction statistics
        reconstruction_quality = self._compute_reconstruction_quality(
            current_hidden_states, final_states, attention_weights
        )
        
        reconstruction_info = {
            "num_memories_used": len(relevant_memories),
            "reconstruction_quality": reconstruction_quality.item(),
            "layer_weight": layer_weight.item(),
            "attention_entropy": self._compute_attention_entropy(attention_weights),
        }
        
        self.reconstruction_stats['total_reconstructions'] += 1
        self.reconstruction_stats['avg_reconstruction_quality'] = (
            (self.reconstruction_stats['avg_reconstruction_quality'] * 
             (self.reconstruction_stats['total_reconstructions'] - 1) + 
             reconstruction_quality.item()) / self.reconstruction_stats['total_reconstructions']
        )
        
        return final_states, reconstruction_info

    def _prepare_memory_states(
        self, memory_entries: List[MemoryEntry], batch_size: int
    ) -> Optional[torch.Tensor]:
        """Prepare memory states tensor with positional embeddings.

        Returns a tensor of shape [batch_size, num_memories, hidden_size] or None if
        memory_entries is empty.
        """
        if memory_entries is None or len(memory_entries) == 0:
            return None
        device = self.layer_weights.device
        memory_tensor = torch.stack([m.hidden_state.squeeze(0) for m in memory_entries]).to(device)
        num_memories = memory_tensor.shape[0]

        # Add position embeddings
        positions = torch.arange(num_memories, device=device).clamp(max=self.max_memory_tokens - 1)
        pos_emb = self.memory_position_embeddings(positions)  # [num_memories, hidden_size]
        memory_tensor = memory_tensor + pos_emb

        memory_tensor = memory_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        return memory_tensor

    def reconstruct_layer_memories(
        self,
        layer_idx: int,
        memory_entries: List[MemoryEntry],
        current_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """High-level API expected by unit tests.

        Prepares memories, reconstructs with the per-layer module, and blends into
        the current hidden states. Returns enhanced hidden states.
        """
        batch_size = current_hidden_states.shape[0]
        memory_states = self._prepare_memory_states(memory_entries, batch_size)
        if memory_states is None:
            return current_hidden_states

        # Optional projection hook
        proj = self.memory_projectors[f"layer_{layer_idx}"]
        memory_states = proj(memory_states)

        # Reconstruct memory representations (shape preserved: [B, M, H])
        reconstructed = self.layer_reconstructors[f"layer_{layer_idx}"](
            memory_states, current_hidden_states
        )

        # Blend with current states
        layer_weight = torch.sigmoid(self.layer_weights[layer_idx])
        enhanced = self.context_blenders[f"layer_{layer_idx}"](
            current_hidden_states, reconstructed, layer_weight
        )
        return enhanced

    def _compute_reconstruction_quality(self,
                                      original: torch.Tensor,
                                      reconstructed: torch.Tensor,
                                      attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction quality metric."""
        # Cosine similarity between original and reconstructed
        cos_sim = F.cosine_similarity(
            original.view(-1, original.size(-1)),
            reconstructed.view(-1, reconstructed.size(-1)),
            dim=-1
        ).mean()

        # Attention diversity (higher is better)
        attention_diversity = 1.0 - torch.var(attention_weights, dim=-1).mean()

        # Combined quality score
        quality = 0.7 * cos_sim + 0.3 * attention_diversity
        return quality

    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention weights."""
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        entropy = -(attention_weights * torch.log(attention_weights + eps)).sum(dim=-1).mean()
        return entropy.item()

    def get_reconstruction_stats(self) -> Dict:
        """Get reconstruction statistics."""
        return self.reconstruction_stats.copy()

    def reset_stats(self):
        """Reset reconstruction statistics."""
        self.reconstruction_stats = {
            'total_reconstructions': 0,
            'avg_reconstruction_quality': 0.0,
            'memory_utilization': 0.0
        }


class HierarchicalReconstructor(nn.Module):
    """Hierarchical reconstruction using multi-scale processing."""

    def __init__(self, hidden_size: int, compressed_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.compressed_size = compressed_size

        # Compression layers
        self.compress = nn.Sequential(
            nn.Linear(hidden_size, compressed_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Reconstruction layers
        self.reconstruct = nn.Sequential(
            nn.Linear(compressed_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # Multi-scale attention
        # Ensure num_heads divides hidden_size
        num_heads_global = min(8, hidden_size // 16)
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads_global,
            dropout=0.1,
            batch_first=True
        )

        num_heads_local = min(4, hidden_size // 32)
        self.local_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads_local,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, memory_states: torch.Tensor, current_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            memory_states: [batch, num_memories, hidden_size]

        Returns:
            reconstructed: [batch, num_memories, hidden_size]
        """
        batch_size, num_memories, hidden_size = memory_states.shape

        # Step 1: Compress memories for efficiency
        compressed = self.compress(memory_states)
        base_reconstructed = self.reconstruct(compressed)

        # Step 2: Global context attention
        global_enhanced, _ = self.global_attention(
            query=base_reconstructed,
            key=base_reconstructed,
            value=base_reconstructed
        )

        # Step 3: Local refinement attention
        local_enhanced, _ = self.local_attention(
            query=global_enhanced,
            key=global_enhanced,
            value=global_enhanced
        )

        # Step 4: Residual connection
        final_reconstructed = base_reconstructed + local_enhanced

        return final_reconstructed


class AttentionBasedReconstructor(nn.Module):
    """Attention-based reconstruction using cross-attention mechanisms."""

    def __init__(self, hidden_size: int, max_memories: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_memories = max_memories

        # Self-attention over memories
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=min(8, hidden_size // 64),
            dropout=0.1,
            batch_first=True
        )

        # Cross-attention: allow memories to incorporate current state context
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=min(8, hidden_size // 64),
            dropout=0.1,
            batch_first=True,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, memory_states: torch.Tensor, current_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            memory_states: [batch, num_memories, hidden_size]
            current_states: [batch, seq_len, hidden_size]

        Returns:
            reconstructed: [batch, num_memories, hidden_size]
        """
        # Self-attention on memories
        self_attended, _ = self.self_attention(
            query=memory_states, key=memory_states, value=memory_states
        )
        mem = self.norm1(memory_states + self_attended)

        # Cross-attention from memories to current states (optional)
        if current_states is not None:
            cross_attended, _ = self.cross_attention(
                query=mem, key=current_states, value=current_states
            )
            mem = self.norm2(mem + cross_attended)
        else:
            # Keep API behavior consistent
            mem = self.norm2(mem)

        # Feed-forward refinement
        ffn_output = self.ffn(mem)
        reconstructed = self.norm3(mem + ffn_output)

        return reconstructed


class MLPReconstructor(nn.Module):
    """Simple MLP-based reconstruction."""

    def __init__(self, hidden_size: int, compressed_size: int):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, compressed_size),
            nn.ReLU(),
            nn.Linear(compressed_size, compressed_size // 2),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(compressed_size // 2, compressed_size),
            nn.ReLU(),
            nn.Linear(compressed_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # Context integration
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, memory_states: torch.Tensor, current_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            memory_states: [batch, num_memories, hidden_size]

        Returns:
            reconstructed: [batch, num_memories, hidden_size]
        """
        # Encode-decode memories
        encoded = self.encoder(memory_states)
        decoded = self.decoder(encoded)

        # Context-aware gating
        combined = torch.cat([memory_states, decoded], dim=-1)
        gate = self.context_gate(combined)

        # Apply gate
        reconstructed = gate * decoded + (1 - gate) * memory_states

        return reconstructed


class MemoryAttention(nn.Module):
    """Attention mechanism for selecting relevant memories."""

    def __init__(self, hidden_size: int, max_memories: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_memories = max_memories

        # Attention scoring
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        self.scale = math.sqrt(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        queries: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: [batch, seq_len, hidden_size]
            memory_keys: [batch, num_memories, hidden_size]
            memory_values: [batch, num_memories, hidden_size]
            memory_mask: [batch, num_memories]

        Returns:
            attended_output: [batch, seq_len, hidden_size]
            attention_weights: [batch, seq_len, num_memories]
        """
        batch_size, seq_len, hidden_size = queries.shape
        if memory_values is None:
            memory_values = memory_keys
        num_memories = memory_keys.shape[1]

        # Project inputs
        Q = self.query_proj(queries)  # [batch, seq_len, hidden_size]
        K = self.key_proj(memory_keys)  # [batch, num_memories, hidden_size]
        V = self.value_proj(memory_values)  # [batch, num_memories, hidden_size]

        # Compute attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [batch, seq_len, num_memories]

        # Apply memory mask if provided
        if memory_mask is not None:
            scores = scores.masked_fill(memory_mask.unsqueeze(1) == 0, float('-inf'))

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        attended_output = torch.bmm(attention_weights, V)  # [batch, seq_len, hidden_size]

        # Apply dropout to attention weights for regularization
        attention_weights = self.dropout(attention_weights)

        return attended_output, attention_weights


class ContextBlender(nn.Module):
    """Module for blending reconstructed memories with current states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Gating mechanism
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        # Integration transformer
        # Ensure nhead divides hidden_size
        nhead = min(8, hidden_size // 16)
        self.integration_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        current_states: torch.Tensor,
        memory_states: torch.Tensor,
        layer_weight: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """
        Blend current states with reconstructed memory states.

        Args:
            current_states: [batch, seq_len, hidden_size]
            memory_states: [batch, num_memories, hidden_size]
            layer_weight: Scalar mixing weight for the layer

        Returns:
            blended_states: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = current_states.shape
        num_memories = memory_states.shape[1]

        # Step 1: Compute memory summary for gating
        memory_summary = memory_states.mean(dim=1, keepdim=True)  # [batch, 1, hidden_size]
        memory_summary = memory_summary.expand(-1, seq_len, -1)  # [batch, seq_len, hidden_size]

        # Step 2: Compute blend gate
        gate_input = torch.cat([current_states, memory_summary], dim=-1)
        blend_gate = self.memory_gate(gate_input)  # [batch, seq_len, hidden_size]

        # Step 3: Create combined sequence for integration
        # Concatenate current states and memory states
        combined_sequence = torch.cat([current_states, memory_states], dim=1)
        # [batch, seq_len + num_memories, hidden_size]

        # Step 4: Apply integration transformer
        integrated_sequence = self.integration_layer(combined_sequence)

        # Step 5: Extract enhanced current states
        enhanced_current = integrated_sequence[:, :seq_len, :]  # [batch, seq_len, hidden_size]

        # Step 6: Apply gating and output projection
        gated_states = blend_gate * enhanced_current + (1 - blend_gate) * current_states
        blended_states = self.output_proj(gated_states)

        return blended_states

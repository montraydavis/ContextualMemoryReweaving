#!/usr/bin/env python3
"""
Performance optimization components for Contextual Memory Reweaving (CMR).

Implements:
- CMRPerformanceOptimizer
- AdaptiveThresholdManager
- BatchProcessingOptimizer
- MemoryPrefetcher
- ComputationScheduler
- BackgroundOptimizer

These implementations are lightweight and CPU-friendly to keep tests and demos fast.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple, List

import torch


class AdaptiveThresholdManager:
    """
    Manages adaptive threshold adjustments for relevance scoring and related knobs.

    Behavior:
    - Tracks threshold and performance histories
    - Suggests new thresholds based on memory pressure and sequence length
    - Provides simple statistics and reset capability
    """

    def __init__(self,
                 base_threshold: float = 0.5,
                 min_threshold: float = 0.1,
                 max_threshold: float = 0.9):
        self.base_threshold: float = float(base_threshold)
        self.min_threshold: float = float(min_threshold)
        self.max_threshold: float = float(max_threshold)

        self.threshold_history: Deque[Tuple[int, float]] = deque(maxlen=256)
        self.performance_history: Deque[float] = deque(maxlen=256)
        self.memory_usage_history: Deque[int] = deque(maxlen=256)

        self.adjustment_count: int = 0

    def update_threshold(self, layer_idx: int, threshold: float, performance: float) -> None:
        """Record a threshold update and associated performance observation."""
        self.threshold_history.append((layer_idx, float(threshold)))
        self.performance_history.append(float(performance))
        self.adjustment_count += 1

    def get_optimal_threshold(self,
                              input_shape: Any,
                              memory_usage: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """
        Heuristic for computing an adjusted threshold.

        - Increase threshold under high memory pressure
        - Increase threshold for very long sequences
        - Otherwise, return None to keep current threshold
        """
        # Coerce to tuple[int, int] and extract sequence length
        try:
            seq_len = int(input_shape[1])
        except Exception:  # noqa: BLE001 - broad except acceptable for fallback coercion
            input_shape = tuple(int(x) for x in list(input_shape))
            seq_len = int(input_shape[1])

        total_entries = 0
        if isinstance(memory_usage, dict):
            total_entries = int(memory_usage.get('total_entries', 0))
        self.memory_usage_history.append(total_entries)

        adjusted: Optional[float] = None

        # Rule 1: High memory pressure → increase threshold
        if total_entries >= 3000:
            adjusted = min(self.max_threshold, self.base_threshold + 0.1)

        # Rule 2: Very long sequences → modestly increase threshold
        if seq_len >= 512:
            inc = 0.05 if adjusted is None else 0.0  # do not double-add
            adjusted = min(self.max_threshold, (adjusted if adjusted is not None else self.base_threshold) + inc)

        # Boundaries and no negative outcomes
        if adjusted is not None:
            adjusted = max(self.min_threshold, min(self.max_threshold, adjusted))
        return adjusted

    def get_stats(self) -> Dict[str, Any]:
        return {
            'adjustment_count': self.adjustment_count,
            'base_threshold': self.base_threshold,
            'min_threshold': self.min_threshold,
            'max_threshold': self.max_threshold,
            'threshold_history_length': len(self.threshold_history),
            'performance_history_length': len(self.performance_history),
            'memory_history_length': len(self.memory_usage_history),
        }

    def reset_stats(self) -> None:
        self.threshold_history.clear()
        self.performance_history.clear()
        self.memory_usage_history.clear()
        self.adjustment_count = 0


class BatchProcessingOptimizer:
    """
    Optimizes batch inputs primarily by trimming excessive right-side padding when
    a significant fraction of tokens are masked.
    """

    def __init__(self):
        self.optimization_count: int = 0
        self.optimization_savings: float = 0.0
        self.batch_size_history: Deque[int] = deque(maxlen=64)

    def optimize_batch(self,
                       input_batch: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        If the maximum unmasked length across the batch is significantly smaller than
        the current sequence length, truncate to that length.
        """
        if attention_mask is None:
            return input_batch, attention_mask

        batch_size = int(input_batch.size(0))
        seq_len = int(input_batch.size(1))
        self.batch_size_history.append(batch_size)

        # Compute maximum actual length across batch
        actual_lengths = attention_mask.sum(dim=1)
        if isinstance(actual_lengths, torch.Tensor):
            max_actual_length = int(actual_lengths.max().item())
        else:
            max_actual_length = int(max(actual_lengths))

        # Apply heuristic: optimize when at least 20% is padding
        if max_actual_length < int(0.8 * seq_len) and max_actual_length > 0:
            truncate_length = max_actual_length  # aggressive trim to demonstrated content length
            optimized_batch = input_batch[:, :truncate_length]
            optimized_mask = attention_mask[:, :truncate_length]

            # Track savings proportionally
            self.optimization_count += 1
            self.optimization_savings += float(seq_len - truncate_length) / float(max(seq_len, 1))
            return optimized_batch, optimized_mask

        return input_batch, attention_mask

    def get_stats(self) -> Dict[str, Any]:
        avg_batch_size = 0.0
        if self.batch_size_history:
            avg_batch_size = sum(self.batch_size_history) / len(self.batch_size_history)
        return {
            'optimization_count': self.optimization_count,
            'optimization_savings': self.optimization_savings,
            'avg_batch_size': avg_batch_size,
        }

    def reset_stats(self) -> None:
        self.optimization_count = 0
        self.optimization_savings = 0.0
        self.batch_size_history.clear()


class MemoryPrefetcher:
    """Predictive memory prefetching placeholder with simple counters."""

    def __init__(self, _memory_buffer: Optional[Any] = None):
        self._buffer = _memory_buffer
        self.prefetch_cache: Dict[str, Any] = {}
        self.hit_count: int = 0
        self.miss_count: int = 0

    def prefetch_for_input(self, _input_ids: torch.Tensor) -> Dict[str, Any]:
        # Minimal no-op prefetcher for tests/demos
        return {'prefetched': 0}

    def get_stats(self) -> Dict[str, Any]:
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'cache_entries': len(self.prefetch_cache),
        }


class ComputationScheduler:
    """Simple computation scheduler placeholder with a task queue."""

    def __init__(self):
        self.task_queue: Deque[Dict[str, Any]] = deque()
        self.completed_tasks: int = 0
        self.total_execution_time: float = 0.0

    def schedule_task(self, task_type: str, priority: str = 'normal', **kwargs: Any) -> int:
        task = {'type': task_type, 'priority': priority, 'meta': kwargs}
        self.task_queue.append(task)
        return len(self.task_queue)

    def complete_one(self, execution_time: float = 0.0) -> None:
        if self.task_queue:
            self.task_queue.popleft()
            self.completed_tasks += 1
            self.total_execution_time += float(execution_time)

    def get_stats(self) -> Dict[str, Any]:
        return {
            'queued_tasks': len(self.task_queue),
            'completed_tasks': self.completed_tasks,
            'total_execution_time': self.total_execution_time,
        }


class BackgroundOptimizer:
    """
    Background optimizer that periodically performs lightweight maintenance.
    """

    def __init__(self, _cmr_model: Any, interval_seconds: float = 30.0):
        self._cmr_model = _cmr_model
        self.interval_seconds = float(interval_seconds)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _loop(self) -> None:
        while self._running:
            try:
                # Intentionally minimal to avoid affecting tests timing
                time.sleep(self.interval_seconds)
            except Exception:  # noqa: BLE001 - tolerate interruptions
                time.sleep(self.interval_seconds)


class CMRPerformanceOptimizer:
    """
    Coordinates optimization passes around the CMR forward path.

    Responsibilities:
    - Adjust thresholds using AdaptiveThresholdManager
    - Reduce padding via BatchProcessingOptimizer
    - Optionally run background optimization loop
    - Provide optimization statistics
    """

    def __init__(self, cmr_model: Any, optimization_config: Optional[Dict[str, Any]] = None):
        self.cmr_model = cmr_model
        self.config = optimization_config or {}

        # Sub-components
        self.adaptive_thresholds = AdaptiveThresholdManager()
        self.batch_optimizer = BatchProcessingOptimizer()
        self.memory_prefetcher = MemoryPrefetcher(getattr(cmr_model, 'memory_buffer', None))
        self.computation_scheduler = ComputationScheduler()

        # Background optimizer (disabled by default in tests)
        enable_bg = bool(self.config.get('enable_background_optimization', False))
        interval = float(self.config.get('optimization_interval', 30.0))
        self.background_optimizer = BackgroundOptimizer(cmr_model, interval_seconds=interval)
        if enable_bg:
            self.background_optimizer.start()

        # Stats
        self.optimization_stats: Dict[str, Any] = {
            'threshold_adjustments': 0,
            'batch_optimizations': 0,
        }

    def optimize_forward_pass(self,
                              input_ids: torch.Tensor,
                              attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Apply pre-forward optimizations:
        - Adjust relevance threshold heuristically
        - Trim excessive padding in the batch
        """
        opt_info: Dict[str, Any] = {}

        # 1) Adaptive thresholding
        memory_usage: Dict[str, Any] = {}
        try:
            if hasattr(self.cmr_model, 'get_memory_usage'):
                memory_usage = self.cmr_model.get_memory_usage()
        except Exception:  # noqa: BLE001 - broad except acceptable for compatibility with mocks
            memory_usage = {}

        # Ensure a real tuple[int, int] is passed
        suggested = self.adaptive_thresholds.get_optimal_threshold(
            input_shape=(int(input_ids.size(0)), int(input_ids.size(1))),
            memory_usage=memory_usage,
        )
        # Set on model if available
        if suggested is not None:
            try:
                scorer = getattr(self.cmr_model, 'relevance_scorer', None)
                if scorer is not None and hasattr(scorer, 'relevance_threshold'):
                    scorer.relevance_threshold = float(suggested)
                else:
                    # Fallback: set on model if present
                    if hasattr(self.cmr_model, 'relevance_threshold'):
                        self.cmr_model.relevance_threshold = float(suggested)
                self.optimization_stats['threshold_adjustments'] += 1
                self.adaptive_thresholds.update_threshold(layer_idx=0, threshold=float(suggested), performance=1.0)
                opt_info['threshold_adjusted'] = float(suggested)
            except Exception:  # noqa: BLE001 - optional components may be absent
                pass

        # 2) Batch optimization
        optimized_inputs, optimized_mask = self.batch_optimizer.optimize_batch(input_ids, attention_mask)
        if optimized_inputs is not input_ids or (optimized_mask is not attention_mask and optimized_mask is not None):
            self.optimization_stats['batch_optimizations'] += 1
            opt_info['batch_optimized'] = True
            opt_info['original_shape'] = tuple(input_ids.shape)
            opt_info['optimized_shape'] = tuple(optimized_inputs.shape)
        else:
            opt_info['batch_optimized'] = False

        return optimized_inputs, optimized_mask, opt_info

    def get_optimization_stats(self) -> Dict[str, Any]:
        stats = dict(self.optimization_stats)
        stats['adaptive_threshold_stats'] = self.adaptive_thresholds.get_stats()
        stats['batch_optimization_stats'] = self.batch_optimizer.get_stats()
        stats['prefetch_stats'] = self.memory_prefetcher.get_stats()
        stats['scheduler_stats'] = self.computation_scheduler.get_stats()
        return stats

    def reset_stats(self) -> None:
        self.optimization_stats['threshold_adjustments'] = 0
        self.optimization_stats['batch_optimizations'] = 0
        self.adaptive_thresholds.reset_stats()
        self.batch_optimizer.reset_stats()



class CMRPerformanceMonitor:
    """
    Monitor and track CMR performance metrics across captures and reconstructions.
    Minimal implementation aligned with tests in Day 8.
    """

    def __init__(self):
        from collections import defaultdict
        self.metrics = {
            'capture_times': [],
            'reconstruction_times': [],
            'states_stored_per_layer': defaultdict(int),
            'reconstructions_per_layer': defaultdict(int),
            'total_captures': 0,
            'total_reconstructions': 0,
        }

    def record_capture(self, layer_idx: int, states_stored: int, capture_time: float) -> None:
        self.metrics['capture_times'].append(float(capture_time))
        self.metrics['total_captures'] += 1
        self.metrics['states_stored_per_layer'][int(layer_idx)] += int(states_stored)

    def record_reconstruction(self, layer_idx: int, memories_used: int, reconstruction_time: float) -> None:
        self.metrics['reconstruction_times'].append(float(reconstruction_time))
        self.metrics['total_reconstructions'] += 1
        self.metrics['reconstructions_per_layer'][int(layer_idx)] += 1

    def get_stats(self) -> Dict[str, Any]:
        import numpy as np
        stats = {
            'total_captures': self.metrics['total_captures'],
            'total_reconstructions': self.metrics['total_reconstructions'],
            'states_stored_per_layer': dict(self.metrics['states_stored_per_layer']),
            'reconstructions_per_layer': dict(self.metrics['reconstructions_per_layer']),
        }
        if self.metrics['capture_times']:
            stats['avg_capture_time'] = float(np.mean(self.metrics['capture_times']))
        if self.metrics['reconstruction_times']:
            stats['avg_reconstruction_time'] = float(np.mean(self.metrics['reconstruction_times']))
        return stats

    def reset(self) -> None:
        self.__init__()




class ReconstructionIntegrator:
    """
    Lightweight reconstruction integrator used by Day 8 tests.

    Provides simple methods to blend external memory vectors back into the
    current hidden states using either a weighted sum or a gated integration.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        memory_buffer: Optional[Any] = None,
        reconstruction_config: Optional[Dict[str, Any]] = None,
    ):
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.memory_buffer = memory_buffer

        reconstruction_config = reconstruction_config or {}
        # Tests expect these fields/values
        self.integration_method: str = reconstruction_config.get("integration_method", "weighted_sum")
        self.memory_weight: float = float(reconstruction_config.get("memory_weight", 0.3))
        layer_weights_cfg = reconstruction_config.get("layer_weights", {})
        # Store per-layer weights with default of 1.0 when unspecified
        self.layer_weights: Dict[int, float] = {int(k): float(v) for k, v in layer_weights_cfg.items()}

        # Gated integration parameters (lazy simple linear gate)
        self._gate_w: Optional[torch.Tensor] = None
        self._gate_b: Optional[torch.Tensor] = None

        # Statistics
        self._integration_count: int = 0
        self._total_integration_time: float = 0.0

    def _ensure_gate_params(self, device: torch.device) -> None:
        if self._gate_w is None:
            # Concatenate current + memory summary → hidden_size; simple per-dim gating
            self._gate_w = torch.randn(self.hidden_size * 2, self.hidden_size, device=device) * 0.01
        if self._gate_b is None:
            self._gate_b = torch.zeros(self.hidden_size, device=device)

    def _compute_memory_summary(self, memories: List[Dict[str, Any]], batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
        if not memories:
            return torch.zeros(batch, seq_len, self.hidden_size, device=device)

        # Simple average of provided memory vectors broadcast across sequence
        mem_vectors = []
        for m in memories:
            vec = m.get("hidden_state")
            if isinstance(vec, torch.Tensor):
                v = vec.to(device)
                if v.dim() == 1:
                    v = v.view(1, -1)
                mem_vectors.append(v.squeeze(0))
        if not mem_vectors:
            return torch.zeros(batch, seq_len, self.hidden_size, device=device)

        mem_avg = torch.stack(mem_vectors, dim=0).mean(dim=0)  # [H]
        mem_avg = mem_avg.view(1, 1, -1).expand(batch, seq_len, -1)  # [B,S,H]
        return mem_avg

    def integrate_memories(
        self,
        hidden_states: torch.Tensor,
        memories: List[Dict[str, Any]],
        layer_idx: int,
    ) -> torch.Tensor:
        start = time.time()
        device = hidden_states.device
        batch, seq_len, hidden = hidden_states.shape
        assert hidden == self.hidden_size, "Hidden size mismatch"
        # If no memories, return original states unchanged (as tests expect)
        if not memories:
            return hidden_states

        memory_summary = self._compute_memory_summary(memories, batch, seq_len, device)

        if self.integration_method == "gated_integration":
            self._ensure_gate_params(device)
            # Gate = sigmoid([h, m] @ W + b) with per-dim parameters
            concat = torch.cat([hidden_states, memory_summary], dim=-1)  # [B,S,2H]
            # Efficient batched linear using einsum
            gate = torch.sigmoid(torch.einsum("bsh,hk->bsk", concat, self._gate_w) + self._gate_b)
            enhanced = gate * memory_summary + (1.0 - gate) * hidden_states
        else:
            # Default: weighted sum
            layer_w = float(self.layer_weights.get(int(layer_idx), 1.0))
            alpha = max(0.0, min(1.0, self.memory_weight * layer_w))
            enhanced = (1.0 - alpha) * hidden_states + alpha * memory_summary

        self._integration_count += 1
        self._total_integration_time += (time.time() - start)
        return enhanced

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "integration_count": self._integration_count,
            "total_integration_time": self._total_integration_time,
        }

    def reset_statistics(self) -> None:
        self._integration_count = 0
        self._total_integration_time = 0.0

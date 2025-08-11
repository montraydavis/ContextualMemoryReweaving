from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass(eq=False)
class MemoryEntry:
    """Data model representing a captured memory state.

    This mirrors the fields and behavior expected by `core.memory_buffer`
    and reconstruction utilities.
    """

    hidden_state: torch.Tensor
    layer_idx: int
    sequence_id: int
    position_idx: int
    relevance_score: float
    timestamp: float

    access_count: int = 0
    last_access: float = 0.0

    def update_access(self) -> None:
        """Record an access to this entry for LRU-style policies."""
        self.access_count += 1
        self.last_access = time.time()



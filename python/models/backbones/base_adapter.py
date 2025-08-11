from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch
import torch.nn as nn


class BackboneAdapter(ABC, nn.Module):
    """
    Abstract interface that normalizes interactions with different transformer backbones.

    Responsibilities:
    - Load or wrap an underlying transformer model
    - Expose consistent layer access for hook registration
    - Provide a normalized forward that returns last_hidden_state (B, T, H)
    - Surface key properties: hidden_size, num_layers, config
    - Handle device placement
    """

    def __init__(self):
        super().__init__()

    # ---- Properties ----
    @property
    @abstractmethod
    def hidden_size(self) -> int:
        ...

    @property
    @abstractmethod
    def num_layers(self) -> int:
        ...

    @property
    @abstractmethod
    def config(self) -> Any:
        ...

    # ---- Layer access ----
    @abstractmethod
    def get_layers(self) -> List[nn.Module]:
        """Return an ordered list of transformer blocks for hook registration."""

    # ---- Forward ----
    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return last_hidden_state of shape (B, T, H)."""

    # ---- Device placement ----
    @abstractmethod
    def move_to(self, device: str) -> None:
        """Move underlying model to the specified device."""



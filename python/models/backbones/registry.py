from __future__ import annotations

from typing import Any, Dict, Optional, Type

from models.backbones.base_adapter import BackboneAdapter
from models.backbones.mistral_adapter import MistralAdapter
from models.backbones.gemma_adapter import GemmaAdapter


class ModelRegistry:
    """Simple registry mapping names to adapter classes."""

    _registry: Dict[str, Type[BackboneAdapter]] = {
        "mistral": MistralAdapter,
        "mistralai/ministral-8b-instruct-2410": MistralAdapter,
        "mistralai": MistralAdapter,
        # Gemma mappings
        "gemma": GemmaAdapter,
        "google/gemma": GemmaAdapter,
        "google/gemma-3-": GemmaAdapter,
    }

    @classmethod
    def register(cls, key: str, adapter_cls: Type[BackboneAdapter]) -> None:
        cls._registry[key.lower()] = adapter_cls

    @classmethod
    def create(
        cls,
        adapter_name: Optional[str] = None,
        *,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> BackboneAdapter:
        # Prefer explicit adapter_name
        if adapter_name:
            key = adapter_name.lower()
            if key not in cls._registry:
                raise ValueError(f"Unknown adapter '{adapter_name}'")
            adapter_cls = cls._registry[key]
            return adapter_cls(model_name=model_name or "", **kwargs)  # type: ignore[call-arg]

        # Infer from model_name prefix
        if not model_name:
            raise ValueError("Either adapter_name or model_name must be provided")

        lowered = model_name.lower()
        for key, adapter_cls in cls._registry.items():
            if lowered.startswith(key):
                return adapter_cls(model_name=model_name, **kwargs)  # type: ignore[call-arg]

        # Default to MistralAdapter as a safe fallback for current codebase
        return MistralAdapter(model_name=model_name, **kwargs)



from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from models.backbones.base_adapter import BackboneAdapter


class GemmaAdapter(BackboneAdapter):
    """Adapter for Google Gemma models (text-only path).

    This normalizes:
    - Config and model loading
    - Layer access for hook registration
    - Forward returning last_hidden_state (B, T, H)

    Note: Gemma 3 IT is multimodal; we ignore image inputs here and use text-only AutoModel/AutoModelForCausalLM.
    """

    # Class-level caches to avoid repeated HF model/config loads across tests
    _CONFIG_CACHE: Dict[str, Any] = {}
    _MODEL_CACHE: Dict[Tuple[str, bool, Optional[str], bool], nn.Module] = {}

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        device: str = "cpu",
        use_causal_lm: bool = False,
        torch_dtype: Optional[torch.dtype] = torch.float32,
        trust_remote_code: bool = True,
        quantization_8bit: bool = False,
        max_memory_gb: Optional[float] = None,
        config: Optional[Any] = None,
    ):
        super().__init__()

        print(f"Initializing GemmaAdapter with model_name: {model_name}")

        self.model_name = model_name
        self._device = device

        # ---- Load or reuse config ----
        cache_key_config = model_name
        if config is not None:
            self._config = config
            if cache_key_config not in self._CONFIG_CACHE:
                self._CONFIG_CACHE[cache_key_config] = config
        else:
            if cache_key_config in self._CONFIG_CACHE:
                self._config = self._CONFIG_CACHE[cache_key_config]
            else:
                self._config = AutoConfig.from_pretrained(model_name)
                self._CONFIG_CACHE[cache_key_config] = self._config

        common_kwargs: Dict[str, Any] = {
            "config": self._config,
            "trust_remote_code": trust_remote_code,
        }
        if torch_dtype is not None:
            common_kwargs["torch_dtype"] = torch_dtype
        if quantization_8bit:
            common_kwargs["load_in_8bit"] = True
        if max_memory_gb is not None:
            common_kwargs["max_memory"] = {0: f"{max_memory_gb}GB"}

        # ---- Load or reuse model ----
        dtype_key: Optional[str] = None if torch_dtype is None else str(torch_dtype)
        model_cache_key: Tuple[str, bool, Optional[str], bool] = (
            model_name,
            bool(use_causal_lm),
            dtype_key,
            bool(quantization_8bit),
        )

        if model_cache_key in self._MODEL_CACHE:
            self._model = self._MODEL_CACHE[model_cache_key]
        else:
            if use_causal_lm:
                self._model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)  # type: ignore[call-arg]
            else:
                self._model = AutoModel.from_pretrained(model_name, **common_kwargs)  # type: ignore[call-arg]
            self._MODEL_CACHE[model_cache_key] = self._model

        # Cache properties
        hidden_size_val: Optional[int] = getattr(self._config, "hidden_size", None)
        if hidden_size_val is None:
            alt_val: Optional[int] = getattr(self._config, "n_embd", None)
            hidden_size_val = int(alt_val) if alt_val is not None else 1024
        self._hidden_size: int = int(hidden_size_val)

        n_layers_any = getattr(self._config, "num_hidden_layers", None)
        if n_layers_any is None:
            # Fallback: introspect known places
            if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
                n_layers_any = len(list(cast(List[nn.Module], self._model.model.layers)))
            elif hasattr(self._model, "transformer") and hasattr(self._model.transformer, "h"):
                n_layers_any = len(list(cast(List[nn.Module], self._model.transformer.h)))
            elif hasattr(self._model, "encoder") and hasattr(self._model.encoder, "layers"):
                n_layers_any = len(list(cast(List[nn.Module], self._model.encoder.layers)))
            else:
                n_layers_any = 12
        self._num_layers: int = int(n_layers_any)

    # ---- Properties ----
    @property
    def hidden_size(self) -> int:
        return int(self._hidden_size)

    @property
    def num_layers(self) -> int:
        return int(self._num_layers)

    @property
    def config(self) -> Any:
        return self._config

    # ---- Layer access ----
    def get_layers(self) -> List[nn.Module]:
        # Try Gemma/Gemma3 typical layouts
        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            return list(cast(List[nn.Module], self._model.model.layers))
        if hasattr(self._model, "transformer") and hasattr(self._model.transformer, "h"):
            return list(cast(List[nn.Module], self._model.transformer.h))
        if hasattr(self._model, "encoder") and hasattr(self._model.encoder, "layers"):
            return list(cast(List[nn.Module], self._model.encoder.layers))
        return []

    # ---- Forward ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self._model(  # type: ignore[call-arg]
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        raise RuntimeError("Unexpected model outputs: missing last_hidden_state")

    # ---- Device placement ----
    def move_to(self, device: str) -> None:
        self._device = device
        # Defer .to(device) to the parent module's .to() traversal

    @property
    def model(self) -> nn.Module:
        return self._model



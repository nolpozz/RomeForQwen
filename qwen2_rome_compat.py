"""
Qwen2–ROME compatibility patch for HuggingFace transformers.

When ROME (via EasyEdit) runs a forward pass on Qwen2 models to extract layer
activations, the Qwen2DecoderLayer in certain transformers versions can raise:

    TypeError: unsupported operand type(s) for +: 'Tensor' and 'dict'

This occurs because the attention submodule can return a structured object
(e.g. a Cache or dict) where a tensor is expected, causing the residual
connection `hidden_states = residual + hidden_states` to fail. This patch
unwraps the attention output when it is dict/tuple-like, restoring the
expected tensor before the residual add.

The patch does NOT modify the ROME algorithm. It only fixes an incompatibility
in the model's forward pass so ROME can obtain the activations it needs.

See IMPLEMENTATION_NOTES.md for full documentation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch

LOG = logging.getLogger(__name__)

_PATCH_APPLIED = False


def _find_tensor_in(obj: Any, depth: int = 0, max_depth: int = 4) -> Optional[torch.Tensor]:
    """
    Recursively find first 3D tensor (batch, seq, hidden) in nested structures.
    Prefers 3D over 4D (e.g. k/v cache) since we need hidden states for residual add.
    """
    if depth > max_depth:
        return None
    if isinstance(obj, torch.Tensor):
        if obj.dim() == 3:  # (batch, seq, hidden) - ideal
            return obj
        if obj.dim() >= 2:  # fallback for 4D etc
            return obj
        return None
    if isinstance(obj, (tuple, list)):
        for el in obj:
            t = _find_tensor_in(el, depth + 1, max_depth)
            if t is not None:
                return t
        return None
    if isinstance(obj, dict):
        for key in ("hidden_states", "attn_output", "last_hidden_state", 0, "0", "output"):
            if key in obj:
                v = obj[key]
                if v is not None:
                    t = _find_tensor_in(v, depth + 1, max_depth)
                    if t is not None:
                        return t
        for v in obj.values():
            t = _find_tensor_in(v, depth + 1, max_depth)
            if t is not None:
                return t
        return None
    # Object with attributes (e.g. ModelOutput, Cache)
    if hasattr(obj, "__dict__"):
        return _find_tensor_in(vars(obj), depth + 1, max_depth)
    try:
        if hasattr(obj, "items") and callable(obj.items):
            return _find_tensor_in(dict(obj), depth + 1, max_depth)
    except Exception:
        pass
    for attr in ("hidden_states", "attn_output", "last_hidden_state", "output"):
        v = getattr(obj, attr, None)
        if v is not None:
            t = _find_tensor_in(v, depth + 1, max_depth)
            if t is not None:
                return t
    return None


def _ensure_tensor(x: Any) -> Optional[torch.Tensor]:
    """
    Extract a tensor for residual add when a module returns a structured
    object (dict, Cache, ModelOutput, tuple) instead of a plain tensor.
    """
    if isinstance(x, torch.Tensor):
        return x
    return _find_tensor_in(x)


def apply_qwen2_rome_compat_patch() -> None:
    """
    Apply the compatibility patch to Qwen2DecoderLayer so that ROME can run
    on Qwen2/Qwen2.5 models without the Tensor+dict TypeError.

    Call this once before loading or using any Qwen2 model with ROME.
    Idempotent: safe to call multiple times.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    except ImportError as e:
        LOG.warning("Qwen2 compatibility patch skipped: %s", e)
        return

    _original_forward = Qwen2DecoderLayer.forward

    def _patched_forward(
        self: Any,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_kwargs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_value": past_key_value,
            "output_attentions": output_attentions,
            "use_cache": use_cache,
            **kwargs,
        }
        if cache_position is not None:
            attn_kwargs["cache_position"] = cache_position
        raw = self.self_attn(**attn_kwargs)

        attn_output = raw[0] if isinstance(raw, (tuple, list)) else raw
        attn_tensor = _ensure_tensor(attn_output)
        if attn_tensor is None:
            raise TypeError(
                "Qwen2 attention returned a non-tensor first element "
                f"({type(attn_output).__name__}) and it could not be unwrapped. "
                "This may indicate a transformers version mismatch."
            )
        hidden_states = residual + attn_tensor

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs: Tuple[Any, ...] = (hidden_states,)
        raw_tuple = raw if isinstance(raw, (tuple, list)) else (raw,)
        if output_attentions:
            outputs += (raw_tuple[1] if len(raw_tuple) > 1 else None,)
        if use_cache:
            outputs += (raw_tuple[2] if len(raw_tuple) > 2 else None,)
        return outputs

    Qwen2DecoderLayer.forward = _patched_forward
    _PATCH_APPLIED = True
    LOG.info("Applied Qwen2–ROME compatibility patch to Qwen2DecoderLayer")

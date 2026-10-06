"""Truncated Qwen3 backbone wrapper: first N layers, LoRA on the upper layers.

The backbone consumes the order-invariant option layout from ``decider.masking``:
a custom 4-D float additive attention mask ([B, 1, T, T]) plus explicit
position ids ([B, T]) under which every option block starts at the prefix
length P. transformers 5.18 returns pre-built 4-D masks as-is (both for the
``sdpa`` and ``eager`` paths), so no internal causal mask is layered on top —
this is verified by ``tests/test_equivalence.py``.

Weights are loaded in fp32; training runs under bf16 autocast (like
bev-decider) and the head always runs in fp32.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, Qwen3Config, Qwen3Model

from .config import ModelConfig
from .tokenizer import resize_and_init


class TruncatedQwen3Backbone(nn.Module):
    """Qwen3-0.6B truncated to ``num_layers`` layers, with optional LoRA."""

    def __init__(
        self,
        base: Qwen3Model,
        model_cfg: ModelConfig,
        tokenizer=None,
        apply_lora: bool = True,
    ) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone: nn.Module = base
        self._truncate(model_cfg)
        self._apply_final_norm_flag(model_cfg)
        if tokenizer is not None:
            # widen the embeddings for the new special tokens before LoRA wrapping
            resize_and_init(self._base_model(), tokenizer)
        if apply_lora and model_cfg.lora_layers:
            self._attach_lora(model_cfg)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_pretrained(
        cls, model_cfg: ModelConfig, tokenizer=None, apply_lora: bool = True
    ) -> "TruncatedQwen3Backbone":
        base = AutoModel.from_pretrained(
            model_cfg.backbone_name,
            dtype=torch.float32,
            attn_implementation=model_cfg.attn_implementation,
        )
        return cls(base, model_cfg, tokenizer=tokenizer, apply_lora=apply_lora)

    @classmethod
    def from_qwen3_config(
        cls,
        qwen3_config: Qwen3Config,
        model_cfg: ModelConfig,
        tokenizer=None,
        apply_lora: bool = True,
        attn_implementation: str | None = None,
    ) -> "TruncatedQwen3Backbone":
        """Random-initialized backbone (tests); respects num_layers from model_cfg."""
        qwen3_config = Qwen3Config(**{**qwen3_config.to_dict(), "num_hidden_layers": model_cfg.num_layers})
        if attn_implementation:
            qwen3_config._attn_implementation = attn_implementation
        base = Qwen3Model(qwen3_config)
        return cls(base, model_cfg, tokenizer=tokenizer, apply_lora=apply_lora)

    def _truncate(self, model_cfg: ModelConfig) -> None:
        base = self._base_model()
        if model_cfg.num_layers >= len(base.layers):
            if model_cfg.num_layers != len(base.layers):
                raise ValueError(
                    f"num_layers={model_cfg.num_layers} exceeds the backbone's {len(base.layers)} layers"
                )
            return
        base.layers = base.layers[: model_cfg.num_layers]
        base.config.num_hidden_layers = model_cfg.num_layers
        layer_types = getattr(base.config, "layer_types", None)
        if layer_types:
            base.config.layer_types = list(layer_types)[: model_cfg.num_layers]

    def _apply_final_norm_flag(self, model_cfg: ModelConfig) -> None:
        # [ASSUMPTION] implemented by swapping the final RMSNorm for Identity, because
        # Qwen3Model in transformers 5.18 has no pre-final-norm output switch.
        if not model_cfg.apply_final_norm:
            self._base_model().norm = nn.Identity()

    def _attach_lora(self, model_cfg: ModelConfig) -> None:
        lora_cfg = LoraConfig(
            r=model_cfg.lora_r,
            lora_alpha=model_cfg.lora_alpha,
            lora_dropout=model_cfg.lora_dropout,
            target_modules=list(model_cfg.lora_targets),
            layers_to_transform=list(model_cfg.lora_layers),
            layers_pattern=["layers"],  # explicit: peft's lazy default regex misses "layers.N.\..." keys
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

    # ------------------------------------------------------------------ access
    def _base_model(self) -> Qwen3Model:
        """Unwrap peft to the raw Qwen3Model."""
        if hasattr(self.backbone, "get_base_model"):
            return self.backbone.get_base_model()
        return self.backbone  # type: ignore[return-value]

    @property
    def qwen3_config(self) -> Qwen3Config:
        return self._base_model().config

    @property
    def hidden_size(self) -> int:
        return self.qwen3_config.hidden_size

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def lora_parameters(self) -> list[torch.nn.Parameter]:
        return [p for n, p in self.backbone.named_parameters() if "lora_" in n]

    def merge_lora(self) -> None:
        """Merge LoRA deltas into the base weights in place."""
        if hasattr(self.backbone, "merge_and_unload"):
            self.backbone = self.backbone.merge_and_unload()

    def enable_gradient_checkpointing(self) -> None:
        self.backbone.gradient_checkpointing_enable()
        # with LoRA the frozen embedding output otherwise has no grad path
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the truncated backbone over the packed option layout.

        input_ids      [B, T] long
        attention_mask [B, 1, T, T] float additive (0.0 attend / -inf masked)
        position_ids   [B, T] long (options share the prefix start position)
        returns        [B, T, H] float32 hidden states
        """
        if attention_mask.ndim != 4:
            raise ValueError(
                f"attention_mask must be [B, 1, T, T], got shape {tuple(attention_mask.shape)}"
            )
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        return out.last_hidden_state.float()

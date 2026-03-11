"""Decoder-only BrainLM with block-causal attention.

All tokens (masked and unmasked) flow through a single transformer stack.
Tokens are ordered timepoint-major:
    [all_parcels_t0, all_parcels_t1, ..., all_parcels_tN]
and a block lower-triangular attention mask ensures that parcels at time t
can attend to each other and to all earlier timepoints, but never forward.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from brainlm_mae.configuration_brainlm import BrainLMConfig

ACT2FN = {
    "gelu": nn.functional.gelu,
    "relu": nn.functional.relu,
    "swish": nn.SiLU(),
    "gelu_new": nn.functional.gelu,
}


@dataclass
class BrainLMPretrainingOutput(ModelOutput):
    """Output type for :class:`BrainLMForPretraining`.

    Attributes:
        loss: Reconstruction loss on masked positions.
        logits: Tuple ``(pred_logits, last_hidden_state)`` where
            *pred_logits* has shape ``[B, V, num_tokens, patch_size]``
            (voxel-major) and *last_hidden_state* has shape
            ``[B, 1 + num_tokens * V, H]`` (includes CLS at position 0).
        mask: Binary mask ``[B, V, num_tokens]`` (1 = masked).
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[Any] = None
    mask: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding applied along the temporal (token) axis.

    Encodes up to ``max_len`` positions with sinusoidal frequencies.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        position = torch.arange(max_len).unsqueeze(1)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: ``[B, V, num_tokens, H]``
        """
        pos_encoding = self.pe[: x.size(2)]
        pos_encoding = pos_encoding.unsqueeze(0).unsqueeze(0).expand(
            x.shape[0], x.shape[1], -1, -1
        )
        x = x + pos_encoding
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Block-causal attention mask
# ---------------------------------------------------------------------------

def build_block_causal_mask(
    num_tokens: int,
    num_voxels: int,
    device: torch.device,
) -> Tensor:
    """Build a block lower-triangular attention mask for timepoint-major ordering.

    Tokens within the same timepoint (block of ``num_voxels``) can attend to
    each other bidirectionally.  Tokens can attend to all earlier timepoints but
    not to any future timepoint.  A CLS token at position 0 attends to and is
    attended by everything.

    Returns:
        Boolean mask ``[1, 1, S+1, S+1]`` where ``True`` = *can attend*.
        This is converted to a float additive mask before being passed to
        ``scaled_dot_product_attention``.
    """
    S = num_tokens * num_voxels
    block_ids = torch.arange(S, device=device) // num_voxels
    can_attend = block_ids.unsqueeze(0) <= block_ids.unsqueeze(1)

    full = torch.ones(S + 1, S + 1, dtype=torch.bool, device=device)
    full[1:, 1:] = can_attend
    return full.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Embeddings (timepoint-major with mask-token replacement)
# ---------------------------------------------------------------------------

class BrainLMEmbeddings(nn.Module):
    """Project fMRI patches into hidden space with xyz + temporal positional
    embeddings, reorder to timepoint-major, and replace masked positions with
    a learned ``[MASK]`` embedding.

    Output shape: ``[B, 1 + num_tokens * V, H]`` (CLS prepended).
    """

    def __init__(self, config: BrainLMConfig):
        super().__init__()
        self.config = config
        self.num_brain_voxels = config.num_brain_voxels
        self.timepoint_patching_size = config.timepoint_patching_size

        self.signal_projection = nn.Linear(
            config.timepoint_patching_size, config.hidden_size, bias=True
        )
        self.xyz_projection = nn.Linear(3, config.hidden_size, bias=True)
        self.temporal_encoding = PositionalEncoding(d_model=config.hidden_size)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        nn.init.normal_(self.mask_token, std=config.initializer_range)
        nn.init.normal_(self.cls_token, std=config.initializer_range)

    def forward(
        self, signal_vectors: Tensor, xyz_vectors: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            signal_vectors: ``[B, V, T]``
            xyz_vectors: ``[B, V, 3]``

        Returns:
            hidden_states: ``[B, 1 + num_tokens * V, H]``
            mask: ``[B, V, num_tokens]`` (1 = masked, voxel-major)
        """
        B, V, T = signal_vectors.shape
        P = self.timepoint_patching_size
        num_tokens = T // P

        # --- patch & project signal ---
        x = signal_vectors.reshape(B, V, num_tokens, P)
        x = self.signal_projection(x)  # [B, V, num_tokens, H]

        # --- add xyz spatial embedding (broadcast over temporal dim) ---
        xyz = self.xyz_projection(xyz_vectors)  # [B, V, H]
        x = x + xyz.unsqueeze(2)

        # --- add sinusoidal temporal positional encoding ---
        x = self.temporal_encoding(x)  # [B, V, num_tokens, H]

        # --- reorder to timepoint-major ---
        x = x.transpose(1, 2).contiguous()  # [B, num_tokens, V, H]
        x = x.reshape(B, num_tokens * V, -1)  # [B, S, H]

        # --- masking (replace chosen positions with learned mask token) ---
        mask = self._create_mask(B, V, num_tokens, x.device)  # [B, V, num_tokens]
        mask_flat = mask.transpose(1, 2).reshape(B, num_tokens * V)  # timepoint-major
        mask_expanded = mask_flat.unsqueeze(-1)  # [B, S, 1]
        x = x * (1.0 - mask_expanded) + self.mask_token * mask_expanded

        # --- prepend CLS token ---
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 1 + S, H]

        return x, mask

    # ----- masking helpers -----

    def _create_mask(
        self, B: int, V: int, num_tokens: int, device: torch.device
    ) -> Tensor:
        """Select masking strategy and return ``[B, V, num_tokens]`` binary mask."""
        use_forward = (
            self.training
            and self.config.forward_mask_prob > 0
            and torch.rand(1).item() < self.config.forward_mask_prob
        )
        if use_forward:
            return self._forward_mask(B, V, num_tokens, device)
        return self._random_mask(B, V, num_tokens, device)

    @staticmethod
    def _forward_mask(
        B: int, V: int, num_tokens: int, device: torch.device
    ) -> Tensor:
        """Mask only the last temporal token for every parcel."""
        mask = torch.zeros(B, V, num_tokens, device=device)
        mask[:, :, -1] = 1.0
        return mask

    def _random_mask(
        self, B: int, V: int, num_tokens: int, device: torch.device
    ) -> Tensor:
        """Independently mask ``mask_ratio`` fraction of tokens per voxel."""
        num_to_mask = max(1, int(num_tokens * self.config.mask_ratio))
        noise = torch.rand(B, V, num_tokens, device=device)
        ids_sorted = torch.argsort(noise, dim=2)
        mask = torch.zeros(B, V, num_tokens, device=device)
        mask.scatter_(2, ids_sorted[:, :, -num_to_mask:], 1.0)
        return mask


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class BrainLMSelfAttention(nn.Module):
    """Multi-head self-attention using ``scaled_dot_product_attention``."""

    def __init__(self, config: BrainLMConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.dropout_p = config.attention_probs_dropout_prob

    def forward(self, hidden_states: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            hidden_states: ``[B, S, H]``
            attn_mask: additive float mask ``[1, 1, S, S]``
                       (0 = attend, -large = block)
        """
        B, S, _ = hidden_states.shape
        q = self.query(hidden_states).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        dp = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dp)
        return out.transpose(1, 2).reshape(B, S, -1)


class BrainLMSelfOutput(nn.Module):
    def __init__(self, config: BrainLMConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.dropout(self.dense(hidden_states))


class BrainLMIntermediate(nn.Module):
    def __init__(self, config: BrainLMConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.act_fn(self.dense(hidden_states))


class BrainLMFFNOutput(nn.Module):
    """Feed-forward output with residual connection."""

    def __init__(self, config: BrainLMConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, residual: Tensor) -> Tensor:
        return self.dropout(self.dense(hidden_states)) + residual


class BrainLMLayer(nn.Module):
    """Pre-norm transformer block with block-causal attention."""

    def __init__(self, config: BrainLMConfig):
        super().__init__()
        self.attention = BrainLMSelfAttention(config)
        self.attn_output = BrainLMSelfOutput(config)
        self.intermediate = BrainLMIntermediate(config)
        self.ffn_output = BrainLMFFNOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        # pre-norm self-attention with residual
        normed = self.layernorm_before(hidden_states)
        attn_out = self.attention(normed, attn_mask=attn_mask)
        attn_out = self.attn_output(attn_out)
        hidden_states = attn_out + hidden_states

        # pre-norm FFN with residual
        normed = self.layernorm_after(hidden_states)
        ffn_out = self.intermediate(normed)
        hidden_states = self.ffn_output(ffn_out, hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class BrainLMPreTrainedModel(PreTrainedModel):
    """Base class providing weight init and config wiring."""

    config_class = BrainLMConfig
    base_model_prefix = "brainlm"
    main_input_name = "signal_vectors"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


class BrainLMModel(BrainLMPreTrainedModel):
    """Decoder-only transformer with block-causal attention for fMRI data.

    Produces contextualised representations for every token (including masked
    positions).  A CLS token is prepended at position 0.
    """

    def __init__(self, config: BrainLMConfig):
        super().__init__(config)
        self.embeddings = BrainLMEmbeddings(config)
        self.layers = nn.ModuleList(
            [BrainLMLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        num_tokens = config.num_timepoints_per_voxel // config.timepoint_patching_size
        bool_mask = build_block_causal_mask(num_tokens, config.num_brain_voxels, device=torch.device("cpu"))
        float_mask = torch.zeros_like(bool_mask, dtype=torch.float32)
        float_mask.masked_fill_(~bool_mask, float("-inf"))
        self.register_buffer("attn_mask", float_mask, persistent=False)

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        signal_vectors: Tensor,
        xyz_vectors: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            last_hidden_state: ``[B, 1 + S, H]``
            mask: ``[B, V, num_tokens]``
        """
        hidden_states, mask = self.embeddings(signal_vectors, xyz_vectors)
        attn_mask = self.attn_mask.to(dtype=hidden_states.dtype)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attn_mask, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, attn_mask=attn_mask)

        hidden_states = self.layernorm(hidden_states)
        return hidden_states, mask


class BrainLMForPretraining(BrainLMPreTrainedModel):
    """BrainLM pretraining wrapper: model + prediction head + loss."""

    main_input_name = "signal_vectors"

    def __init__(self, config: BrainLMConfig):
        super().__init__(config)
        self.model = BrainLMModel(config)
        self.pred_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(config.hidden_size // 2, config.timepoint_patching_size),
        )
        self.post_init()

    def forward(
        self,
        signal_vectors: Tensor = None,
        xyz_vectors: Tensor = None,
        labels: Tensor = None,
        input_ids: Tensor = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BrainLMPretrainingOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        B, V, T = signal_vectors.shape
        P = self.config.timepoint_patching_size
        num_tokens = T // P

        hidden_states, mask = self.model(signal_vectors, xyz_vectors)

        logits = self.pred_head(hidden_states[:, 1:, :])  # [B, num_tokens*V, P]

        # reshape from timepoint-major to voxel-major
        logits = logits.reshape(B, num_tokens, V, P).transpose(1, 2).contiguous()
        # logits: [B, V, num_tokens, P]

        signal_target = signal_vectors.reshape(B, V, num_tokens, P)
        loss = self._compute_loss(signal_target, logits, mask)

        if not return_dict:
            return (loss, (logits, hidden_states), mask)

        return BrainLMPretrainingOutput(
            loss=loss,
            logits=(logits, hidden_states),
            mask=mask,
        )

    def _compute_loss(self, target: Tensor, pred: Tensor, mask: Tensor) -> Tensor:
        """MSE or MAE on masked positions only.

        Args:
            target: ``[B, V, num_tokens, P]``
            pred: ``[B, V, num_tokens, P]``
            mask: ``[B, V, num_tokens]`` (1 = masked)
        """
        mask_expanded = mask.unsqueeze(-1).expand_as(pred)
        if self.config.loss_fn == "mse":
            return (((pred - target) ** 2) * mask_expanded).sum() / mask_expanded.sum()
        if self.config.loss_fn == "mae":
            return (torch.abs(pred - target) * mask_expanded).sum() / mask_expanded.sum()
        raise NotImplementedError(f"Unknown loss function: {self.config.loss_fn}")

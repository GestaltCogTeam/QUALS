import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Union, List
import math
import collections.abc
import numpy as np


def meshgrid(*tensors: Union[torch.Tensor, List[torch.Tensor]], indexing: str = "ij"):
    """
    Wrapper around torch.meshgrid to avoid warning messages about the introduced `indexing` argument.

    Reference: https://pytorch.org/docs/1.13/generated/torch.meshgrid.html
    """
    return torch.meshgrid(*tensors, indexing=indexing)

def window_partition1D(input_feature: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partitions the given input into windows.
    """
    B, H, W, C = input_feature.shape
    # print(B,H,W,C,window_size)
    assert W == 1, "This 1D version only supports W=1 inputs."

    assert H % window_size == 0, f"H={H} must be divisible by window_size={window_size}."

    num_windows = H // window_size
    # reshape to [B, num_windows, window_size, 1, C]
    windows = input_feature.view(B, num_windows, window_size, 1, C)

    # merge batch and window index -> [B*num_windows, window_size, 1, C]
    windows = windows.reshape(B * num_windows, window_size, 1, C)

    return windows

def window_reverse1D(windows, window_size, H):
    """
    Reverse 1D windows back to shape [B, H, 1, C].
    Input:
        windows: [B * num_windows, window_size, 1, C]
    Output:
        input_feature: [B, H, 1, C]
    """
    B_times_win, window_h, _, C = windows.shape
    assert window_h == window_size, "window_size mismatch."

    num_windows = H // window_size
    B = B_times_win // num_windows

    # reshape back: [B, num_windows, window_size, 1, C]
    windows = windows.view(B, num_windows, window_size, 1, C)

    # merge windows back: [B, H, 1, C]
    input_feature = windows.reshape(B, H, 1, C)

    return input_feature


class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """Frozen sinusoidal positions used by Hugging Face RoFormer RoPE."""

    def __init__(self, num_positions: int, embedding_dim: int) -> None:
        super().__init__(num_positions, embedding_dim, _freeze=True)
        self.weight = nn.Parameter(self.create_weight(), requires_grad=False)

    def create_weight(self):
        n_pos, dim = self.weight.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out = torch.empty(n_pos, dim, dtype=self.weight.dtype, requires_grad=False)
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        return out

    @torch.no_grad()
    def forward(self, seq_len: int) -> torch.Tensor:
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(position_ids)


class SwinAttentionBlock(nn.Module):
    """
    Unified Swin attention block.
    If kv_states is None → self-attention
    If kv_states is not None → cross-attention
    """
    def __init__(self, config, dim, num_heads, window_size, use_rope=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.all_head_dim = dim
        self.use_rope = use_rope
        self.rotary_value = getattr(config, "rotary_value", False)
        self.use_relative_position_bias = getattr(config, "use_relative_position_bias", True)

        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable)
            else (window_size, 1)
        )

        # --- Relative Position Bias ---
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2*self.window_size[0]-1) * (2*self.window_size[1]-1),
                num_heads
            ),
            requires_grad=self.use_relative_position_bias,
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1,2,0).contiguous()
        relative_coords[:,:,0] += self.window_size[0]-1
        relative_coords[:,:,1] += self.window_size[1]-1
        relative_coords[:,:,0] *= (2*self.window_size[1]-1)
        self.register_buffer(
            "relative_position_index",
            relative_coords.sum(-1)
        )
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
            self.window_size[0] * self.window_size[1],
            self.head_dim,
        ) if self.use_rope else None

        # --- Q/K/V projections (shared for self and cross) ---
        self.query = nn.Linear(dim, dim, bias=config.qkv_bias)
        self.key   = nn.Linear(dim, dim, bias=config.qkv_bias)
        self.value = nn.Linear(dim, dim, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    @staticmethod
    def _apply_rotary_position_embedding(sinusoidal_pos, layer):
        # Hugging Face RoFormer layout:
        # sin/cos are packed into the last dimension and then repeated pairwise.
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        rotate_half_layer = torch.stack([-layer[..., 1::2], layer[..., ::2]], dim=-1).reshape_as(layer)
        return layer * cos_pos + rotate_half_layer * sin_pos

    def apply_rotary_position_embeddings(self, query_layer, key_layer, value_layer=None):
        q_pos = self.embed_positions(query_layer.shape[-2])[None, None, :, :]
        query_layer = self._apply_rotary_position_embedding(q_pos, query_layer)

        k_pos = self.embed_positions(key_layer.shape[-2])[None, None, :, :]
        key_layer = self._apply_rotary_position_embedding(k_pos, key_layer)

        if value_layer is not None:
            v_pos = self.embed_positions(value_layer.shape[-2])[None, None, :, :]
            value_layer = self._apply_rotary_position_embedding(v_pos, value_layer)
            return query_layer, key_layer, value_layer

        return query_layer, key_layer

    def forward(
        self,
        hidden_states,          # Q
        kv_states=None,         # KV (None → self-attn)
        attention_mask=None,
        output_attentions=False
    ):
        """
        hidden_states : (B, Lq, C)
        kv_states     : (B, Lkv, C) or None
        """

        # ----------------------
        # 1. Q projection
        # ----------------------
        B, Lq, C = hidden_states.shape
        q = self.query(hidden_states)
        q = q.view(B, Lq, self.num_heads, self.head_dim).permute(0,2,1,3)

        # ----------------------
        # 2. K/V projection
        # ----------------------


        if kv_states is None:
            kv_states = hidden_states   # self-attn
        '''else:
            print('Cross Attention ! ')'''
        
        B, Lkv, _ = kv_states.shape
        k = self.key(kv_states).view(B, Lkv, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = self.value(kv_states).view(B, Lkv, self.num_heads, self.head_dim).permute(0,2,1,3)

        if self.use_rope:
            if self.rotary_value:
                q, k, v = self.apply_rotary_position_embeddings(q, k, v)
            else:
                q, k = self.apply_rotary_position_embeddings(q, k)

        # ----------------------
        # 3. Attention scores
        # ----------------------
        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        # --- relative position bias ---
        if self.use_relative_position_bias:
            rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
            rel_bias = rel_bias.view(
                self.window_size[0]*self.window_size[1],
                self.window_size[0]*self.window_size[1],
                -1
            )
            rel_bias = rel_bias.permute(2,0,1).unsqueeze(0)
            attn_scores = attn_scores + rel_bias

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # ----------------------
        # 4. Softmax
        # ----------------------
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # ----------------------
        # 5. Attention output
        # ----------------------
        context = torch.matmul(attn_probs, v)
        context = context.permute(0,2,1,3).contiguous().view(B, Lq, C)

        return (context, attn_probs) if output_attentions else (context,)
    
    
class SwinAttentionOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class SwinAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size, use_rope=False):
        super().__init__()
        self.block = SwinAttentionBlock(config, dim, num_heads, window_size, use_rope=use_rope)
        self.output = SwinAttentionOutput(config, dim)

    def forward(
        self,
        hidden_states,       # Q
        kv_states=None,      # None = self-attention
        attention_mask=None,
        output_attentions=False
    ):
        attn_outputs = self.block(
            hidden_states,
            kv_states=kv_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        context = attn_outputs[0]

        # 这里保持你原样：output(context, hidden_states)
        projected = self.output(context, hidden_states)

        return (projected,) + attn_outputs[1:]



def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->Swin
class SwinDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"

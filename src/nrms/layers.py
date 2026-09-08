"""The two attention layers NRMS is built from, ported from
ebnerd-benchmark's `src/ebrec/models/newsrec/layers.py` (AttLayer2 and
SelfAttention, themselves from Microsoft's `recommenders`).

Ported faithfully, including the choices we would not otherwise make:

  * Q/K/V projections carry no bias, because their `add_weight` calls create
    only the three matrices.
  * No masking. NRMS invokes their SelfAttention with three identical inputs
    and no sequence-length arguments, so the masking branch never fires --
    padding tokens and padded history slots participate in attention. We match
    that rather than silently "fixing" it, since it would change the baseline
    we are meant to reproduce.

The one deliberate numerical deviation is documented on AdditiveAttention.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    """Their `AttLayer2`: a learned query attends over a sequence and collapses
    it to one vector.

        a = tanh(x W + b) q            -> (B, S)
        alpha = exp(a) / (sum exp(a) + eps)
        out = sum_s alpha_s * x_s      -> (B, D_in)

    Output width is the INPUT width, not `hidden_dim` -- the layer returns a
    weighted sum of its inputs and `hidden_dim` is only the projection size.

    Deviation: they compute `exp(a)` directly, which overflows to inf for
    large logits and produces NaNs. We subtract the row max first. This is the
    standard stabilisation and is mathematically identical up to the epsilon,
    whose effect only shrinks (after subtraction the denominator is >= 1).
    """

    def __init__(self, input_dim: int, hidden_dim: int, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        self.W = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        self.q = nn.Parameter(torch.empty(hidden_dim, 1))
        # Glorot uniform on W and q, zeros on b -- their initializers.
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.q)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, S, D_in)
        attention = torch.tanh(inputs @ self.W + self.b)  # (B, S, H)
        attention = (attention @ self.q).squeeze(-1)  # (B, S)

        attention = torch.exp(attention - attention.max(dim=-1, keepdim=True).values)
        weights = attention / (attention.sum(dim=-1, keepdim=True) + self.eps)

        return (inputs * weights.unsqueeze(-1)).sum(dim=1)  # (B, D_in)


class MultiHeadSelfAttention(nn.Module):
    """Their `SelfAttention`: scaled dot-product self-attention.

        Q, K, V = x WQ, x WK, x WV     each (B, S, heads * head_dim)
        reshape -> (B, S, heads, head_dim) -> permute (B, heads, S, head_dim)
        A = softmax(Q K^T / sqrt(head_dim))
        O = A V -> permute back -> (B, S, heads * head_dim)
    """

    def __init__(self, input_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.output_dim = num_heads * head_dim

        self.WQ = nn.Parameter(torch.empty(input_dim, self.output_dim))
        self.WK = nn.Parameter(torch.empty(input_dim, self.output_dim))
        self.WV = nn.Parameter(torch.empty(input_dim, self.output_dim))
        for weight in (self.WQ, self.WK, self.WV):
            nn.init.xavier_uniform_(weight)

    def _project(self, inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = inputs.shape
        projected = inputs @ weight  # (B, S, heads * head_dim)
        projected = projected.view(batch, seq, self.num_heads, self.head_dim)
        return projected.permute(0, 2, 1, 3)  # (B, heads, S, head_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, S, D_in)
        batch, seq, _ = inputs.shape
        query = self._project(inputs, self.WQ)
        key = self._project(inputs, self.WK)
        value = self._project(inputs, self.WV)

        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)

        out = attention @ value  # (B, heads, S, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(batch, seq, self.output_dim)
        return out

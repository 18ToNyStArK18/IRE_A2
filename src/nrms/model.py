"""NRMS (Wu et al., 2019), matching ebnerd-benchmark's `NRMSModel`.

    news encoder:  Embedding -> Dropout -> SelfAttention -> Dropout
                   -> AdditiveAttention                     (B, T) -> (B, 400)
    user encoder:  news encoder applied per history slot -> SelfAttention
                   -> AdditiveAttention                  (B, H, T) -> (B, 400)
    score:         dot(candidate vector, user vector)

Their Keras build produces two graphs sharing weights: `model` (softmax over
npratio+1 candidates, categorical cross-entropy) and `scorer` (sigmoid on a
single candidate). One torch module covers both -- `forward` returns raw dot
products, which are the softmax logits for training, and `score` applies the
sigmoid for parity. Sigmoid is monotone, so it leaves AUC/MRR/nDCG unchanged
and evaluation ranks on the raw logits.

The news encoder is shared by reference between the candidate path and the
user encoder, as in their graph -- there is exactly one set of news-encoder
weights.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.nrms.layers import AdditiveAttention, MultiHeadSelfAttention


class NewsEncoder(nn.Module):
    """(B, title_size) token ids -> (B, num_heads * head_dim) article vector."""

    def __init__(
        self,
        embedding_weights: np.ndarray,
        num_heads: int,
        head_dim: int,
        attention_hidden_dim: int,
        dropout: float,
        padding_idx: int | None = None,
    ):
        super().__init__()
        weights = torch.as_tensor(embedding_weights, dtype=torch.float32)
        # trainable, as in their Embedding(..., trainable=True): the pretrained
        # matrix is an initialisation, not a frozen feature extractor.
        #
        # padding_idx defaults to None, and the baseline leaves it that way.
        # `from_pretrained` does NOT zero that row (it skips reset_parameters
        # when weights are supplied) -- it only zeroes the row's gradient, which
        # pins the pad vector at its pretrained value forever. Their Keras
        # Embedding trains every row, and since we reproduce their absence of
        # attention masking (see layers.py), pad tokens fill most of the title
        # slots for short titles and participate fully in attention: that vector
        # is load-bearing, not inert. Exposed as a parameter so the choice stays
        # expressible, but do not set it when reproducing the baseline.
        self.embedding = nn.Embedding.from_pretrained(
            weights, freeze=False, padding_idx=padding_idx
        )
        self.dropout_in = nn.Dropout(dropout)
        self.self_attention = MultiHeadSelfAttention(weights.shape[1], num_heads, head_dim)
        self.dropout_out = nn.Dropout(dropout)
        self.additive_attention = AdditiveAttention(
            self.self_attention.output_dim, attention_hidden_dim
        )
        self.output_dim = self.self_attention.output_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout_in(self.embedding(tokens))
        attended = self.dropout_out(self.self_attention(embedded))
        return self.additive_attention(attended)


class UserEncoder(nn.Module):
    """(B, history_size, title_size) -> (B, news_dim) user vector.

    The per-slot application of the news encoder is their
    `TimeDistributed(titleencoder)`; flattening to (B*H, T) and reshaping back
    is the torch equivalent. No dropout here, matching their graph.
    """

    def __init__(
        self,
        news_encoder: NewsEncoder,
        num_heads: int,
        head_dim: int,
        attention_hidden_dim: int,
    ):
        super().__init__()
        self.news_encoder = news_encoder
        self.self_attention = MultiHeadSelfAttention(
            news_encoder.output_dim, num_heads, head_dim
        )
        self.additive_attention = AdditiveAttention(
            self.self_attention.output_dim, attention_hidden_dim
        )
        self.output_dim = self.self_attention.output_dim

    def forward(self, history_tokens: torch.Tensor) -> torch.Tensor:
        batch, history_size, title_size = history_tokens.shape
        flat = history_tokens.reshape(batch * history_size, title_size)
        vectors = self.news_encoder(flat).reshape(batch, history_size, -1)
        attended = self.self_attention(vectors)
        return self.additive_attention(attended)


class FreshnessHead(nn.Module):
    """(log1p(age_hours), known) -> an additive adjustment to the logit.

    Deliberately additive on the logit rather than concatenated into the news
    vector: it keeps the dot-product scorer that defines NRMS untouched, so the
    ablation isolates "does knowing article age help" rather than also changing
    how the user and candidate representations interact.

    The output layer is zero-initialised, so at step 0 the adjustment is exactly
    0 and the model's logits are bit-identical to the baseline's. The arm can
    only depart from the baseline by learning, which is what makes the
    comparison clean -- and it is why the flag cannot accidentally look like an
    improvement simply from perturbed initialisation.

    ~50 parameters against ~192M in the XLM-R embedding matrix, so a gain here
    cannot be dismissed as added capacity.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, log_age: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        features = torch.stack([log_age, known], dim=-1)
        return self.mlp(features).squeeze(-1)


class NRMS(nn.Module):
    def __init__(
        self,
        embedding_weights: np.ndarray,
        num_heads: int,
        head_dim: int,
        attention_hidden_dim: int,
        dropout: float,
        padding_idx: int | None = None,
        freshness: bool = False,
        freshness_hidden_dim: int = 16,
    ):
        super().__init__()
        self.news_encoder = NewsEncoder(
            embedding_weights, num_heads, head_dim, attention_hidden_dim, dropout, padding_idx
        )
        self.user_encoder = UserEncoder(
            self.news_encoder, num_heads, head_dim, attention_hidden_dim
        )
        self.output_dim = self.news_encoder.output_dim

        # Constructed only when enabled, and last: building it unconditionally
        # would draw from the RNG even when disabled, shifting every subsequent
        # parameter's initialisation and quietly stopping the baseline arm from
        # reproducing the baseline runs it is being compared against.
        self.freshness_head = FreshnessHead(freshness_hidden_dim) if freshness else None

    def forward(
        self,
        history_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_log_age: torch.Tensor | None = None,
        candidate_age_known: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(B, H, T), (B, C, T) -> (B, C) raw dot-product logits.

        The two age tensors are (B, C) and ignored unless a freshness head is
        attached, so the baseline and freshness arms share one call signature.
        """
        user_vector = self.user_encoder(history_tokens)  # (B, D)

        batch, n_candidates, title_size = candidate_tokens.shape
        flat = candidate_tokens.reshape(batch * n_candidates, title_size)
        candidate_vectors = self.news_encoder(flat).reshape(batch, n_candidates, -1)

        # One user vector broadcast across that impression's candidates.
        logits = torch.einsum("bcd,bd->bc", candidate_vectors, user_vector)

        if self.freshness_head is not None:
            if candidate_log_age is None or candidate_age_known is None:
                raise ValueError("freshness head is attached but candidate age tensors were not supplied")
            logits = logits + self.freshness_head(candidate_log_age, candidate_age_known)
        return logits

    def score(
        self,
        history_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_log_age: torch.Tensor | None = None,
        candidate_age_known: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Their `scorer` head. Monotone in `forward`, so ranking is unchanged."""
        return torch.sigmoid(
            self.forward(history_tokens, candidate_tokens, candidate_log_age, candidate_age_known)
        )

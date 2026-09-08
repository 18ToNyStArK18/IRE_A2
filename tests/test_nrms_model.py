"""Shape and gradient checks for the NRMS modules, on random tensors.

No dataset, no pretrained download, no training -- this exists to catch a
transposed reshape or a mis-sized projection before GPU time is spent on it.
Skipped entirely where torch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.nrms.layers import AdditiveAttention, MultiHeadSelfAttention  # noqa: E402
from src.nrms.model import NRMS, NewsEncoder, UserEncoder  # noqa: E402

VOCAB, EMB_DIM = 40, 16
HEADS, HEAD_DIM, HIDDEN = 4, 8, 12
TITLE, HISTORY, BATCH, CANDIDATES = 6, 5, 3, 4
NEWS_DIM = HEADS * HEAD_DIM


@pytest.fixture
def embedding_weights():
    return np.random.default_rng(0).normal(size=(VOCAB, EMB_DIM)).astype(np.float32)


def _model(embedding_weights, dropout=0.0):
    return NRMS(embedding_weights, HEADS, HEAD_DIM, HIDDEN, dropout, padding_idx=0)


def test_additive_attention_preserves_input_width():
    """Its output width is the INPUT width, not the hidden dim -- it returns a
    weighted sum of its inputs. Getting this backwards silently mis-sizes the
    layer that follows."""
    layer = AdditiveAttention(input_dim=NEWS_DIM, hidden_dim=HIDDEN)
    assert layer(torch.randn(BATCH, HISTORY, NEWS_DIM)).shape == (BATCH, NEWS_DIM)


def test_additive_attention_weights_sum_to_one():
    layer = AdditiveAttention(input_dim=1, hidden_dim=HIDDEN)
    ones = torch.ones(1, 7, 1)
    # a convex combination of identical vectors is that vector
    assert layer(ones).item() == pytest.approx(1.0, abs=1e-5)


def test_additive_attention_survives_large_logits():
    """Their implementation calls exp() directly and overflows to inf here; the
    port subtracts the row max first."""
    layer = AdditiveAttention(input_dim=4, hidden_dim=HIDDEN)
    out = layer(torch.full((2, 5, 4), 1e4))
    assert torch.isfinite(out).all()


def test_self_attention_output_is_heads_times_head_dim():
    layer = MultiHeadSelfAttention(EMB_DIM, HEADS, HEAD_DIM)
    assert layer(torch.randn(BATCH, TITLE, EMB_DIM)).shape == (BATCH, TITLE, NEWS_DIM)


def test_news_encoder_shape(embedding_weights):
    encoder = NewsEncoder(embedding_weights, HEADS, HEAD_DIM, HIDDEN, 0.0, padding_idx=0)
    tokens = torch.randint(0, VOCAB, (BATCH, TITLE))
    assert encoder(tokens).shape == (BATCH, NEWS_DIM)
    assert encoder.output_dim == NEWS_DIM


def test_user_encoder_shape(embedding_weights):
    news = NewsEncoder(embedding_weights, HEADS, HEAD_DIM, HIDDEN, 0.0, padding_idx=0)
    user = UserEncoder(news, HEADS, HEAD_DIM, HIDDEN)
    tokens = torch.randint(0, VOCAB, (BATCH, HISTORY, TITLE))
    assert user(tokens).shape == (BATCH, NEWS_DIM)


def test_nrms_forward_shape(embedding_weights):
    model = _model(embedding_weights)
    history = torch.randint(0, VOCAB, (BATCH, HISTORY, TITLE))
    candidates = torch.randint(0, VOCAB, (BATCH, CANDIDATES, TITLE))
    assert model(history, candidates).shape == (BATCH, CANDIDATES)


def test_nrms_shares_one_news_encoder(embedding_weights):
    """Their Keras graph reuses a single news encoder for both the candidate
    path and the history path; two copies would double the parameters and
    change the model."""
    model = _model(embedding_weights)
    assert model.news_encoder is model.user_encoder.news_encoder


def test_nrms_handles_variable_candidate_counts(embedding_weights):
    model = _model(embedding_weights)
    history = torch.randint(0, VOCAB, (BATCH, HISTORY, TITLE))
    for n_candidates in (1, 2, 9):
        candidates = torch.randint(0, VOCAB, (BATCH, n_candidates, TITLE))
        assert model(history, candidates).shape == (BATCH, n_candidates)


def test_score_is_monotone_in_forward(embedding_weights):
    """Evaluation ranks on raw logits; the sigmoid head exists only for parity
    with their scorer, so it must not reorder anything."""
    model = _model(embedding_weights).eval()
    history = torch.randint(0, VOCAB, (BATCH, HISTORY, TITLE))
    candidates = torch.randint(0, VOCAB, (BATCH, CANDIDATES, TITLE))
    with torch.no_grad():
        logits = model(history, candidates)
        scores = model.score(history, candidates)
    assert torch.equal(logits.argsort(dim=-1), scores.argsort(dim=-1))


def test_backward_reaches_the_embedding(embedding_weights):
    """A finite loss whose gradient never reaches the embedding would train
    nothing but the attention layers."""
    model = _model(embedding_weights)
    history = torch.randint(1, VOCAB, (BATCH, HISTORY, TITLE))
    candidates = torch.randint(1, VOCAB, (BATCH, CANDIDATES, TITLE))
    labels = torch.randint(0, CANDIDATES, (BATCH,))

    loss = torch.nn.CrossEntropyLoss()(model(history, candidates), labels)
    assert torch.isfinite(loss)
    loss.backward()

    grad = model.news_encoder.embedding.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0

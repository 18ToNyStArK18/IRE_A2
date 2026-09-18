"""Scoring NRMS with a cached article-vector table, for volumes the
per-impression path cannot reach (A2 Q5 submissions, and the honest answer to
Q4's serving question for NRMS).

`evaluate.py` re-encodes every candidate and every history slot of every
impression. That is the right shape for a labelled split of 25k-73k impressions,
but the competition test sets are 2.37M and 13.5M impressions: measured at 282
and 111 impressions/s, that is ~13.5 h and ~6 h of pure GPU time.

`NewsEncoder`'s output depends only on an article's own tokens -- there is no
context, no user, no position -- so encoding the catalogue ONCE and gathering
vectors by row index is *exactly* the same computation, not an approximation.
What remains per impression is a gather of HISTORY_SIZE vectors, the user
encoder's self- and additive attention, and one dot product per candidate.
`scripts/verify_fast_scorer.py` asserts the equivalence numerically by
reproducing the committed metrics of both trained arms.

Row 0 of the table is the padding article: the news encoder applied to the
all-zero token row, which is what the model saw for padded history slots during
training (`layers.py` deliberately does no attention masking, so that vector is
load-bearing rather than inert).

Two callers index it differently, so the offset is explicit rather than assumed.
A submission catalogue is 0-based with -1 for "not in catalogue", so it prepends
a pad row and passes `row_offset=1`. The processed pipeline's token matrix
already reserves code 0 for padding, so it passes `prepend_pad=False` and
`row_offset=0` and indexes with article codes directly.
"""

from __future__ import annotations

import numpy as np
import torch

from src.nrms import config as nrms_config


@torch.no_grad()
def encode_catalogue(
    model, token_matrix: np.ndarray, device, batch_size: int = 512, prepend_pad: bool = True
) -> torch.Tensor:
    """(n_articles, title_size) token ids -> (n_articles + 1, news_dim) vectors.

    With `prepend_pad` (the default) index 0 is the pad article and catalogue row
    i lives at i + 1. Pass False for a matrix that already carries the pad row.
    """
    model = model.to(device).eval()
    tokens = np.asarray(token_matrix, dtype=np.int64)
    if prepend_pad:
        tokens = np.concatenate([np.zeros((1, tokens.shape[1]), dtype=np.int64), tokens], axis=0)

    out = torch.empty((len(tokens), model.output_dim), dtype=torch.float32, device=device)
    for start in range(0, len(tokens), batch_size):
        batch = torch.from_numpy(tokens[start : start + batch_size]).to(device)
        out[start : start + batch.shape[0]] = model.news_encoder(batch).float()
    return out


@torch.no_grad()
def user_vectors(
    model, article_vectors: torch.Tensor, history_rows: np.ndarray, row_offset: int = 1
) -> torch.Tensor:
    """(B, history_size) catalogue rows -> (B, news_dim) user vectors.

    Skips the news encoder by gathering precomputed vectors, then runs the rest
    of `UserEncoder.forward` unchanged.
    """
    rows = torch.from_numpy(np.asarray(history_rows, dtype=np.int64) + row_offset).to(article_vectors.device)
    vectors = article_vectors[rows]  # (B, H, D)
    attended = model.user_encoder.self_attention(vectors)
    return model.user_encoder.additive_attention(attended)


@torch.no_grad()
def score_chunk(
    model,
    article_vectors: torch.Tensor,
    history_rows: np.ndarray,
    cand_rows: np.ndarray,
    offsets: np.ndarray,
    log_age: np.ndarray | None = None,
    known: np.ndarray | None = None,
    batch_pairs: int = 200_000,
    row_offset: int = 1,
) -> np.ndarray:
    """Flat per-candidate scores for one chunk, aligned with `cand_rows`.

    Candidates stay ragged (flat array + offsets) rather than being padded to the
    widest impression: padding a 13.5M-impression run to its widest in-view list
    would multiply the work several times over for rows that are then discarded.

    Batching is by (impression, candidate) PAIRS, not impressions, because the
    gather is (pairs, news_dim) float32 and that is what the GPU has to hold: a
    fixed impression count silently scales with list width, and 8192 MIND
    impressions at ~37 candidates asked the allocator for 501 MB on an 8 GB card.
    """
    device = article_vectors.device
    widths = np.diff(offsets)
    scores = np.empty(len(cand_rows), dtype=np.float32)

    start = 0
    while start < len(widths):
        # Take as many impressions as fit the pair budget, but always at least
        # one, so a single very wide impression cannot stall the loop.
        stop = int(np.searchsorted(offsets, offsets[start] + batch_pairs, side="right")) - 1
        stop = min(max(stop, start + 1), len(widths))
        users = user_vectors(model, article_vectors, history_rows[start:stop], row_offset)  # (b, D)

        lo, hi = int(offsets[start]), int(offsets[stop])
        rows = torch.from_numpy(cand_rows[lo:hi].astype(np.int64) + row_offset).to(device)
        # Which impression each flat candidate belongs to.
        owner = torch.from_numpy(np.repeat(np.arange(stop - start), widths[start:stop])).to(device)
        piece = (article_vectors[rows] * users[owner]).sum(dim=-1)

        if model.freshness_head is not None:
            if log_age is None or known is None:
                raise ValueError("freshness head is attached but candidate ages were not supplied")
            piece = piece + model.freshness_head(
                torch.from_numpy(log_age[lo:hi]).to(device),
                torch.from_numpy(known[lo:hi]).to(device),
            )
        scores[lo:hi] = piece.float().cpu().numpy()
        start = stop
    return scores


def load_trained_model(checkpoint_path, device, freshness: bool | None = None):
    """Rebuild NRMS from a checkpoint without touching the HuggingFace weights.

    The embedding matrix is restored from the checkpoint itself, so submission
    runs do not re-download or re-load a 1.1 GB transformer only to overwrite it.

    `freshness=None` reads the arm off the checkpoint: only the freshness arm
    carries `freshness_head.*` weights, and building the head for a baseline
    checkpoint (or omitting it for a freshness one) would fail load_state_dict.
    """
    from src.nrms.model import NRMS

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if freshness is None:
        freshness = any(k.startswith("freshness_head") for k in state)
    vocab, dim = state["news_encoder.embedding.weight"].shape
    model = NRMS(
        np.zeros((vocab, dim), dtype=np.float32),
        num_heads=nrms_config.HEAD_NUM,
        head_dim=nrms_config.HEAD_DIM,
        attention_hidden_dim=nrms_config.ATTENTION_HIDDEN_DIM,
        dropout=nrms_config.DROPOUT,
        freshness=freshness,
        freshness_hidden_dim=nrms_config.FRESHNESS_HIDDEN_DIM,
    )
    model.load_state_dict(state)
    return model.to(device).eval()

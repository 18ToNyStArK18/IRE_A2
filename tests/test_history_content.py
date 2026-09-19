"""History-content features (A2 Q1.1): candidate similarity to the titles and
embeddings of the user's clicked articles, on a hand-built processed dir.

Pinned: related candidates score above unrelated ones on both channels, "no
history" reads as missing rather than dissimilar, and the only user input read is
the history handed in -- nothing about the impression's own outcome (Q9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, feature_pipeline, history_content

ARTICLES = {
    # id: (title, embedding direction)
    "h1": ("football match tonight", [1.0, 0.0, 0.0]),
    "h2": ("football league final", [0.9, 0.1, 0.0]),
    "c_related": ("football final result", [1.0, 0.05, 0.0]),
    "c_unrelated": ("weather storm warning", [0.0, 0.0, 1.0]),
    "c_no_vector": ("football cup", None),
}


@pytest.fixture
def processed_dir(tmp_path):
    ids = list(ARTICLES)
    pd.DataFrame({
        "article_id": ids,
        "title": [ARTICLES[a][0] for a in ids],
        "abstract": [""] * len(ids),
        "category": ["sports", "sports", "sports", "weather", "sports"],
    }).to_parquet(tmp_path / "articles.parquet", index=False)
    (tmp_path / "feature_store").mkdir()
    embedded = [a for a in ids if ARTICLES[a][1] is not None]
    pd.DataFrame({
        "article_id": embedded,
        "embedding": [np.asarray(ARTICLES[a][1], dtype=np.float32) for a in embedded],
    }).to_parquet(tmp_path / "feature_store" / "embeddings.parquet", index=False)
    return tmp_path


@pytest.mark.parametrize("dataset", ["mind", "ebnerd"])  # different query/pooling configs
def test_related_candidate_outscores_unrelated_on_both_channels(processed_dir, dataset):
    scorer = history_content.HistoryContentScorer(processed_dir, dataset)
    title, cosine = scorer.score(["h1", "h2"], ["c_related", "c_unrelated"])
    assert title[0] > title[1]
    assert cosine[0] > cosine[1]
    assert cosine[0] == pytest.approx(1.0, abs=0.02)  # same direction as the history
    assert cosine[1] == pytest.approx(0.0, abs=1e-6)  # orthogonal


def test_no_history_is_missing_not_dissimilar(processed_dir):
    scorer = history_content.HistoryContentScorer(processed_dir, "ebnerd")
    for history in ([], None):
        title, cosine = scorer.score(history, ["c_related", "c_unrelated"])
        assert np.isnan(title).all() and np.isnan(cosine).all()


def test_candidate_without_a_vector_is_missing_on_the_embedding_channel_only(processed_dir):
    scorer = history_content.HistoryContentScorer(processed_dir, "ebnerd")
    title, cosine = scorer.score(["h1", "h2"], ["c_no_vector"])
    assert title[0] > 0            # its title still matches "football"
    assert np.isnan(cosine[0])     # but "no vector" must not read as orthogonal


def test_scores_depend_only_on_the_history_given(processed_dir):
    """Q9: the scorer sees the clicks before the impression and the candidate
    ids -- nothing else about the user or the impression's outcome. Same
    history in, same scores out; a different history changes them."""
    scorer = history_content.HistoryContentScorer(processed_dir, "ebnerd")
    a = scorer.score(["h1", "h2"], ["c_related", "c_unrelated"])
    b = scorer.score(["h1", "h2"], ["c_related", "c_unrelated"])
    c = scorer.score(["c_unrelated"], ["c_related", "c_unrelated"])
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert c[1][1] > c[1][0]  # now the weather article is the similar one


def test_columns_align_with_candidate_rows_across_impressions(processed_dir):
    """feature_pipeline scores each impression once and writes back by row
    position; interleaved impressions must still land on their own rows."""
    candidates_df = pd.DataFrame({
        "impression_id": ["i1", "i2", "i1", "i2"],
        "article_id": ["c_related", "c_related", "c_unrelated", "c_unrelated"],
    })
    histories = {"i1": ["h1", "h2"], "i2": ["c_unrelated"]}
    history_content.get_scorer.cache_clear()
    title, cosine = feature_pipeline.history_content_columns(processed_dir, "ebnerd", candidates_df, histories)
    history_content.get_scorer.cache_clear()
    assert cosine[0] > cosine[2]  # i1 (football history): related beats unrelated
    assert cosine[3] > cosine[1]  # i2 (weather history): the reverse
    assert title[0] > title[2]


def test_switch_is_on_by_default():
    assert config.HISTORY_CONTENT_FEATURES is True
    assert history_content.FEATURES == ("history_title_bm25", "history_embedding_cosine")

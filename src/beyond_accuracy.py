"""Beyond-accuracy metrics over the lists the system actually recommends:
intra-list diversity, novelty, and catalogue coverage (A2 Q5).

Ported from A1's module of the same name. These describe properties of *what
gets recommended*, independent of whether it was clicked, so unlike
`src/metrics.py` they are computed on the re-ranked top-K rather than on
labels.

Diversity and novelty need a notion of item similarity and item popularity that
does not depend on which stage-1 method produced the candidates, or the arms
could not be compared on the same yardstick. Both use fixed references: the
semantic article embeddings, and train-split click counts.
"""

from __future__ import annotations

import numpy as np


def intra_list_diversity(
    candidate_ids: list[str],
    embeddings_lookup: dict[str, np.ndarray],
) -> float | None:
    """1 - average pairwise cosine similarity among the recommended articles.

    None when fewer than two of them have an embedding: a pair is the smallest
    thing a diversity score can describe, and returning 0.0 would read as
    "maximally similar" rather than "not measurable".
    """
    vectors = [embeddings_lookup[a] for a in candidate_ids if a in embeddings_lookup]
    if len(vectors) < 2:
        return None

    matrix = np.stack(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = matrix / norms

    similarity = normed @ normed.T
    n = len(vectors)
    upper_triangle_sum = (similarity.sum() - np.trace(similarity)) / 2
    return float(1 - upper_triangle_sum / (n * (n - 1) / 2))


def random_pair_diversity(
    embeddings_lookup: dict[str, np.ndarray], n_pairs: int = 4000, seed: int = 42
) -> float:
    """Diversity of two articles drawn at random -- the yardstick raw diversity
    has to be read against.

    Without it the number is uninterpretable and cross-dataset comparison is
    actively misleading. Measured here: a random EB-NeRD pair has cosine 0.951
    (diversity 0.049) because the provided multilingual BERT vectors are
    anisotropic and sit in a narrow cone, while a random MIND pair has cosine
    0.055 (diversity 0.945) because MiniLM is contrastively trained. So EB-NeRD
    recommendations scoring 0.037 and MIND's scoring 0.857 are *both* slightly
    less diverse than chance for their own space -- the 23x raw gap is the
    embedding model, not the recommender.
    """
    vectors = list(embeddings_lookup.values())
    if len(vectors) < 2:
        return float("nan")
    matrix = np.stack(vectors).astype(np.float64)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    rng = np.random.default_rng(seed)
    pairs = rng.choice(len(matrix), size=(n_pairs, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if not len(pairs):
        return float("nan")
    similarity = (matrix[pairs[:, 0]] * matrix[pairs[:, 1]]).sum(axis=1)
    return float(1 - similarity.mean())


def novelty(
    candidate_ids: list[str],
    click_probability: dict[str, float],
    default_probability: float,
) -> float:
    """Mean self-information -log2(p) of the recommended articles: rarer items
    score higher, so this is inverse popularity. `default_probability` covers
    articles with no train clicks, which would otherwise be -log2(0)."""
    if not candidate_ids:
        return 0.0
    return float(
        np.mean([-np.log2(click_probability.get(a, default_probability)) for a in candidate_ids])
    )


def coverage(recommended_ids_by_impression, catalogue_size: int) -> float:
    """Fraction of the catalogue appearing in at least one recommended list.

    Conditional on how many impressions were scored -- more impressions can only
    cover more catalogue -- so callers must report the impression count
    alongside it. It is not a per-impression mean, so it needs the dedicated
    resampler in bootstrap.coverage_ci rather than bootstrap.bootstrap_ci.
    """
    if catalogue_size == 0:
        return 0.0
    seen: set[str] = set()
    for ids in recommended_ids_by_impression:
        seen.update(ids)
    return len(seen) / catalogue_size


def click_probabilities(popularity_counts: dict[str, int]) -> tuple[dict[str, float], float]:
    """Train click counts -> (per-article probability, default for unseen).

    Zero-count articles are excluded rather than mapped to p=0, so they fall
    through to `default_probability` in novelty(). This matters because
    `popularity.parquet` lists every article ever *displayed* in train, not only
    the clicked ones (display_count was added for the CTR feature), so most rows
    have click_count 0 -- and -log2(0) is inf, which propagates through the mean
    and makes novelty and its whole CI inf/nan.
    """
    total = sum(popularity_counts.values())
    if total == 0:
        return {}, 1.0
    return (
        {a: c / total for a, c in popularity_counts.items() if c > 0},
        1.0 / (total + 1),
    )

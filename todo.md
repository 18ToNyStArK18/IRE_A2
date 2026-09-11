# TODO

## Open

### ~~Raise A1's retrieval recall so more candidates contain the click~~ — done 2026-09-11

Fixed by fresh-pool candidate generation (`src/fresh_pool.py`, the `popular`
method; DesignChoices.md §2E). Test recall@200 went from 2.55% / 2.90% to
**97.0% (EB-NeRD) / 93.9% (MIND)**. Two things were wrong: retrieval searched the
whole 2000–2023 catalogue while clicks go to articles a median 3.1 h old, and
history similarity is too sparse to rank a fresh pool (MIND title queries score
~20 of ~2,000 pool articles above zero), where popularity is not.

---

### Q3 framing: NRMS and the re-ranker are still not directly comparable

With fresh-pool candidates the two-stage pipeline is now competitive in absolute
terms (DesignChoices.md §2E): test MRR **0.2102 (EB-NeRD) / 0.2239 (MIND)**,
against NRMS's 0.3394 / 0.2750. But it ranks 200 candidates while NRMS ranks the
~9–23 in-view articles, so those still aren't the same task. A fair "beat the
baseline" needs identical candidate sets — NRMS scoring the same 200, or the
GBDT scoring the in-view lists.

---

### Smaller items

- [ ] Bi-encoder similarity feature — the one genuine Q1.1 gap (Q1.1 asks for
      "titles, categories, **embeddings**"; only the category channel exists).
      Cheap: embeddings are precomputed, so it is one dot product per row.
- [ ] Embeddings are not built for either dataset (the background job was killed
      mid-run), so semantic candidates cannot be generated yet.
      `python build_pipeline.py --stage embeddings`.
- [x] ~~Only `ebnerd/candidates_bm25_val.parquet` exists~~ — `bm25`,
      `bm25_fresh` and `popular` candidates now exist for every split of both
      datasets. Semantic candidates still wait on embeddings.
- [ ] Paired bootstrap 95% CIs for the §2E before/after deltas (Q3 requires them).
- [ ] Q1 feature-group ablation under the chosen random-negative sampling.

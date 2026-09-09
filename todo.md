# TODO

## Open

### Raise A1's retrieval recall so more candidates contain the click

**The problem.** A1's catalogue-wide retrieval finds the clicked article in only
~3.5% (EB-NeRD) / ~4.5% (MIND) of impressions at K=200, and no config in A1's
39-config ablation exceeded 4.6%. So ~96% of impressions reach the re-ranker
with no positive in the candidate set at all, which caps every Q2/Q5 metric
(perfect re-ranking still yields MRR ≈ 0.035 on EB-NeRD).

**Why it is parked, not blocking.** The re-ranker code is unaffected by this —
it consumes the `(impression_id, article_id, rank, retrieval_score)` contract
regardless of how good the candidates are. Better candidates change the
*numbers*, not the code. So the re-ranker gets built first, and A1 can be
improved underneath it afterwards without any rework on the A2 side.

**Now the single highest-value item, confirmed by results** (DesignChoices.md
§2D): swapping BM25 for embedding retrieval changed the final MRR by <2% within
a dataset, while the recall difference between datasets moved headroom capture
from 11% to 72%. Retrieval *recall* is the binding constraint; the retrieval
*method* is nearly irrelevant once the re-ranker is applied.

Also note MIND/bm25's recall collapses 5.9x from train (17.0%) to test (2.9%),
which is severe enough that its re-ranker ends up *worse* than the ordering it
started from. Semantic retrieval degrades only 1.6x and wins on test recall
despite losing on train — so a time-aware or embedding-based retriever is the
more promising direction.

**Ideas to try in A1, roughly cheapest first:**

- Restrict the retrieval pool by time — only articles published/first-seen
  before the impression, and within a recency window. The catalogue spans the
  whole period, so a large share of what BM25 currently returns could not have
  been shown at that impression at all. This should raise recall *and* speed
  retrieval up.
- Blend in popularity/recency rather than using it only as an empty-history
  fallback: news clicks concentrate hard on a few fresh articles, so a
  popularity prior over the recent window may beat pure history similarity.
- Union BM25 ∪ semantic candidates instead of running them as separate sets —
  A1's ablation shows they win on different datasets, so they likely retrieve
  different articles.
- Raise K (recall is monotone in K; cost is linear).
- Re-check the query construction: A1's winner uses title-only over a 10-click
  window, chosen on Recall@100 — worth confirming that still wins at K=200.

**When this changes, re-run:** `python -m src.candidates` for the affected
dataset/method/splits, then retrain the re-ranker. Nothing else needs touching.

---

### Q3 framing: NRMS and the re-ranker are not comparable

**Now confirmed with numbers on both sides** (DesignChoices.md §2C). Same 25,356
EB-NeRD test impressions: NRMS scores AUC 0.5425 on the in-view set, the
re-ranker 0.9592 on A1's top-200. The re-ranker is not better — it is solving an
easier discrimination. Meanwhile NRMS's MRR is 0.3394 against the re-ranker's
0.0182, because the two-stage pipeline is capped by A1's ~3.5% recall.


NRMS scores the impression's **in-view set** (~20 candidates, always containing
the click). The re-ranker scores A1's **top-200** (~96% with no click at all).
Q3 asks to "reproduce the baseline, then beat it", but the re-ranker
structurally cannot beat NRMS on numbers computed over different candidate sets.

Needs a decision on what "beat the baseline" means here. The apples-to-apples
comparison that *does* work is retrieval order vs re-ranked order on the same
candidates (the 97%-unclaimed-headroom result).

---

### Smaller items

- [ ] Bi-encoder similarity feature — the one genuine Q1.1 gap (Q1.1 asks for
      "titles, categories, **embeddings**"; only the category channel exists).
      Cheap: embeddings are precomputed, so it is one dot product per row.
- [ ] Embeddings are not built for either dataset (the background job was killed
      mid-run), so semantic candidates cannot be generated yet.
      `python build_pipeline.py --stage embeddings`.
- [ ] Only `ebnerd/candidates_bm25_val.parquet` exists — no MIND candidates, no
      semantic candidates, no train/test splits.

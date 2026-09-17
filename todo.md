# TODO

Open work only, as of 2026-09-17 (due 2026-09-20). Finished items live in
`DesignChoices.md` §3 and its changelog.

## Deliverables

- [ ] **Upload both Codabench submissions and capture leaderboard screenshots**
      (A2 Q5, Q7.3). Files are built and verified: `data/submissions/{mind,ebnerd}_nrms_freshness.zip` (§2I).
- [ ] Design note PDF (Q6) and AI usage log (Q7.4).

## Results still missing

- [ ] Paired bootstrap 95% CIs for the §2E re-ranker before/after deltas (Q3 requires CIs on claimed gains).
- [ ] Q1 feature-group ablation under the chosen 49-random-negative sampling (§2E's
      candidate-level-only arm ran under the discarded hard negatives).
- [ ] Q9: report metrics with and without features unavailable at serving time.
- [ ] Re-run `scripts/serving_benchmark.py` on the shipped `popular` model: both
      `results/serving_*.json` timed `lambdamart_bm25.txt` (§2G caveat), and the
      `popular` models now exist on this machine.

## Nice to have

- [ ] A fair Q3 head-to-head of NRMS and the re-ranker on identical candidate sets
      (MRR over 200 candidates is not NRMS's MRR over the in-view list).
- [ ] Bi-encoder similarity feature for the re-ranker (§2.6; closes the Q1.1
      "embeddings" channel). Embeddings are now built for both datasets.

# TODO

Open work only, as of 2026-09-17 (due 2026-09-20). Finished items live in
`DesignChoices.md` §3 and its changelog.

## Deliverables

- [ ] **Upload both Codabench submissions and capture leaderboard screenshots**
      (A2 Q5, Q7.3). Files are built and verified: `data/submissions/{mind,ebnerd}_nrms_freshness.zip` (§2I).
- [ ] Design note PDF (Q6) and AI usage log (Q7.4).

## Results still missing

- [ ] **Run `scripts/rerun_history_content.sh`** (history-content features are now
      default on, so every re-ranker artifact must be regenerated), then fill in
      DesignChoices §2K and its dated notes.

- [x] ~~Paired bootstrap 95% CIs for the §2E re-ranker before/after deltas~~ — in
      `results/serving_features_ablation_*.json` (`deltas_vs_stage1`), 2026-09-19.
- [ ] Q1 feature-group ablation under the chosen 49-random-negative sampling (§2E's
      candidate-level-only arm ran under the discarded hard negatives).
- [x] ~~Q9: metrics with and without features unavailable at serving time~~ —
      `scripts/serving_features_ablation.py`, 2026-09-19. Not yet written up in DesignChoices / the design note.
- [x] ~~Re-run `scripts/serving_benchmark.py` on the shipped `popular` model~~ — done in the
      2026-09-17 rerun; `design_note.md` §7 still quotes the old bm25 numbers.

## Nice to have

- [ ] A fair Q3 head-to-head of NRMS and the re-ranker on identical candidate sets
      (MRR over 200 candidates is not NRMS's MRR over the in-view list).
- [x] ~~Bi-encoder similarity feature for the re-ranker~~ — implemented as the
      history-content features (DesignChoices §2K), 2026-09-19.

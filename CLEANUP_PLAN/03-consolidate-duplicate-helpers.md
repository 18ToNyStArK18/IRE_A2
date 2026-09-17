# 03-consolidate-duplicate-helpers.md — medium; A2 code only

| # | Change | Verify |
|---|---|---|
| 3a | Add `config.PROCESSED_DIRS`. `candidates.PROCESSED_DIRS` and `nrms_config.PROCESSED_DIRS` alias it, so importers are unchanged. `serving_benchmark` imports it. Tests untouched | V1, V2 |
| 3b | `FreshnessLookup.from_references(references, codec)`. `build()` delegates to it and `age_signal.lookup_from_references` is replaced | V1, plus `age_signal.py` stdout byte-identical before/after for `mind`, `ebnerd`, `ebnerd --reference first-seen` |
| 3c | `article_stats.first_seen_times(processed_dir, splits)`, shared by `freshness_reference_times` (MIND branch) and `age_signal.first_seen_reference_times` | V1 (real-data equivalence test), plus the same 3 diffs |
| 3d | `metrics.per_impression_metrics(labels, scores) -> dict`, used by `extended_eval` and `paired_bootstrap` | V1, plus `paired_bootstrap.py` stdout identical for both datasets, plus `extended_eval.py --dataset ebnerd --method popular --out $CLAUDE_JOB_DIR/tmp/…` JSON identical before/after (MIND skipped: it OOM'd this laptop, §2H) |
- Optional, needs sign-off: one additive test pinning `per_impression_metrics` to `evaluate_impressions`.
- Not doing: a shared `reranker._article_index` in serving_benchmark (contaminates RSS), a unified model-path lookup (fallback semantics differ), the `NS_PER_HOUR` copies, `evaluate.score_split`'s device default.

---

## Shared verification:
- **V1** `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -q tests/` → **134 passed, 0 skipped**.
- **V2** Import every `src` / `src.nrms` module; `--help` exits 0 for `build_pipeline.py`, `run_nrms.py`, `run_submit.py`, `python -m src.candidates`, `python -m src.reranker`, and every `scripts/*.py`.
- Output comparisons go to `$CLAUDE_JOB_DIR/tmp/` only. Apart from 00B, nothing writes to `data/`, `results/`, `log/`, `wandb/`.
- After each file: diff summary, V1/V2 output, then **stop**. No commits unless you ask (one-line style).

### Execution protocol
1. On approval, write `CODEBASE_MAP.md` and `CLEANUP_PLAN/00…05`. **Stop.**
2. On "go 00" (etc.): run that one file, show the diff summary and verification output, then stop. Order is 00 → 04. 05 is answered item by item.
3. If verification fails: revert only that plan's edits, report the output, and wait.

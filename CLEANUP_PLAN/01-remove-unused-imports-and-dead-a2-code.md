# 01-remove-unused-imports-and-dead-a2-code.md — lowest risk

| File | Remove | Why safe |
|---|---|---|
| `src/feature_pipeline.py` | `load_candidates` re-export | A2-only wrapper with no caller. `reranker` calls `candidates.load_candidates` directly |
| `scripts/age_signal.py` | `from collections import defaultdict` | Unused |
| `src/lexical_retrieval.py`, `src/semantic_retrieval.py` [A1] | unused `from src import config` | Import only. No A1 function touched |
- **Explicitly kept (A1):** `lexical_retrieval.retrieve_for_split` / `compute_recall_at_k` / `run_bm25_pipeline` / `K_VALUES` and `semantic_retrieval.retrieve_for_split` / `run_semantic_pipeline` (A1 Q2.4/Q3.4 recall@K), `inverted_index.score_for_ids` and `ann_index.score_for_ids` (A1 Q4.5), `feature_store` outputs (A1 Q1.4), `popularity_top.parquet`, `split_all` / `build_all` / `compute_all`, `config.DATASETS`.
- **Verify:** V1, V2, and re-run the unused-import scan, which should show none remaining.

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

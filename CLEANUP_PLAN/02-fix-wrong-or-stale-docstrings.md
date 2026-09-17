# 02-fix-wrong-or-stale-docstrings.md — zero behaviour change

Only docstrings and `#` comments. Each stays short and explains why. A1 context is kept; only statements that are now false get fixed.
1. `src/embeddings_index.py`: MIND defaults to MiniLM (`MIND_SEMANTIC_MODEL`), not bert-base-uncased.
2. `src/feature_store.py`: the `embedding` column stays None, because `embeddings_index` writes `embeddings.parquet`.
3. `src/query_construction.py`, `src/inverted_index.py`: references to `plan.md` / `run_ablation_study.py` / `run_bm25_experiments.py` get "(A1 repo)" added so readers don't search for them here. Wording otherwise unchanged. Same for `lexical_retrieval.py`'s mention.
4. `src/feature_pipeline.py`: add `retrieval_score` to the contract, and scope "whole catalog" to the bm25/semantic arms.
5. `src/reranker.py`: fix the "Two things" / 3-items mismatch, and mark points 1–2 as the catalogue-candidate regime (§0). The `DEFAULT_PARAMS` truncation comment notes that median-rank-80 is a catalogue-candidate figure.
6. `src/candidate_features.py` `position_bias`: "A1 top-K" → "stage-1 rank" (covers all 4 generators).
7. `src/metrics.py`: "Q1 re-ranker" → "Q2 re-ranker".
8. `src/nrms/__init__.py` module map: add `freshness`, `serve`, `tracking`.
9. `build_pipeline.py` docstring: add `-> embeddings`.
10. `scripts/paired_bootstrap.py` seed hint: `--freshness --seed 43 --run-tag freshness_s43`.
- **Verify:** V1, V2, plus an **AST-equivalence check** per touched file: `ast.dump` of HEAD vs working tree with docstrings stripped must be identical, which proves only comments and docstrings changed.

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

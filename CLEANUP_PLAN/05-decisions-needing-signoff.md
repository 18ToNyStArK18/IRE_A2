# 05-decisions-needing-signoff.md — nothing runs without an explicit yes

1. `nrms_config.NEWS_VECTOR_DIM` is unused but documentary. Recommend keeping it.
2. Docs: fill or delete the empty `CLAUDE.md`; update or delete stale `todo.md` (README links it); delete the resolved `COMMIT_ad5c555_README.md`.
3. `DesignChoices.md` §2F CLI (~line 862) is missing `--freshness`. It's your authoritative doc, so I'll edit it only if asked.
4. Gaps, not cleanup:
   - README has no one-command reproduce (A1/A2 Q7.1).
   - No CLI exposes A1's `run_bm25_pipeline` / `run_semantic_pipeline` recall reports.
5. Additive test candidates: `split._split_one` ordering, `candidates.recall_at_k`, `lexical_retrieval.compute_recall_at_k`, `query_construction` tfidf_keywords, `dataset.eval_collate`, `SubmissionWriter` resume, `iter_ebnerd_chunks` skip, `adapter.load_impressions` `drop_no_click`, `avg_history_dwell_time`.
6. Other `../ire-a1` symlinks (`mind/train`, `mind/dev`, `ebnerd/demo`, `ebnerd/artifacts/bert`): copy them too? (Also asked in 00B.)
7. Leftover `data/processed/mind_large/` from the shelved WIP.

## Decisions (2026-09-17) and what was done
1. `NEWS_VECTOR_DIM`: **kept**.
2. `CLAUDE.md`: **filled** with orientation and conventions. `todo.md`: **rewritten** to the genuinely open items. `COMMIT_ad5c555_README.md`: **deleted**.
3. `DesignChoices.md`: **fixed** the missing `--freshness` in both seed-rerun commands (lines 862 and 1070).
4. README: **added** a "Reproduce everything" section, and updated the test-set note from symlinks to copies. A1 recall CLI: **skipped**.
5. Tests: **added** `tests/test_split.py` (4 tests) and `tests/test_recall.py` (3 tests). Suite: 141.
6. All `data/raw` symlinks into `../ire-a1` were **replaced with copies** (each verified identical with `diff -rq`).
7. `data/processed/mind_large/` was **deleted**. Other large-bundle leftovers remain untouched: `data/interim/mind_large`, `data/raw/mind/large_{train,dev}`, `data/raw/ebnerd/large`.

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

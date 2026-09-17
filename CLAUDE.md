# CLAUDE.md

CS4.406 A2 (team of 2): two-stage news recommendation on EB-NeRD and MIND, built on
A1's retrieval pipeline. Specs: `A2.pdf`, `Assignment1_v1.pdf`.

## Read first
- `DesignChoices.md` — authoritative record of decisions and results (§2E fresh-pool
  stage 1, §2F NRMS freshness arm, §2G serving, §2H extended eval, §2I submissions).
- `CODEBASE_MAP.md` — per-file map: role, callers, assignment question, test coverage,
  and the design choices that make code look redundant but must stay.
- `todo.md` — open work. `README.md` — reproduce commands.

## Layout
- `build_pipeline.py` + `src/{download,parse,split,feature_store,popularity,embeddings_index}.py` — A1 data pipeline.
- `src/{inverted_index,query_construction,lexical_retrieval,ann_index,semantic_retrieval}.py` — A1 retrieval.
  Keep A1 code even where A2 doesn't call it (A1 recall@K, eval harness, feature store).
- `src/fresh_pool.py`, `src/candidates.py` — stage 1 (`popular` shipped; others are ablation arms).
- `src/{article_stats,sessionize,impression_features,candidate_features,feature_pipeline,reranker}.py` — Q1 features + Q2 LambdaMART.
- `src/nrms/`, `run_nrms.py` — Q3 NRMS baseline and `--freshness` arm.
- `src/{metrics,bootstrap,beyond_accuracy,slicing}.py`, `scripts/` — evaluation, Q4 serving, Q5.
- `src/submission.py`, `src/nrms/serve.py`, `run_submit.py` — Codabench files.

## Commands
- Tests: `.venv/bin/python -m pytest tests/` (some tests read `data/processed/*` and skip if absent).
- Run long jobs with `.venv/bin/python -u`; heavy jobs one at a time (~5 GB free RAM, 8 GB GPU).

## Conventions
- Never open or print `.env`; it holds `WANDB_API_KEY`, loaded via python-dotenv.
- Don't modify `data/`, `results/`, `log/`, `wandb/` unless asked; write experiment output to a temp dir.
- Every feature must be as-of strictly before the impression time (Q9); keep the leakage tests passing.
- Commit messages: one lowercase line in the repo's `verb: description` style, no trailers; get approval first.
- Prefer minimal changes; measure a problem before fixing it.

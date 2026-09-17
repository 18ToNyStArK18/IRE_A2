# Codebase map — IRE_A2

**Baseline (2026-09-17):** `pytest tests/` gives 137 passed, 0 skipped (24 s). Expected 134 after plan 00 shelves the large-bundle WIP and its 3 streaming-parse tests.

## Assignment summary

**A1 (the foundation A2 extends):**
- Q1: one-command pipeline. Download MIND-small and EB-NeRD demo → unified schema → temporal split → feature store (article: title/abstract/body/category/entities/embeddings; user: click history, recency).
- Q2: BM25 inverted index over title+abstract, history→query, top-K, recall@{50,100,200}.
- Q3: embeddings (provided or computed), ANN (brute force OK), mean-pooled user vector, recall@K, lexical vs semantic by slice.
- Q4: AUC/MRR/nDCG@5/@10, diversity/novelty/coverage, one slice, bootstrap CIs, on both retrievers.
- Q5: Codabench.
- Q7–Q9: same deliverable and anti-gaming rules as A2.

**A2 (inputs):** same datasets, plus MINDlarge_test (2.37M) and ebnerd_testset (13.5M) for Codabench.
- Q1: behavioural features (history, recency decay, session, dwell, popularity/CTR/freshness, category match) with a strict as-of boundary.
- Q2: A1's generator as stage 1 (K 100–200), then a GBDT or neural re-ranker. Metrics before and after re-ranking.
- Q3: reproduce NRMS, one principled improvement, ablation, paired-bootstrap 95% CI excluding 0.
- Q4: index and feature-store memory, p99 latency, cost/QPS at an SLA, 10× scaling.
- Q5: all 7 metrics, cold/warm and head/tail slices, bootstrap CIs, Codabench submissions with screenshots.
- Q6: design note.
- Q7: README one-command reproduce.
- Q9: metrics with and without serving-unavailable features, plus a leakage test.

**Grading:** correctness, design, ablation rigour, scale analysis, note clarity. Rank is not graded.

**How the repo answers it:**
- A1 layer: `build_pipeline` + `download` / `parse` / `split` / `feature_store` / `popularity` / `embeddings_index`, `inverted_index` / `query_construction` / `lexical_retrieval`, `ann_index` / `semantic_retrieval`, and the ported `bootstrap` / `beyond_accuracy` / `slicing` / `submission`.
- A2 Q1: `article_stats` / `impression_features` / `candidate_features` / `sessionize`.
- A2 Q2: `fresh_pool` + `candidates` (stage 1, shipped method `popular`; `bm25` / `semantic` are A1's retrievers) → `feature_pipeline` + `reranker` (LambdaMART).
- A2 Q3: `src/nrms/*` + `run_nrms.py --freshness` + `scripts/paired_bootstrap.py`.
- A2 Q4: `scripts/serving_benchmark.py`.
- A2 Q5: `scripts/extended_eval.py`; `run_submit.py` + `nrms/serve` + `scripts/verify_fast_scorer.py`.
- Q9 tests: `test_article_stats_scope`, `test_feature_engineering`, `test_fresh_pool`, `test_freshness`.

### Documented design choices that make code *look* redundant (do not "clean")
| Looks redundant | Why it stays (source) |
|---|---|
| A1 retrieval eval loops (`retrieve_for_split`, `compute_recall_at_k`, `run_*_pipeline`, `score_for_ids`) not called by A2 code | A1 Q2.4/Q3.4/Q4.5 deliverables. A2 extends A1 |
| `bm25` / `semantic` / `bm25_fresh` generators | Stage-1 ablation arms (DesignChoices §2E, §2G) |
| `feature_store` article/user parquet and `popularity_top.parquet`, unread by A2 | A1 Q1.4 feature store and A1's fallback list |
| `TrainEventIndex(splits=("train",))` default vs the re-ranker's all-splits scope | Q1 semantics stay default and tested. All-splits fixes the §2E train/serve skew |
| `freshness_reference_times` re-deriving `as_of`'s reference | Speed, one per article. Equality tested |
| `config.ARTICLE_STATS_SPLITS` living in config | Re-ranker and NRMS can't drift (§2F correction 1) |
| Hard-negative branch with `TRAIN_HARD_NEGATIVES = 0` | Tried and hurt (§2E bug 2). Tested |
| `evaluate_before_after` (test-only) and `evaluate_chunked` | Reference for the chunking equality test |
| `reranker.evaluate_over_population` vs `metrics.evaluate_impressions` | Exact full-population denominator (tested) |
| `NRMS.score`, `padding_idx` param | Benchmark parity and the padding finding (tested) |
| `FreshnessHead` conditional, built last, inside `fork_rng` | Identical init and data order across arms (§2F) |
| No attention masking in `layers.py` | Faithful to the benchmark |
| `argsort[::-1]` tie order in `metrics` | Benchmark parity. Matters for tied "before" scores |
| `nrms/serve.py` alongside `nrms/evaluate.py` | Proven identical by `verify_fast_scorer.py` (§2I) |
| `MIND_SEMANTIC_MODEL` vs `MIND_BERT_MODEL` | Retrieval vs the NRMS tokenizer (config comment) |
| Both NRMS arms emit age tensors | Same batch shape, only the model differs |
| `age_signal.first_seen_reference_times` copy | article_stats short-circuits EB-NeRD. Can share a helper (plan 03) |
| `serving_benchmark` not calling `reranker._article_index` | That would import lightgbm into the RSS-measurement subprocesses |
| `bootstrap_ci` loop rather than an index matrix | Memory, ~585 MB per metric on MIND |

---

## Per-file map

Legend: **[A1]** = came from A1 (commits `b679970`, `a12cf9f`, or later ports). **Live** = reached from an A2 entry point. **A1-only** = an A1 deliverable not called by A2 code, which stays. **Dead** = no caller and no A1 role. No getattr/importlib dispatch exists, so grep plus an AST scan is reliable.

### Entry points
- **`build_pipeline.py` [A1]**: one-command rebuild (A1 Q1.5). Stages download → parse → split → feature_store → popularity → embeddings. Docstring's stage list is missing `embeddings`. Tests: none.
- **`run_nrms.py`** (committed version): Q3.1 baseline and `--freshness` arm. Calls `ids`, `articles`, `adapter`, `sampling`, `NRMS`, `FreshnessLookup`, `train`, `evaluate`, `tracking`. `_run_stages(artifact_dir=…)` actually receives `run_dir`, so the name misleads. Tests: none directly.
- **`run_submit.py`**: Q5/Q7.3 Codabench files via `submission`, `serve`, `FreshnessLookup`. Reads `data/raw/testsets/*` (currently symlinks into `../ire-a1`). Tests: components in `test_submission`, gated by `verify_fast_scorer.py`.

### src/ — A1 data pipeline
- **`config.py` [A1+A2]**: paths and constants, imported everywhere. `DATASETS` is A1-only. The `PROCESSED_DIRS` mapping is re-declared in `candidates.py`, `nrms/config.py`, `scripts/serving_benchmark.py` (and 3 tests).
- **`download.py` [A1]**: ← build_pipeline, plus CLI. Tests: none.
- **`parse.py` [A1]**: ← build_pipeline, plus CLI. Unified schema, and documents the Q9 exclusion of `read_time`/`next_*`. Tests: none (after stash).
- **`split.py` [A1]**: ← build_pipeline, plus CLI. Temporal split with inline overlap asserts. `split_all` is A1-only. Tests: none.
- **`feature_store.py` [A1]**: ← build_pipeline. A1 Q1.4 article/user store (not read by A2 code). The `embedding` column is never filled, because embeddings go to `embeddings.parquet`. Its docstring says otherwise. `build_all` is A1-only. Tests: none.
- **`popularity.py` [A1]**: ← build_pipeline. `popularity.parquet` ← `lexical_retrieval`, `extended_eval`. `popularity_top.parquet` and `compute_all` are A1-only. Tests: consumer only.
- **`embeddings_index.py` [A1]**: ← build_pipeline, plus CLI. Writes `feature_store/embeddings.parquet`. **Docstring wrong:** it says MIND uses bert-base-uncased, but the default is MiniLM. Tests: none.

### src/ — A1 retrieval, used as A2 stage 1 arms
- **`text_utils.py` [A1]**: `tokenize`. Tests: indirect.
- **`inverted_index.py` [A1]**: BM25 index and `concat_fields` ← lexical_retrieval, query_construction, embeddings_index, candidates (`score_for_ids` for bm25_fresh), serving_benchmark. Docstrings cite A1-repo files (`run_ablation_study.py`). Tests: indirect (`test_fresh_pool`).
- **`query_construction.py` [A1]**: `build_query` ← candidates. Cites A1's `plan.md` / `run_ablation_study.py`. Tests: indirect, recency_weighted only. **tfidf_keywords untested.**
- **`lexical_retrieval.py` [A1]**: live pieces are `load_articles_lookup`, `build_corpus_index`, `load_popularity_fallback`. A1-only: `retrieve_for_split`, `compute_recall_at_k`, `run_bm25_pipeline`, `K_VALUES` (A1 Q2 recall@K). Unused import: `config`. Tests: none directly.
- **`semantic_retrieval.py` [A1]**: live pieces are `load_embeddings_lookup` and `build_user_representation` with its pools. A1-only: `retrieve_for_split`, `run_semantic_pipeline` (A1 Q3). Unused import: `config`. Tests: none.
- **`ann_index.py` [A1]**: `BruteForceANN` ← semantic_retrieval. `score_for_ids` is A1-only (A1 Q4.5). Tests: none.
- *Gap:* no CLI in this repo invokes `run_bm25_pipeline` / `run_semantic_pipeline`. A2's `candidates.recall_at_k` reports the same recall@{50,100,200}.

### src/ — A2 stage 1 and features
- **`fresh_pool.py`**: ← candidates, serving_benchmark. Shipped stage 1 and the Q9 window. Tests: `test_fresh_pool`.
- **`candidates.py`**: 4 generators, `recall_at_k`, `load_candidates`, `PROCESSED_DIRS`, `METHODS`, CLI. Tests: popular/bm25_fresh contract. **bm25/semantic generators and `recall_at_k` untested.**
- **`article_stats.py`**: ← reranker, feature_pipeline, serving_benchmark, nrms/freshness, age_signal. Its first-seen scan is duplicated in age_signal. Tests: 3 files.
- **`sessionize.py`**: Tests: `test_feature_engineering`.
- **`impression_features.py`**: `recency_weighted_engagement` tested. **`history_category_weights`, `avg_history_dwell_time`, `build_impression_features` untested.**
- **`candidate_features.py`**: `position_bias` and `category_features` tested; the others only indirectly. `position_bias` docstring says "A1 top-K" rather than stage-1 rank.
- **`feature_pipeline.py`**: `build_feature_matrix` ← reranker; `load_articles_lookup` ← serving_benchmark. **Dead:** the `load_candidates` re-export (A2-only, no caller). Docstring contract omits `retrieval_score`. Tests: indirect.
- **`reranker.py`**: Q2 LambdaMART, CLI, ← extended_eval. **`train`, `run`, `_training_impressions`, `summary_table` untested.** Docstring says "Two things" but lists 3, and points 1–2 are catalogue-era.

### src/ — evaluation
- **`metrics.py`**: shared harness. Per-impression row logic is duplicated in `extended_eval` and `paired_bootstrap`. Docstring says "Q1 re-ranker" (should be Q2). Tests: `test_nrms`, `test_reranker`.
- **`bootstrap.py` / `beyond_accuracy.py` / `slicing.py` [A1 port]**: ← extended_eval. Tests: `test_extended_eval` (**`random_pair_diversity` untested**).
- **`submission.py` [A1 port]**: ← run_submit. **Stream iterators, `first_seen_reference_ns`, `load_ebnerd_histories`, `SubmissionWriter` resume, `verify_submission` untested.**

### src/nrms/ (A2 Q3)
- `__init__` module map is **missing `freshness`, `serve`, `tracking`**.
- `config`: `NEWS_VECTOR_DIM` is unused but documentary. `PROCESSED_DIRS` is a duplicate.
- `ids`: **`build_article_codec`, `load_or_build_codec` untested.**
- `adapter`: **`load_impressions`, `load_article_text` untested.**
- `articles`: **cache loader and `load_word_embeddings` untested.**
- `sampling`: tested.
- `dataset`: **`eval_collate` untested.**
- `layers`, `model`, `freshness`, `serve`, `tracking`: tested.
- `train`: `_state_dict_to_cpu` tested; **loop untested.**
- `evaluate`: **untested directly.**

### scripts/ (CLIs in README, no unit tests)
- `paired_bootstrap` (Q3.4): duplicate `per_impression`. The docstring seed hint is missing `--freshness`.
- `age_signal` (§2F): duplicates `first_seen_reference_times` and `lookup_from_references`. Unused import `defaultdict`.
- `extended_eval` (Q5): duplicate `per_impression_accuracy`.
- `serving_benchmark` (Q4): duplicate `PROCESSED_DIRS`.
- `verify_fast_scorer` (§2I gate).

### tests/
141 after plan 05 added `test_split` (temporal split ordering) and `test_recall` (A1 and A2 recall@K definitions). Coverage per file:
- `test_article_stats_scope`: scope and lag.
- `test_extended_eval`: Q5 modules.
- `test_feature_engineering`: Q1 and Q9 leakage (real data).
- `test_fresh_pool`: stage 1 as-of.
- `test_freshness`: ablation validity and age gate (real data).
- `test_nrms`: codec, truncation, sampling, metrics, token matrix.
- `test_nrms_model`: shapes, gradients, padding.
- `test_reranker`: exactness, sampling, chunking (real data).
- `test_submission`: format, fast scorer.
- `test_tracking`.

### Data layout note
`data/raw/{mind/train, mind/dev, ebnerd/demo, ebnerd/artifacts/bert, testsets/MINDlarge_test, testsets/ebnerd_testset}` were symlinks into `../ire-a1/data/interim/`; all are now real copies (plans 00B and 05), so the repo no longer depends on `../ire-a1`.

### Debug/print clutter
None. Every `print` is CLI progress or result output.

### Non-code files
- `CLAUDE.md` is empty.
- `todo.md` is stale.
- `COMMIT_ad5c555_README.md` is an untracked, resolved review.
- `Assignment1_v1.pdf` is tracked; it's the A1 spec, which is now clearly relevant.
- README lacks a one-command reproduce for the whole pipeline.

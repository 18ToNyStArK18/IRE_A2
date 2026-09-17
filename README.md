# IRE_A2

Two-stage news recommendation on EB-NeRD and MIND (CS4.406 A2), built on top of
the A1 retrieval pipeline. `DesignChoices.md` is the reasoning and results log,
`CODEBASE_MAP.md` maps every file to the assignment, and `todo.md` tracks open work.

## Reproduce everything

Run from the repo root with the project venv. Every step reads the previous
step's output under `data/` (gitignored).

```bash
pip install -r requirements.txt

# 1. A1 data pipeline: download -> parse -> temporal split -> feature store
#    -> popularity -> embeddings. MIND is gated on HuggingFace: pass --hf-token
#    or set HF_TOKEN.
python build_pipeline.py

# 2. Stage 1 candidates (A2 Q2). `popular` is the shipped method; bm25,
#    semantic and bm25_fresh are the stage-1 ablation arms (DesignChoices §2E).
python -m src.candidates --method all

# 3. Stage 2 LambdaMART re-ranker, before/after metrics per method (A2 Q2).
python -m src.reranker --method popular

# 4. NRMS baseline and the freshness arm (A2 Q3). Epoch caps match the
#    reported runs; both early-stop inside them.
python run_nrms.py --dataset ebnerd --stage all --epochs 20
python run_nrms.py --dataset ebnerd --stage all --epochs 20 --freshness
python run_nrms.py --dataset mind   --stage all --epochs 10
python run_nrms.py --dataset mind   --stage all --epochs 10 --freshness

# 5. Tests, including the Q9 no-future-leakage checks.
python -m pytest tests/
```

Steps 2–3 accept `--dataset mind|ebnerd` to run one dataset. NRMS logs to W&B
when `WANDB_API_KEY` is set (via `.env`); pass `--no-wandb` to skip it.

## Analysis scripts

Reproduce the §2F freshness-ablation figures from the artifacts a run leaves in
`data/processed/<dataset>/nrms/`:

```bash
python scripts/paired_bootstrap.py --dataset ebnerd            # arm vs baseline, paired 95% CIs
python scripts/age_signal.py --dataset mind                    # how much signal article age carries
python scripts/age_signal.py --dataset ebnerd --reference first-seen   # what MIND's proxy costs
```

Q4 serving and scale (§2G), writing `results/serving_<dataset>.json`:

```bash
python scripts/serving_benchmark.py --dataset ebnerd    # memory, p50/p95/p99, QPS, cost, scaling
```

Q5 extended evaluation (§2H), writing `results/extended_eval_<dataset>_<method>.json`:

```bash
python scripts/extended_eval.py --dataset ebnerd --method popular   # all 7 metrics, 2 slices, bootstrap CIs
```

These reuse `src/metrics.py`, so they cannot drift from the `metrics_<split>.json`
each run writes.

Q5/Q7.3 Codabench submissions (§2I), writing `data/submissions/<dataset>_nrms_freshness.zip`:

```bash
python scripts/verify_fast_scorer.py --dataset ebnerd   # gate: fast scorer == committed metrics
python run_submit.py --dataset mind --limit 5000        # smoke test
python run_submit.py --dataset mind                     # full file, ~3 min
python run_submit.py --dataset ebnerd                   # full file, ~4 min
```

The official test sets are downloaded separately from the training bundles
(`MINDlarge_test.zip` from the HuggingFace MIND repo, `ebnerd_testset.zip` from
the EB-NeRD S3 bucket) and extracted to `data/raw/testsets/MINDlarge_test/`
(`behaviors.tsv`, `news.tsv`) and `data/raw/testsets/ebnerd_testset/`
(`articles.parquet`, `test/`). Pass `--test-root` if yours live elsewhere.

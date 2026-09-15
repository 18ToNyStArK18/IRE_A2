# IRE_A2

Two-stage news recommendation on EB-NeRD and MIND (CS4.406 A2).
`DesignChoices.md` is the reasoning and results log; `todo.md` tracks open work.

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

Both reuse `src/metrics.py`, so they cannot drift from the `metrics_<split>.json`
each run writes.

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

Both reuse `src/metrics.py`, so they cannot drift from the `metrics_<split>.json`
each run writes.

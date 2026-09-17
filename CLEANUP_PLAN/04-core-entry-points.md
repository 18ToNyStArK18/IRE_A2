# 04-core-entry-points.md — highest risk

- **run_nrms.py:** rename `_run_stages`' parameter `artifact_dir` → `run_dir`. Local only; call sites are positional.
- **build_pipeline.py:** no code change. The explicit per-dataset blocks stay, given your minimal-changes preference.
- **run_submit.py:** reviewed, no change.
- **Verify:** V1, V2, plus a before/after GPU smoke that writes only to temp: `run_nrms.py --dataset ebnerd --stage all --freshness --fraction 0.01 --epochs 1 --eval-split val --no-wandb --run-tag $CLAUDE_JOB_DIR/tmp/nrms_smoke_{before,after} --log-dir $CLAUDE_JOB_DIR/tmp`. The absolute `--run-tag` resolves outside `data/`, and the caches are only read. `metrics_val.json` should agree within ≤1e-4.

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

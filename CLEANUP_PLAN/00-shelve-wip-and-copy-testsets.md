# 00-shelve-wip-and-copy-testsets.md

**A. Shelve large-bundle WIP (reversible)**
- Run `git stash push -u -m "large-bundle WIP (shelved 2026-09-17)" -- run_nrms.py src/config.py src/download.py src/parse.py src/split.py tests/test_parse_streaming.py`.
- Preserved: all committed behaviour. Nothing committed references the WIP. Undo with `git stash pop`.
- `data/processed/mind_large/` is left alone; delete it by hand if you want.
- Verify: `git status` is clean apart from untracked docs; V1 = 134; V2.

**B. Replace the test-set symlinks with real copies**
- Run `cp -rL <link target> data/raw/testsets/<name>.copy` for `MINDlarge_test` (1.5 GB) and `ebnerd_testset` (1.8 GB), then `rm` the symlink (this removes the link only; the source in `../ire-a1` is untouched), then `mv <name>.copy <name>`.
- Paths are unchanged, so `run_submit.TEST_ROOTS` needs no edit.
- `.gitignore` already covers `data/`, so nothing large reaches git.
- Verify:
  - `test -L` is false for both.
  - `diff -rq <ire-a1 source> data/raw/testsets/<name>` shows no differences.
  - `run_submit.py --dataset mind --limit 5000 --out $CLAUDE_JOB_DIR/tmp/smoke_mind.zip --log-dir $CLAUDE_JOB_DIR/tmp` and the same for `--dataset ebnerd` both run. Token and first-seen caches already exist, so these runs only read `data/processed`.
- **Question inside this plan file:** do you also want `mind/train`, `mind/dev`, `ebnerd/demo`, `ebnerd/artifacts/bert` copied rather than linked? You said "the large dataset", so by default only the test sets are copied.

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

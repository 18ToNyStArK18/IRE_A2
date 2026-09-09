"""Run observability: file logging and Weights & Biases tracking.

Two independent concerns, both designed so a failure here can never take down a
training run:

  setup_logging  -- console + a durable file under log/. Training runs are long
                    enough that "it printed something an hour ago" is not a
                    record; and because stdout is block-buffered when redirected,
                    console output alone can go silent for minutes at a time.

  WandbTracker   -- a thin, failure-tolerant wrapper around wandb. A 100-minute
                    run must not die because a network call failed, so every
                    call is guarded: a failing init retries once offline, and if
                    that fails too the tracker disables itself and training
                    continues uninterrupted.

Secrets: WANDB_API_KEY is read from the environment, which `load_dotenv()`
populates from .env. This module never opens, echoes or logs the file or the
key -- only whether a key is present, which is what you need to diagnose a
missing one.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

LOGGER_NAME = "nrms"
_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(dataset: str, log_dir: Path, level: int = logging.INFO) -> Path:
    """Attach a timestamped file handler plus a console handler to the `nrms`
    logger. Returns the log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"nrms_{dataset}_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.handlers.clear()  # re-running in one process must not double-log
    logger.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    return path


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def load_env() -> bool:
    """Populate os.environ from .env. Returns whether WANDB_API_KEY is present.

    The key itself is never returned, logged or inspected -- only its presence,
    so that "wandb rejected my credentials" can be told apart from "there were
    no credentials" without ever handling the secret.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        get_logger("tracking").warning(
            "python-dotenv not installed; relying on the ambient environment"
        )
    return bool(os.environ.get("WANDB_API_KEY"))


class WandbTracker:
    """Guarded wandb wrapper. Every method is a no-op once disabled."""

    def __init__(self, run=None, enabled: bool = False):
        self._run = run
        self.enabled = enabled and run is not None

    @classmethod
    def start(
        cls,
        *,
        enabled: bool,
        project: str,
        run_name: str | None,
        group: str | None,
        config: dict,
    ) -> "WandbTracker":
        log = get_logger("tracking")
        if not enabled:
            log.info("wandb: disabled by flag")
            return cls(enabled=False)

        has_key = load_env()
        log.info("wandb: WANDB_API_KEY present: %s", "yes" if has_key else "no")
        if not has_key:
            log.warning("wandb: no API key found; falling back to offline mode")
            os.environ.setdefault("WANDB_MODE", "offline")

        try:
            import wandb
        except ImportError:
            log.warning("wandb: not installed; tracking disabled")
            return cls(enabled=False)

        for mode in (os.environ.get("WANDB_MODE"), "offline"):
            try:
                run = wandb.init(
                    project=project,
                    name=run_name,
                    group=group,
                    config=config,
                    mode=mode,
                    reinit=True,
                )
                log.info("wandb: run started (mode=%s) %s", mode or "online", run.url or "")
                return cls(run=run, enabled=True)
            except Exception as exc:  # noqa: BLE001 - tracking must never be fatal
                log.warning("wandb: init failed (mode=%s): %s", mode or "online", exc)

        log.warning("wandb: could not start; continuing without tracking")
        return cls(enabled=False)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if not self.enabled:
            return
        try:
            self._run.log(metrics, step=step)
        except Exception as exc:  # noqa: BLE001
            get_logger("tracking").warning("wandb: log failed, disabling: %s", exc)
            self.enabled = False

    def set_summary(self, summary: dict) -> None:
        if not self.enabled:
            return
        try:
            for key, value in summary.items():
                self._run.summary[key] = value
        except Exception as exc:  # noqa: BLE001
            get_logger("tracking").warning("wandb: summary failed, disabling: %s", exc)
            self.enabled = False

    def finish(self) -> None:
        if not self.enabled:
            return
        try:
            self._run.finish()
        except Exception as exc:  # noqa: BLE001
            get_logger("tracking").warning("wandb: finish failed: %s", exc)
        finally:
            self.enabled = False

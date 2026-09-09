"""Tracking must never be able to kill a training run.

A 100-minute job cannot die because a network call failed, so these pin the
degradation path: a disabled tracker no-ops, and a tracker whose backend raises
turns itself off rather than propagating.
"""

from __future__ import annotations

import logging

from src.nrms.tracking import WandbTracker, get_logger, setup_logging


def test_disabled_tracker_no_ops():
    tracker = WandbTracker(enabled=False)
    assert not tracker.enabled
    tracker.log_metrics({"a": 1}, step=1)   # must not raise
    tracker.set_summary({"b": 2})
    tracker.finish()


def test_start_with_enabled_false_is_disabled():
    tracker = WandbTracker.start(
        enabled=False, project="p", run_name="r", group="g", config={}
    )
    assert not tracker.enabled


class _ExplodingRun:
    url = ""
    summary: dict = {}

    def log(self, *a, **k):
        raise RuntimeError("network down")

    def finish(self):
        raise RuntimeError("network down")


def test_failing_log_disables_instead_of_raising():
    """The failure mode that matters: wandb dies mid-run and training continues."""
    tracker = WandbTracker(run=_ExplodingRun(), enabled=True)
    tracker.log_metrics({"loss": 1.0}, step=1)  # must swallow
    assert not tracker.enabled
    tracker.log_metrics({"loss": 0.9}, step=2)  # still safe once disabled


def test_failing_summary_disables_instead_of_raising():
    class _BadSummary(_ExplodingRun):
        @property
        def summary(self):
            raise RuntimeError("network down")

    tracker = WandbTracker(run=_BadSummary(), enabled=True)
    tracker.set_summary({"auc": 0.5})
    assert not tracker.enabled


def test_failing_finish_does_not_raise():
    tracker = WandbTracker(run=_ExplodingRun(), enabled=True)
    tracker.finish()
    assert not tracker.enabled


def test_setup_logging_writes_a_file(tmp_path):
    path = setup_logging("testds", tmp_path)
    assert path.parent == tmp_path and path.name.startswith("nrms_testds_")

    get_logger("unit").info("hello from the test")
    for handler in logging.getLogger("nrms").handlers:
        handler.flush()
    assert "hello from the test" in path.read_text()


def test_setup_logging_is_idempotent(tmp_path):
    """Re-running in one process must not duplicate every line."""
    setup_logging("a", tmp_path)
    path = setup_logging("b", tmp_path)
    get_logger("unit").info("once")
    for handler in logging.getLogger("nrms").handlers:
        handler.flush()
    assert path.read_text().count("once") == 1

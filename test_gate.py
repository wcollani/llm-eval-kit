#!/usr/bin/env python3
"""Tests for --fail-under, which is the only thing standing between a regression and a
green run.

These exist because the gate was silently inert for a while: ToolCallPassRate was only
produced by _aggregate_samples, so a repeats=1 run carried a failure mode but no pass
rate, the threshold had nothing to compare against, and it printed PASS unconditionally.
It looked exactly like a working gate.

So the cases that matter here are the ones that must FAIL. A gate that can only be
observed passing has not been observed at all.

Run: python3 test_gate.py
"""
import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

import typer

spec = importlib.util.spec_from_file_location("cli", Path(__file__).parent / "cli.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def _run(scores_list, threshold):
    """Returns None if the gate passed, or the exit code if it failed."""
    results = {"runs": [{"pipeline": {"model": "m"}, "scores": s} for s in scores_list]}
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            cli._print_headline(results, threshold)
            return None
        except typer.Exit as exc:
            return exc.exit_code


OK = {"ToolCallMetric": 1.0, "ToolCallPassRate": 1.0, "ToolCallFailureMode": "ok"}
WRONG = {"ToolCallMetric": 0.0, "ToolCallPassRate": 0.0, "ToolCallFailureMode": "wrong_tool"}
UNSCORED = {"ToolCallMetric": None, "ToolCallPassRate": None, "ToolCallFailureMode": None}


class TestFailUnder(unittest.TestCase):
    def test_passes_when_above_threshold(self):
        self.assertIsNone(_run([OK], 0.95))

    def test_fails_when_below_threshold(self):
        self.assertEqual(_run([WRONG], 0.95), 1)

    def test_fails_against_an_unreachable_threshold(self):
        # Proves the gate is actually comparing rather than always passing.
        self.assertEqual(_run([OK], 1.01), 1)

    def test_fails_when_nothing_is_measurable(self):
        # The original bug. A threshold that cannot be evaluated must not report success.
        self.assertEqual(_run([UNSCORED], 0.95), 1)

    def test_one_bad_pipeline_fails_the_run(self):
        self.assertEqual(_run([OK, WRONG], 0.95), 1)

    def test_no_threshold_never_fails(self):
        self.assertIsNone(_run([WRONG], None))


class TestSingleSamplePassRate(unittest.TestCase):
    """The repeats=1 path must produce a pass rate, or the gate above has no input."""

    def test_headline_reports_a_rate_for_a_single_ok_sample(self):
        h = cli._headline_scores({"runs": [{"pipeline": {"model": "m"}, "scores": OK}]})
        self.assertEqual(h["{'model': 'm'}"]["ToolCallPassRate"], 1.0)

    def test_incorrect_actions_counted_separately_from_no_call(self):
        # no_call is a reliability failure, wrong_tool is a correctness failure. Lane
        # additions gate on the latter; conflating them hides the regression.
        runs = [
            {"pipeline": {"model": "m"}, "scores": {**OK, "ToolCallFailureMode": "no_call", "ToolCallPassRate": 0.0}},
            {"pipeline": {"model": "m"}, "scores": WRONG},
        ]
        h = cli._headline_scores({"runs": runs})["{'model': 'm'}"]
        self.assertEqual(h["incorrect_actions"], 1)
        self.assertEqual(h["failure_modes"], {"no_call": 1, "wrong_tool": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)

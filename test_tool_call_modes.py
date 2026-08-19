#!/usr/bin/env python3
"""Tests for the shared unparsed-tool-call vocabulary.

This module is vendored **verbatim** into harness-bench, so it has two consumers that never
import each other. These tests are the only thing that notices when a change here breaks the
other one.

The cases that matter are in two groups, and the second is the important one:

  - Recall: every emission shape observed in this homelab must still match. Losing one
    silently re-labels a plumbing bug as a model failure, which is a reason to stop
    investigating a thing that is still broken.
  - **Precision: ordinary prose must NOT match.** A false positive is worse than a miss. It
    manufactures a "the harness dropped it" story for a model that simply answered in words,
    and that story is the one nobody re-checks.

Run: python3 test_tool_call_modes.py
"""
import importlib.util
import unittest
from pathlib import Path

import tool_call_modes as tcm

spec = importlib.util.spec_from_file_location("cli", Path(__file__).parent / "cli.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class TestRecall(unittest.TestCase):
    """Each of these is a real emission, not a constructed one."""

    def test_tool_call_wrapper(self):
        # The tag qwen2.5-coder's chat template declares.
        self.assertTrue(tcm.looks_like_unparsed_call('<tool_call>{"name": "read"}</tool_call>'))

    def test_bare_name_arguments_object(self):
        # qwen2.5-coder's actual emission: 0/3 parsed at 7b, 14b and 32b.
        self.assertTrue(tcm.looks_like_unparsed_call(
            'I will read it.\n{"name": "read_file", "arguments": {"path": "a.py"}}'
        ))

    def test_openai_shaped_function_object(self):
        self.assertTrue(tcm.looks_like_unparsed_call('{"function": {"name": "ls"}}'))

    def test_xml_function_form(self):
        # harness-bench round 2a: omp emitted <function=glob>, cline <function=read_files>,
        # both with zero parsed tool calls. The three patterns above all miss this.
        self.assertTrue(tcm.looks_like_unparsed_call("<function=glob>{'pattern': '*.py'}"))
        self.assertTrue(tcm.looks_like_unparsed_call("<function=read_files>"))

    def test_pipe_delimited_special_token(self):
        self.assertTrue(tcm.looks_like_unparsed_call("<|tool_call|>"))
        self.assertTrue(tcm.looks_like_unparsed_call("<tool_call_begin>"))

    def test_evidence_reports_pattern_and_offset(self):
        evidence = tcm.unparsed_call_evidence("ok then <function=glob>")
        self.assertEqual(len(evidence), 1)
        self.assertIn("offset 8", evidence[0])

    def test_evidence_is_empty_when_nothing_matched(self):
        self.assertEqual(tcm.unparsed_call_evidence("just prose"), [])


class TestPrecision(unittest.TestCase):
    """The half that keeps this from inventing plumbing bugs."""

    def test_plain_prose_does_not_match(self):
        self.assertFalse(tcm.looks_like_unparsed_call(
            "I looked at normalize.py and the bug is on line 42. Want me to fix it?"
        ))

    def test_talking_about_tool_calls_does_not_match(self):
        # A model narrating its own behaviour is not a dropped call.
        self.assertFalse(tcm.looks_like_unparsed_call(
            "I tried to make a tool call but no tool was available."
        ))

    def test_unrelated_json_does_not_match(self):
        # A model quoting a config file must not read as an emitted call.
        self.assertFalse(tcm.looks_like_unparsed_call(
            '{"name": "my-service"}\n{"arguments": ["--verbose"]}'
        ))

    def test_prose_mentioning_a_function_does_not_match(self):
        self.assertFalse(tcm.looks_like_unparsed_call(
            "The function = the thing that broke. See def function(x) below."
        ))

    def test_empty_and_none_are_safe(self):
        self.assertFalse(tcm.looks_like_unparsed_call(""))
        self.assertFalse(tcm.looks_like_unparsed_call(None))


class TestToolCallMetricStillDelegates(unittest.TestCase):
    """ToolCallMetric now calls into this module. Prove the behaviour did not move."""

    def _mode(self, raw_content, tool_calls=()):
        metric = cli.ToolCallMetric(
            expected_calls=[{"name": "read_file"}],
            tool_calls=list(tool_calls),
            raw_content=raw_content,
        )
        metric.measure(test_case=None)
        return metric.failure_mode

    def test_unparsed_call_still_detected(self):
        self.assertEqual(
            self._mode('{"name": "read_file", "arguments": {"path": "a.py"}}'),
            "unparsed_call",
        )

    def test_new_pattern_reaches_the_metric(self):
        # llm-eval-kit under-detected this until 2026-08-19; it scored as no_call.
        self.assertEqual(self._mode("<function=read_file>"), "unparsed_call")

    def test_prose_is_still_no_call(self):
        self.assertEqual(self._mode("I cannot do that."), "no_call")

    def test_a_parsed_call_is_unaffected(self):
        self.assertEqual(self._mode("", tool_calls=[{"name": "read_file", "arguments": {}}]), "ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)

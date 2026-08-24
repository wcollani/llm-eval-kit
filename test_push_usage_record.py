"""homelab#95: push_usage_record actually calls push_to_gateway with the right shape."""

import unittest
from unittest import mock

import eval_logger
from usage_record import UsageRecord


class TestPushUsageRecord(unittest.TestCase):
    def test_a_missing_gateway_url_no_ops(self):
        with mock.patch.dict("os.environ", {}, clear=True), \
             mock.patch("eval_logger.push_to_gateway") as pushed:
            eval_logger.push_usage_record(UsageRecord(
                source="harness-run", harness="claude-code", model="haiku",
                cost_class="subscription_quota", tokens_in=10, tokens_out=5,
            ))
        pushed.assert_not_called()

    def test_a_real_push_carries_the_grouping_labels(self):
        with mock.patch("eval_logger.push_to_gateway") as pushed:
            eval_logger.push_usage_record(UsageRecord(
                source="harness-run", harness="claude-code", model="haiku",
                cost_class="subscription_quota", tokens_in=10, tokens_out=5,
                usd=None, gpu_seconds=None,
            ), gateway_url="http://pushgateway:9091")
        pushed.assert_called_once()
        _, kwargs = pushed.call_args
        self.assertEqual(kwargs["job"], "homelab_usage")
        self.assertEqual(kwargs["grouping_key"],
                         dict(source="harness-run", harness="claude-code",
                              model="haiku", cost_class="subscription_quota"))

    def test_a_push_error_is_swallowed_not_raised(self):
        with mock.patch("eval_logger.push_to_gateway", side_effect=OSError("unreachable")):
            eval_logger.push_usage_record(
                UsageRecord(source="x", harness="x", model="x", cost_class="unavailable",
                           tokens_in=0, tokens_out=0),
                gateway_url="http://pushgateway:9091",
            )  # must not raise


if __name__ == "__main__":
    unittest.main()

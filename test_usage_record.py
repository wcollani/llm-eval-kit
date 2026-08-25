"""homelab#93: round-trip UsageRecord against two real trial records rather than invented ones."""

import unittest

from usage_record import COST_CLASSES, UsageRecord


class TestUsageRecordRoundTrips(unittest.TestCase):
    def test_a_real_agy_subscription_quota_run(self):
        """harness-bench results/Round_2b_Subscription_Tier_20260818_191733.json,
        agy+sonnet on triage-shared-root-cause: no dollar figure, pure plan quota."""
        raw = {
            "source": "harness-bench", "harness": "agy", "model": "sonnet",
            "cost_class": "subscription_quota",
            "tokens_in": 25403, "tokens_out": 1871,
            "tokens_cache_read": 0, "tokens_cache_write": 0,
            "usd": None, "wall_clock_s": 44.98122317299999,
        }
        rec = UsageRecord.from_dict(raw)
        self.assertEqual(rec.cost_class, "subscription_quota")
        self.assertIsNone(rec.usd)
        self.assertIsNone(rec.gpu_seconds)
        self.assertEqual(UsageRecord.from_dict(rec.to_dict()), rec)

    def test_a_real_free_local_ollama_run(self):
        """harness-bench results/opencode_stderr_verification_20260817_130106.json,
        opencode+ollama/gpt-oss:20b on add-loud-flag. GPU-seconds is not real data -- nothing in
        the estate emits it yet (that's #94's job); this is an illustrative value on top of the
        real token/latency/cost_class fields, noted rather than passed off as measured."""
        raw = {
            "source": "harness-bench", "harness": "opencode", "model": "ollama/gpt-oss:20b",
            "cost_class": "free_local",
            "tokens_in": 0, "tokens_out": 0,
            "usd": None, "wall_clock_s": 300.02527195199946,
            "gpu_seconds": 297.5,  # illustrative -- see docstring
        }
        rec = UsageRecord.from_dict(raw)
        self.assertEqual(rec.cost_class, "free_local")
        self.assertIsNone(rec.usd)
        self.assertEqual(rec.gpu_seconds, 297.5)
        self.assertEqual(UsageRecord.from_dict(rec.to_dict()), rec)

    def test_cost_classes_match_harness_bench_exactly(self):
        # Checked against harness_bench/core/config.py's COST_CLASSES directly, not from memory.
        self.assertEqual(COST_CLASSES, ("free_local", "metered", "subscription_quota", "unavailable"))

    def test_an_unknown_cost_class_is_rejected(self):
        with self.assertRaises(ValueError):
            UsageRecord(source="x", harness="x", model="x", cost_class="made_up",
                       tokens_in=0, tokens_out=0)

    def test_run_id_defaults_to_unattributed(self):
        """homelab#98. A record built with no run_id is "" -- a real, reportable value, not a
        missing field an older caller forgot to set."""
        rec = UsageRecord(source="x", harness="x", model="x", cost_class="unavailable",
                          tokens_in=0, tokens_out=0)
        self.assertEqual(rec.run_id, "")

    def test_run_id_round_trips(self):
        raw = {"source": "harness-run", "harness": "claude-code", "model": "haiku",
              "cost_class": "subscription_quota", "tokens_in": 1, "tokens_out": 1,
              "run_id": "collani-homelab/homelab-toy#232"}
        rec = UsageRecord.from_dict(raw)
        self.assertEqual(rec.run_id, "collani-homelab/homelab-toy#232")
        self.assertEqual(UsageRecord.from_dict(rec.to_dict()), rec)


if __name__ == "__main__":
    unittest.main()

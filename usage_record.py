"""One usage-record shape for every run source in the estate. homelab#93.

Vendored verbatim into harness-bench (same pattern as eval_logger.py) rather than shared as a
package dependency -- this stays a single dependency-free file on purpose.

`usd` and `gpu_seconds` are the two fields that are legitimately absent for a whole cost class,
not merely unmeasured: `agy` reports no dollar figure at all (pure plan quota), and nothing that
never touches local hardware has a GPU second to report. A schema that required either would be
unable to represent the estate's largest non-Claude spender.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Optional

# Mirrored from harness_bench.core.config.COST_CLASSES, not re-derived -- that repo already made
# this distinction and lives with it.
COST_CLASSES = ("free_local", "metered", "subscription_quota", "unavailable")


@dataclass(frozen=True)
class UsageRecord:
    source: str  # the emitting system, e.g. "harness-bench", "agy-cli", "interactive-session"
    harness: str
    model: str
    cost_class: str  # one of COST_CLASSES
    tokens_in: int
    tokens_out: int
    tokens_thinking: int = 0
    tokens_cache_read: int = 0
    tokens_cache_write: int = 0
    usd: Optional[float] = None
    wall_clock_s: float = 0.0
    gpu_seconds: Optional[float] = None

    def __post_init__(self):
        if self.cost_class not in COST_CLASSES:
            raise ValueError(f"unknown cost_class {self.cost_class!r}, expected one of {COST_CLASSES}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UsageRecord":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

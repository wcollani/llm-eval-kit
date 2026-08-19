"""One definition of what a dropped tool call looks like, shared by two repos.

Imports nothing but `re` on purpose: harness-bench vendors this file **verbatim** (the same
way it vendors `eval_logger.py`), and it must not drag deepeval, litellm or anything else
across with it. Keep it dependency-free or the copy stops being possible.

Why this is shared rather than duplicated: llm-eval-kit asks "did the model emit the call"
over one request/response, harness-bench asks it over a multi-turn trial, and both need the
same answer to the same question — *did the model choose a tool and did the plumbing drop
it?* Two copies of these regexes drift silently, and when they drift both repos keep
reporting confident numbers computed from different rules. That is the exact failure class
both tools exist to catch.
"""

import re

# The eight values ToolCallMetric.failure_mode can take. harness-bench's `trial_class.label`
# is a superset of this vocabulary, not a rename — see harness_bench/classify.py.
TOOL_CALL_FAILURE_MODES = (
    "ok",
    "no_call",
    "unparsed_call",
    "wrong_tool",
    "bad_arguments",
    "unwanted_call",
    "no_expectation",
    "error",
)

# A tool call the model emitted as *text* that the server's parser never turned into a
# structured call. Each pattern is a real emission shape observed in this homelab, not a
# guess:
#
#   <tool_call>       qwen2.5-coder's own chat template declares these tags. The model never
#                     emits them, so Ollama's parser yields nothing — but other models in the
#                     Qwen lineage do emit them, and then the wrapper survives into content.
#   {"name": ...,     qwen2.5-coder's actual emission: a bare call object in the content
#    "arguments": …}  field. Measured 0/3 parsed at 7b, 14b and 32b.
#   {"function": {…}} the OpenAI-shaped variant of the same thing.
#   <function=…>      added 2026-08-19. Observed from qwen3-coder:30b-a3b under both omp
#                     (`<function=glob>`) and cline (`<function=read_files>`) in harness-bench
#                     round 2a, with zero parsed tool calls. The three patterns above all miss
#                     it, so both repos were counting these as the model declining to act.
#   <|tool_call|>     the pipe-delimited special-token form; some GGUF conversions leave the
#                     literal text in place when the template and the tokenizer disagree.
#
# Precision matters more than recall here: a false positive relabels a genuine model failure
# as a plumbing bug, which is worse than missing one, because it manufactures a reason to
# stop investigating. Measured across 549 harness-bench trials, these patterns fire zero
# times on any trial that actually edited a file.
UNPARSED_PATTERNS = (
    re.compile(r"<tool_call>", re.IGNORECASE),
    re.compile(r'\{\s*"name"\s*:\s*".+?"\s*,\s*"arguments"\s*:', re.DOTALL),
    re.compile(r'\{\s*"function"\s*:\s*\{', re.DOTALL),
    re.compile(r"<function\s*=", re.IGNORECASE),
    re.compile(r"<\|?tool_call", re.IGNORECASE),
)


def looks_like_unparsed_call(text: str) -> bool:
    """True if `text` contains a tool call the parser should have extracted and did not."""
    return any(p.search(text or "") for p in UNPARSED_PATTERNS)


def unparsed_call_evidence(text: str) -> list[str]:
    """Which patterns matched, and where — for a classifier that has to show its work.

    Returns [] when nothing matched, so it doubles as `looks_like_unparsed_call`. The offset
    is included because these decisions get audited against a 3KB clipped transcript weeks
    later, and "it matched somewhere" is not enough to re-check by hand.
    """
    hits = []
    for pattern in UNPARSED_PATTERNS:
        match = pattern.search(text or "")
        if match:
            hits.append(f"matched {pattern.pattern!r} at offset {match.start()}")
    return hits

# Security battery results

`examples/security-review.yaml`, 11 cases x 3 repeats. Scored by `ToolCallMetric` --
exact match on a fixed `vulnerability_class` vocabulary, no judge model.

## 2026-08-19 (`security-battery-2026-08-19.json`)

| Model | Detection (7 vulnerable) | False positives (4 clean) | Overall |
|---|---|---|---|
| `claude-cli/sonnet` | **7 / 7** | **0** | 1.000 |
| `claude-cli/haiku` | **7 / 7** | **1** | 0.909 |

**Detection is identical. The entire difference is false-positive rate**, which is the
result worth having and the reason half this battery is clean code. A battery of only
vulnerable files would have scored both models 1.000 and settled nothing.

Haiku's single failure is `clean-subprocess-list`: `subprocess.run([...], shell=False)`
with the input interpolated into a filename argument. That is safe -- there is no shell,
and argv is a list, so `;` or `$()` in the name is a literal filename. Haiku flagged it
anyway, pattern-matching "subprocess plus user input" over reading what the call actually
does. The other three lookalikes -- parameterised query, resolved-and-checked path,
environment token -- it correctly left alone.

### What this justifies, and what it does not

It justifies the Security Review lane running sonnet: a reviewer that cries wolf on safe
code trains people to ignore it, and that is the one axis where the tiers differ here.

It does not justify much else. Four clean cases at three repeats is a small sample, and a
single false positive is one observation, not a rate. If this lane's cost ever needs to
come down, the honest next step is widening the clean set -- more lookalikes, more
classes -- rather than re-running this one and hoping.

And it says nothing about the lane itself. This measures recognising a vulnerability in a
file put directly in front of the model. The lane is agentic: it chooses what to open and
reasons about reachability across a worktree. An agentic security fixture in harness-bench
remains the follow-up, and no model should be promoted to the lane on this file alone.

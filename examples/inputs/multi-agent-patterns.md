# Multi-Agent Patterns

A reference guide to common patterns for building systems with multiple cooperating LLM agents.

## Generator-Critic-Refiner

Three agents in a pipeline: a Generator produces a first draft, a Critic reviews it against criteria, and a Refiner incorporates the feedback into a final output. Useful for content creation, code review, and document editing.

**When to use:** Output quality benefits from a review loop. The domain has clear quality criteria (accuracy, tone, completeness).

**Trade-off:** 3x the LLM calls and latency of a single-agent approach.

## Mob of Experts

An Orchestrator generates distinct expert personas, fans out to multiple Generator-Critic-Refiner pipelines (one per persona), then synthesizes the outputs. Produces higher-quality outputs than a single pipeline at the cost of significant compute.

**When to use:** Creative or analytical tasks where diverse perspectives improve output (research synthesis, architecture review, content for different audiences).

**Trade-off:** N×3 LLM calls where N is the number of expert pipelines. Sequential execution on a single GPU can be slow.

## Orchestrator-Subagent

An Orchestrator breaks a complex task into sub-tasks and delegates them to specialized Subagents. Subagents operate independently and return structured results to the Orchestrator for synthesis.

**When to use:** Tasks that decompose naturally into independent sub-problems (alert triage: one subagent handles metrics queries, another handles log queries, orchestrator synthesizes).

**Trade-off:** Requires careful prompt engineering to ensure subagent outputs are structured enough for the orchestrator to reliably synthesize.

## Single Agent with Tool Use

A single agent is given access to tools (functions, APIs, search) and reasons about which tools to call in a ReAct (Reasoning + Acting) loop. Simpler to build than multi-agent but limited by single-model reasoning capacity.

**When to use:** Tasks that require external data retrieval but don't benefit from parallelism or diverse perspectives.

**Trade-off:** All reasoning happens in one model context window. Complex multi-step tasks can exceed context limits.

## Evaluation Patterns

- **LLM-as-Judge:** A separate judge model scores outputs against criteria. Enables automated quality scoring without ground truth labels.
- **Execution Metric:** For code tasks, actually run the output and score based on exit code. More reliable than LLM scoring for functional correctness.
- **Human-in-the-Loop:** Gate high-stakes actions behind human approval. The agent proposes, a human confirms.

# Experiment Ideas

Proposed experiments to add to `examples/`. Each entry describes what to test, why it's interesting, and what a good `expected_output_criteria` looks like.

---

## Context Window & Long-Context

**Context degradation / needle-in-haystack**
Use the bundled `examples/inputs/haystack_1k.txt` and `haystack_10k.txt` to test retrieval accuracy as context grows. Hide a specific fact in the haystack and ask models to find it. Run the same case at `num_ctx: 4096`, `16384`, and `32768` to measure score vs. context size.

Pairs with `generate_haystack.py` to generate haystacks at any size. `generate_realistic_haystack.py` can build a haystack from a real codebase for code-search retrieval tests.

**Instruction following under long context**
Embed formatting instructions (e.g. "respond in exactly 3 bullet points") at different positions in a long prompt (start, middle, end). Test whether models follow instructions consistently regardless of position. Models often drop instructions buried in the middle of long prompts.

---

## Code Generation

**Python function generation with execution scoring**
Ask models to implement a specific function signature with documented behavior. Score with `ExecutionMetric` (does it run?) and `GEval` (does it handle edge cases?). Good showcase of the dual-metric approach: a function can execute cleanly but still be logically wrong.

Example task: implement a rate limiter, a retry decorator, or a simple LRU cache.

**Multi-language output**
Generate SQL queries, regex patterns, JSON Schema definitions, or shell one-liners from natural language descriptions. GEval criteria can check for structural correctness (e.g. "the SQL must use a JOIN, must not use SELECT *").

**Refactor quality comparison**
Extend `examples/code-refactor.yaml` with a second test case that uses a more complex input (a script with nested logic, global state, or mixed concerns). Compare small models (7B) vs. large (14B+) on refactor quality — a useful benchmark for model selection.

---

## Reasoning Model Studies

**Reasoning vs. standard on complex tasks**
Run the same PromQL generation or alert triage case with `deepseek-r1:14b` (reasoning model, strips `<think>` tags) and `qwen2.5-coder:14b` (standard) and compare scores. The `<think>` stripping in `agent_task()` already handles this — it just needs a dedicated experiment YAML.

**Chain-of-thought prompting ablation**
Same model, same task, two system prompts: one direct ("return only the query") and one CoT ("think step by step, then return the query"). Does CoT improve GEval scores for complex tasks? Does it hurt latency enough to matter?

**Reasoning model judge bias**
Use a reasoning model as the `judge_model` vs. a standard model. Compare whether reasoning judges give more consistent or more lenient scores. Relevant because judge model choice significantly affects GEval scores.

---

## Vision Models

**Image description baseline**
Pass the same image to `llava:13b`, `moondream:latest`, and `llama3.2-vision:11b` via the existing base64 image path in `cli.py`. Score on accuracy and detail. The infrastructure supports this — just needs a bundled example image and YAML. A Grafana dashboard screenshot or a chart with labeled anomalies would be a good test image.

**Chart reading**
Give models a PNG of a time-series graph (e.g. a Prometheus chart) and ask them to describe any visible anomalies. This is the "vision monitoring" project idea from the homelab — useful as a standalone benchmark before committing to that build.

---

## Prompt Engineering

**System prompt ablation**
Run the same task with three system prompts: none, generic ("you are a helpful assistant"), and task-specific. How much does a well-crafted system prompt move GEval scores on summarization or code tasks? Useful for justifying prompt engineering effort.

**Output format constraint compliance**
Test strict format instructions: "return only JSON with keys X, Y, Z", "respond in exactly 3 bullet points", "use no more than 50 words". GEval criteria can check for format compliance. Smaller models often fail these constraints even when they get the content right.

**Few-shot vs. zero-shot**
Add 1-3 worked examples to the `task_prompt` for a code or triage task and compare scores against the zero-shot equivalent. Most practitioners assume few-shot helps — worth measuring the delta on specific local models.

---

## Tool Calling

**Tool-call format reliability across a model family**
Run `workflow: tool_calling` over every size in a family (e.g. 7B / 14B / 32B) against the same
tool schemas. The interesting outcome is not the score but the `ToolCallFailureMode`: a whole
family can share an `unparsed_call` defect regardless of size, which no amount of scaling fixes
and which text-based scoring will not reveal. Pair with `arguments_contains` so a legitimately
varying query or path doesn't count as a failure.

**Transport parity**
Run the same tool-calling experiment twice, once with `ws/`/`sre/` prefixes (OpenAI-compat `/v1`)
and once with `direct_ws/`/`direct_sre/` (native `/api/chat`), then `compare` the two results
files. Any per-model delta is a transport or parser difference rather than a model capability
difference — worth pinning before trusting either number.

**Tool count and context pressure**
Offer 1, 5, and 20 tool schemas at a fixed `num_ctx` and watch where tool selection accuracy
falls off. Real agents present far more tools than a benchmark usually does, and schemas consume
context before the task prompt is even read.

**Prompt explicitness**
The same tool and task, phrased as an implicit question ("what is the CPU usage of X?") vs. an
explicit instruction ("call get_prometheus_metric for X"). A model that only calls tools when
told to is unsuitable for an autonomous lane, and the gap between the two phrasings measures that
directly.

## Multi-Agent Pipeline Quality

**Pipeline depth comparison**
Compare single-agent vs. Generator-Critic-Refiner vs. Mob of Experts on the same blog generation task (the examples directory already has all three). How much does each additional agent stage improve GEval score, and what's the latency cost? A concrete Pareto plot of quality vs. compute.

**Critic model ablation**
In the Generator-Critic-Refiner pipeline, swap out only the critic model (keep generator and refiner fixed) and measure final output quality. Tests whether the critic model is actually the bottleneck in pipeline quality.

**Subagent specialization**
In `multi_agent_triage`, test whether using a code-specialized model (`qwen2.5-coder`) as the subagent (for PromQL/LogQL) vs. a general model (`llama3.1`) measurably improves the quality of generated queries. Hypothesis: specialized models should score higher on the PromQL/LogQL sub-tasks.

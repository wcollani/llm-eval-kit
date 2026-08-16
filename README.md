# llm-eval-kit

A lightweight, YAML-driven CLI for evaluating LLM agents against criteria. Runs against local Ollama models (or any OpenAI-compatible endpoint), scores outputs with [DeepEval](https://github.com/confident-ai/deepeval), and emits OTLP traces for every LLM call.

No cloud account required. No framework lock-in. Define an experiment in YAML, run it, get scored JSON results and traces.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.11+. A local [Ollama](https://ollama.com) instance or [LiteLLM](https://github.com/BerriAI/litellm) proxy is expected at `http://localhost:4000/v1` by default.

## Quickstart

```bash
# Pull a model if you don't have one
ollama pull llama3.1:8b

# Copy and edit environment config
cp .env.example .env

# Run the simplest example
python cli.py examples/summarization.yaml
```

Results are written to `results/<experiment_name>_<timestamp>.json`.

## Writing Experiments

An experiment is a YAML file with three required sections:

```yaml
experiment_name: "My Experiment"
workflow: "single_agent"            # see Workflow Types below
system_prompt: "You are a ..."

models_to_test:
  - "ollama/llama3.1:8b"
  - "ollama/qwen2.5-coder:14b"

judge_model: "ollama/qwen2.5-coder:14b"

test_cases:
  - name: "My Test Case"
    input_file: "path/to/input.txt"
    task_prompt: "Do X with the input."
    expected_output_criteria: "The output must contain Y. It must not contain Z."
```

`expected_output_criteria` is passed directly to DeepEval's GEval metric as the scoring rubric. The judge model evaluates the actual output against it and returns a 0–1 score with a reason.

## Workflow Types

| `workflow` | Description | Required YAML fields |
|---|---|---|
| `single_agent` (default) | One model per test case | `models_to_test` |
| `multi_agent_blog_gen` | Generator → Critic → Refiner pipeline | `pipeline_combinations` |
| `mob_of_experts` | Orchestrator fans out to two parallel Gen-Crit-Ref pipelines, then synthesizes | `mob_combinations` |
| `multi_agent_triage` | Subagent generates PromQL + LogQL; orchestrator synthesizes remediation plan | `orchestrator_models`, `subagent_model` |
| `tool_calling` | Offers real tool schemas and scores the tool calls the model emits | `models_to_test`, `tools` |
| `embedding_quality` | Ranks a correct passage against distractors by embedding similarity | `models_to_test`, per-case `query`/`correct` |

See `examples/` for a working YAML for each workflow type.

Any experiment can add `repeats: N` to run every case N times per model instead of once and
report an average alongside the per-sample results — see [Repeats](#repeats) below. Useful
whenever the model's behavior isn't deterministic, which tool calling in particular often isn't.

### Tool calling

`tool_calling` sends a real `tools` array and scores what comes back with `ToolCallMetric` — no
judge model involved, since either the right tool arrived with the right arguments or it did not.

```yaml
workflow: "tool_calling"
num_ctx: 16384
tools:                      # inline schemas, or a path to a .json/.yaml file
  - type: function
    function:
      name: get_prometheus_metric
      description: Query a Prometheus metric
      parameters:
        type: object
        properties:
          query: {type: string}
        required: ["query"]
test_cases:
  - name: "Implicit metric lookup"
    task_prompt: "What is the CPU usage of container api-server-1 right now?"
    expected_tool_calls:
      - name: get_prometheus_metric
        arguments_contains:        # substring match, for values that vary
          query: "api-server-1"
```

Use `arguments` for exact matches (string comparison is case-insensitive) and
`arguments_contains` for values that legitimately vary between runs, like a generated query or a
path. `input_file` is optional for this workflow — a `task_prompt` alone is enough — and
`expected_output_criteria` is optional too; supply it only if you also want the prose graded.

`ToolCallMetric` reports *how* a case failed, because the fixes differ:

| `ToolCallFailureMode` | Meaning |
|---|---|
| `ok` | Expected call(s) emitted and parsed. |
| `no_call` | Model answered in prose and never attempted a tool call. |
| `unparsed_call` | Model emitted a tool call **in its content** that the server never parsed into structured `tool_calls`. The model chose correctly; the plumbing dropped it — usually a chat template/parser mismatch for that tag. An agent sees no tool call either way. |
| `wrong_tool` | A tool was called, but not the expected one. |
| `bad_arguments` | Right tool, wrong or missing arguments (scores 0.5). |
| `unwanted_call` | A case marked `expect_no_tool_call: true` got a tool call anyway. |

Set `expect_no_tool_call: true` (instead of `expected_tool_calls`) on a case the model should
answer directly. Over-eager tool use is its own failure — an agent that reaches for a tool on
every turn burns context and latency on questions it could have answered.

`unparsed_call` is worth calling out: a model can advertise `tools` in its Ollama `capabilities`,
pick the right tool, produce perfect arguments, and still be unusable in an agent because the
call never parses. Text-based scoring rates that output highly. This metric does not.

### Embedding quality

`embedding_quality` scores embedding models on retrieval: for each case, it embeds a `query`
alongside a `correct` passage and some `distractors`, then checks whether the correct passage's
embedding is the one closest to the query's — `EmbeddingRetrievalMetric`, no judge model,
`1 / rank` (1.0 for first place, 0.5 for second, etc.) rather than pass/fail.

```yaml
workflow: "embedding_quality"
models_to_test:
  - "sre/bge-large:335m-en-v1.5-fp16"
  - "ws/nomic-embed-text"
test_cases:
  - name: "PromQL vs LogQL tool"
    query: "How do I check container CPU usage?"
    correct: "query_promql: Execute a PromQL query against Prometheus to retrieve metrics."
    distractors:
      - "query_logql: Execute a LogQL query against Loki to retrieve logs."
      - "send_notification: Send a push notification to the operator via ntfy."
```

`models_to_test` here are embedding model tags, not chat models — routed through the same
`ws/`/`sre/`/`direct_ws/`/`direct_sre/`/`ollama/` prefixes as everything else, just hitting
Ollama's `/api/embed` (native) or `/v1/embeddings` (OpenAI-compat) instead of chat completions.
`ExecutionMetric`, `GEval`, and `ToolCallMetric` are always `null` for this workflow — none of
them apply to a model that only produces vectors.

### Repeats

Add `repeats: N` to any experiment to run each case N times per model rather than once:

```yaml
repeats: 3
```

The results JSON then carries a `samples: [...]` list (one entry per run, same shape a
single-sample result would have) plus aggregated top-level fields: numeric scores are averaged,
`latency_sec` is averaged, `tokens` is summed across samples. For `ToolCallMetric` specifically,
the aggregate also includes `ToolCallPassRate` — the fraction of samples that scored a clean
`ok` — because a mean score alone can't tell "always half-right" apart from "right half the time,
wrong half the time," and those need different fixes. `ToolCallFailureMode` becomes
`mixed(mode_a,mode_b)` when samples disagree on how a case failed.

`repeats: 1` (the default) produces exactly the same result shape as before this existed — no
`samples` key, no aggregation. This matters most for `tool_calling`: the same model on the same
prompt can call a tool on one sample and answer in prose on the next, and a single sample can't
tell you which is typical.

## Routing Models

The model name prefix controls which endpoint is used:

| Prefix | Routes to |
|---|---|
| `ollama/<model>` | `LITELLM_API_BASE` proxy (default) |
| `openai/<model>` | `LITELLM_API_BASE` proxy, OpenAI-compat |
| `ws/<model>` | `OLLAMA_WS_URL` directly (e.g. a workstation GPU) |
| `sre/<model>` | `OLLAMA_SRE_URL` directly (e.g. a secondary GPU node) |
| `direct_ws/<model>` | Native Ollama `/api/chat` on `OLLAMA_WS_URL` |
| `direct_sre/<model>` | Native Ollama `/api/chat` on `OLLAMA_SRE_URL` |

For a single-machine setup with Ollama running locally, use `ollama/<model>` and point `LITELLM_API_BASE` at your LiteLLM proxy, or set it to `http://localhost:11434/v1` directly.

## Tracing

Every LLM call is automatically traced via LiteLLM's OTEL integration. Set `OTEL_EXPORTER_OTLP_ENDPOINT` to send spans to your backend:

- **[Jaeger](https://www.jaegertracing.io/)** — `http://localhost:4317/v1/traces`
- **[Arize Phoenix](https://phoenix.arize.com/)** — `http://localhost:6006/v1/traces`
- **Grafana Alloy** — `http://alloy-host:4317/v1/traces`

If the endpoint is unreachable, evaluation still runs — traces are just dropped.

## Results

Each run writes `results/<safe_name>_<timestamp>.json`:

```json
{
  "experiment_name": "...",
  "runs": [
    {
      "pipeline": {"model": "ollama/llama3.1:8b"},
      "case_name": "My Test Case",
      "latency_sec": 4.21,
      "tokens": {"prompt_tokens": 312, "completion_tokens": 89, "total_tokens": 401},
      "actual_output": "...",
      "tool_calls": [],
      "scores": {
        "ExecutionMetric": 1.0,
        "ExecutionReason": "Code executed successfully with Exit Code 0.",
        "GEval": 0.85,
        "GEvalReason": "The output mentions patterns A and B but omits C.",
        "ToolCallMetric": null,
        "ToolCallReason": "Skipped (no expected_tool_calls)",
        "ToolCallFailureMode": null,
        "EmbeddingRetrievalMetric": null,
        "EmbeddingRetrievalReason": null
      }
    }
  ]
}
```

With `repeats: N` (N > 1), each run additionally carries `"repeats": N` and `"samples": [...]` —
a list of N per-sample dicts in this same shape — and the top-level `scores` become averages
across those samples (see [Repeats](#repeats)).

`ExecutionMetric` only applies to code generation tasks (runs the output as Python and checks exit code). For non-code tasks it will score 0 — rely on `GEval` for those.

`ToolCallMetric` is `null` unless the case defines `expected_tool_calls`, and `GEval` is `null`
unless it defines `expected_output_criteria` — a judge is never asked to grade against an empty
rubric. `tool_calls` holds the parsed calls, normalized to `{"name", "arguments"}` regardless of
which transport produced them.

Multi-agent workflow artifacts (blog drafts, mob expert outputs) are saved to `results/artifacts/`.

## Examples

| File | Workflow | What it tests |
|---|---|---|
| `examples/summarization.yaml` | single_agent | Basic summarization quality |
| `examples/single-agent-blog.yaml` | single_agent | Blog generation, compare models |
| `examples/multi-agent-blog.yaml` | multi_agent_blog_gen | Generator-Critic-Refiner pipeline |
| `examples/mob-of-experts.yaml` | mob_of_experts | Mob of experts synthesis quality |
| `examples/code-refactor.yaml` | single_agent | Bash → Python refactor with execution scoring |
| `examples/promql-generation.yaml` | single_agent | PromQL query generation accuracy |
| `examples/logql-summarization.yaml` | single_agent | Log summarization / root cause identification |
| `examples/alert-triage.yaml` | multi_agent_triage | 3-phase alert triage pipeline |
| `examples/tool-calling.yaml` | tool_calling | Whether a model emits parseable tool calls |
| `examples/embedding-quality.yaml` | embedding_quality | Whether an embedding model ranks the right passage first |
| `examples/homelab/` | various | Homelab-specific reference experiments |

## Utilities

- `generate_haystack.py` — generates synthetic needle-in-haystack test data at specified sizes
- `generate_synthetic_codebase.py` — generates dummy Python files for context window tests
- `swarm_profiler.py` — measures concurrency and latency across model endpoints
- `vuln_scanner_chunker.py` — chunks repo files into prompt-sized segments for batch tasks

## Grafana Dashboard

`grafana/dashboard.json` is an importable Grafana dashboard for visualizing multi-agent experiment results when traces are flowing to a Grafana stack.

### Score & Latency Metrics

If `PROMETHEUS_PUSHGATEWAY_URL` is set, every test case pushes `llm_eval_score{model, experiment, case, metric}` (one series per `ExecutionMetric`/`GEval`/`ToolCallMetric`/`EmbeddingRetrievalMetric`) and `llm_eval_latency_ms{model, experiment, case}` to the Pushgateway. With `repeats: N`, the pushed score is the mean across samples. This requires a Pushgateway scraped by Prometheus — pushing is skipped silently if the env var is unset or the gateway is unreachable.

`grafana/eval-metrics-dashboard.json` visualizes these series: a quality/latency scatter (Pareto frontier per model), score trend over time per experiment, a per-model win-rate table, and a latest-run summary panel. Import it into Grafana, or drop it into a provisioning folder pointed at a Prometheus datasource.

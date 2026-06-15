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

See `examples/` for a working YAML for each workflow type.

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
      "scores": {
        "ExecutionMetric": 1.0,
        "ExecutionReason": "Code executed successfully with Exit Code 0.",
        "GEval": 0.85,
        "GEvalReason": "The output mentions patterns A and B but omits C."
      }
    }
  ]
}
```

`ExecutionMetric` only applies to code generation tasks (runs the output as Python and checks exit code). For non-code tasks it will score 0 — rely on `GEval` for those.

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
| `examples/homelab/` | various | Homelab-specific reference experiments |

## Utilities

- `generate_haystack.py` — generates synthetic needle-in-haystack test data at specified sizes
- `generate_synthetic_codebase.py` — generates dummy Python files for context window tests
- `swarm_profiler.py` — measures concurrency and latency across model endpoints
- `vuln_scanner_chunker.py` — chunks repo files into prompt-sized segments for batch tasks

## Grafana Dashboard

`grafana/dashboard.json` is an importable Grafana dashboard for visualizing multi-agent experiment results when traces are flowing to a Grafana stack.

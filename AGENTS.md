# llm-eval-kit AI Agent Context (AGENTS.md)

This file is the primary context layer for AI agents operating in this repository.

## 1. Repository Purpose & Scope

A lightweight, YAML-driven CLI for evaluating LLM agents against criteria. Runs against local Ollama (or any OpenAI-compatible endpoint), scores outputs with [DeepEval](https://github.com/confident-ai/deepeval) GEval, and emits OTLP traces for every LLM call.

**This is a standalone public portfolio tool** — it has no runtime dependency on the homelab. The homelab uses it for experiment scoring and prompt optimization, but it runs fine without any homelab infrastructure.

- **Language:** Python 3.11+
- **Install:** `pip install -e .` (local) or `pip install llm-eval-kit` (once published to PyPI)
- **Entry point:** `llm-eval <experiment.yaml>` (console script) or `python cli.py <experiment.yaml>`

## 2. Experiment YAML Format

```yaml
experiment_name: "My Experiment"
workflow: "single_agent"           # single_agent | multi_agent_blog_gen | mob_of_experts | multi_agent_triage
system_prompt: "You are a ..."

models_to_test:
  - "ws/qwen2.5-coder:7b"          # prefix controls routing (see below)

judge_model: "ws/qwen2.5-coder:14b"

test_cases:
  - name: "My Test"
    input_file: "path/to/input.txt"
    task_prompt: "Do X with the input."
    expected_output_criteria: "The output must contain Y and not contain Z."
```

Results go to `results/<experiment_name>_<timestamp>.json`. OTLP traces go to `$OTEL_EXPORTER_OTLP_ENDPOINT` (default: `localhost:4319`).

## 3. Model Routing Prefixes

| Prefix | Routes to |
|--------|-----------|
| `ollama/<model>` | `LITELLM_API_BASE` proxy |
| `ws/<model>` | `OLLAMA_WS_URL` directly (workstation GPU) |
| `sre/<model>` | `OLLAMA_SRE_URL` directly (SRE machine GPU) |
| `direct_ws/<model>` | Native Ollama `/api/chat` on `OLLAMA_WS_URL` |

## 4. Key Source Files

| File | Role |
|------|------|
| `cli.py` | Entry point — parses YAML, dispatches to workflow |
| `eval_logger.py` | Prometheus metrics push (`llm_eval_score`, `llm_eval_latency_ms`) |
| `examples/` | Working YAML for each workflow type |

Do not add new workflow types without a corresponding example YAML in `examples/`.

## 5. Homelab Integration

When running inside the homelab:
- Set `PROMETHEUS_PUSHGATEWAY_URL` to push metrics to the Pushgateway at `http://192.168.99.178:9091`
- Set `OTEL_EXPORTER_OTLP_ENDPOINT=http://192.168.99.178:4319` to send traces to Grafana Alloy → Arize Phoenix
- Production experiment configs live in `homelab-platform/services/dagu/dags/experiments-config/` as Dagu DAG YAMLs

## 6. CI/CD & Releases

**This is a public repo — CI runs on GitHub-hosted runners only.**

- `release.yml` triggers on `v*` tags and creates a GitHub Release with auto-generated notes
- To cut a release: `git tag v0.2.0 && git push --tags`
- Homelab can pin installs to a release tag: `pip install git+https://github.com/wcollani/llm-eval-kit.git@v0.2.0`

No PyPI publish yet. Use git+https installs for now.

## 7. Coding Rules

- All scoring is done via DeepEval `GEval` metric — do not introduce other scoring libraries without a compelling reason
- Metrics pushed to Prometheus must use the existing `eval_logger.py` module — do not add ad-hoc push logic in `cli.py`
- `DEEPEVAL_TELEMETRY_OPT_OUT=YES` should always be set in Docker/Dagu contexts to prevent DeepEval from phoning home

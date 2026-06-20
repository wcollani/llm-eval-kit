# Roadmap

Engineering improvements to llm-eval-kit, organized by phase. Phase 1 focuses on developer experience; later phases add evaluation depth and observability.

---

## Phase 1 — Polish & Developer Experience

**PyPI packaging**
Make the CLI installable via `pip install llm-eval-kit` or `pipx install llm-eval-kit`. Add `pyproject.toml` with a `[project.scripts]` entry so `llm-eval` works as a shell command. Currently requires cloning the repo.

**`--dry-run` flag** ✅ *done*
Validate a YAML experiment and print what would run (models, test cases, workflow, file paths) without calling any LLM. Catches missing input files and schema errors before a long run.

**GitHub Actions CI**
Add `.github/workflows/ci.yml` that lints with `ruff` and runs a smoke test: `python cli.py examples/summarization.yaml` against a mocked LLM response (monkeypatched `litellm.completion`). Currently there are no automated checks on PRs.

**`--output` flag**
Let users specify a custom output path for the results JSON instead of always writing to `results/`. Useful for CI pipelines that need results at a known path.

---

## Phase 2 — Evaluation Depth

**Parallel model evaluation** ✅ *done*
Run multiple models concurrently via `asyncio.gather` instead of a sequential loop. With 8 models and 30s average latency, sequential takes 4+ minutes; parallel takes ~30s. Enable with `--parallel`.

**Custom metric plugins**
Allow users to define a custom `BaseMetric` subclass in a Python file and reference it from the YAML (`custom_metric: path/to/metric.py`). Currently limited to `ExecutionMetric` and `GEval`. Would let users add regex-match metrics, length checks, JSON schema validation, etc.

**Result comparison CLI command** ✅ *done*
`python cli.py compare results/run_a.json results/run_b.json`. Outputs a table of score and latency differences per model/case. Useful when iterating on prompts or swapping models.

**`--resume` flag** ✅ *done*
Skip test cases that already have results in the most recent JSON file for the experiment. Long experiments (10+ models × 5 cases) can fail partway through; resuming saves time and tokens.

**LiteLLM config file support**
Add `--litellm-config path/to/config.yaml` to load a LiteLLM router config for complex routing setups (load balancing, fallbacks, rate limits). Currently the flag exists as a dead stub — wire it up or remove it entirely.

---

## Phase 3 — Observability & Reporting

**Prometheus metrics push**
After each run, push `llm_eval_score{model, experiment, case}` and `llm_eval_latency_ms{model, experiment, case}` to a Prometheus Pushgateway. Enables tracking score trends over time as models or prompts change. The Grafana dashboard in `grafana/` is the natural consumer.

**Markdown / HTML report generation**
Add a `--report` flag that generates a human-readable summary alongside the JSON. A markdown table of model × case scores with latency is more scannable than raw JSON for sharing results.

**Span enrichment**
Add experiment name, model name, and case name as OTEL span attributes on each LLM call. Currently LiteLLM emits basic spans; enriching them makes filtering in Jaeger/Phoenix much more useful.

---

## Deferred / Stretch

- **Streaming output** — show model output token-by-token in the terminal during generation
- **Temperature/sampling sweeps** — YAML field to run the same case at multiple temperatures
- **`generate_realistic_haystack.py` documentation** — this script scans a real codebase to build context-degradation test data; it's useful but currently undocumented

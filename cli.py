#!/usr/bin/env python3
import typer
import os
import tempfile
import yaml
import time
import subprocess
import litellm
import asyncio
import re
import base64
import requests
import json

from eval_logger import setup_tracing, save_experiment_results, push_metrics_to_prometheus
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval, BaseMetric
from deepeval.models.base_model import DeepEvalBaseLLM

app = typer.Typer(help="Agent Testing CLI to run experiments with OTEL tracing")

# Configure litellm to emit OTEL spans for every LLM call
litellm.success_callback = ["otel"]
litellm.failure_callback = ["otel"]

def resolve_endpoint(model_name: str):
    """Parses prefix to determine api_base, actual model name, and if it's a direct connection.

    Supported prefixes:
      ollama/<model>      — route through LITELLM_API_BASE proxy
      ws/<model>          — direct to OLLAMA_WS_URL (e.g. a workstation GPU)
      sre/<model>         — direct to OLLAMA_SRE_URL (e.g. a secondary GPU node)
      direct_ws/<model>   — native Ollama /api/chat on OLLAMA_WS_URL
      direct_sre/<model>  — native Ollama /api/chat on OLLAMA_SRE_URL
      openai/<model>      — treated as OpenAI-compatible, routed through proxy
    """
    api_base = os.getenv("LITELLM_API_BASE", "http://localhost:4000/v1")
    is_direct = False

    if model_name.startswith("direct_ws/"):
        api_base = os.getenv("OLLAMA_WS_URL", "http://localhost:11434") + "/api/chat"
        model_name = model_name[10:]
        is_direct = True
    elif model_name.startswith("direct_sre/"):
        api_base = os.getenv("OLLAMA_SRE_URL", "http://localhost:11434") + "/api/chat"
        model_name = model_name[11:]
        is_direct = True
    elif model_name.startswith("ws/"):
        api_base = os.getenv("OLLAMA_WS_URL", "http://localhost:11434") + "/v1"
        model_name = model_name[3:]
    elif model_name.startswith("sre/"):
        api_base = os.getenv("OLLAMA_SRE_URL", "http://localhost:11434") + "/v1"
        model_name = model_name[4:]

    if model_name.startswith("ollama/"):
        if is_direct or "11434" in api_base:
            model_name = model_name[7:]

    proxy_model = f"openai/{model_name}" if not model_name.startswith("openai/") and not is_direct else model_name
    return proxy_model, api_base, is_direct

class CustomLiteLLM(DeepEvalBaseLLM):
    """Wrapper to allow DeepEval to use LiteLLM configured endpoints."""
    def __init__(self, model_name):
        self.model_name = model_name

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        proxy_model, api_base, is_direct = resolve_endpoint(self.model_name)
        if is_direct:
            raise NotImplementedError("DeepEval Judge models currently do not support direct routing.")
        res = litellm.completion(
            model=proxy_model,
            messages=[{"role": "user", "content": prompt}],
            api_base=api_base,
            api_key="sk-dummy",
            response_format={"type": "json_object"},
            timeout=1200
        )
        return res.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        proxy_model, api_base, is_direct = resolve_endpoint(self.model_name)
        if is_direct:
            raise NotImplementedError("DeepEval Judge models currently do not support direct routing.")
        res = await litellm.acompletion(
            model=proxy_model,
            messages=[{"role": "user", "content": prompt}],
            api_base=api_base,
            api_key="sk-dummy",
            response_format={"type": "json_object"},
            timeout=1200
        )
        return res.choices[0].message.content or ""

    def get_model_name(self):
        return self.model_name

class ExecutionMetric(BaseMetric):
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0
        self.reason = None
        self.success = False

    def measure(self, test_case: LLMTestCase):
        code = test_case.actual_output.strip()
        if code.startswith("```python"):
            code = code[9:]
        elif code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(code.strip())
            tmp_path = f.name

        try:
            subprocess.run(["python3", tmp_path], timeout=10, check=True, capture_output=True)
            self.score = 1.0
            self.success = True
            self.reason = "Code executed successfully with Exit Code 0."
        except subprocess.TimeoutExpired:
            self.score = 0.0
            self.success = False
            self.reason = "Code execution timed out."
        except subprocess.CalledProcessError as e:
            self.score = 0.0
            self.success = False
            self.reason = f"Code failed with exit code {e.returncode}. Stderr: {e.stderr.decode('utf-8')[:200]}"
        finally:
            os.unlink(tmp_path)

        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Programmatic Execution"

def agent_task(model_name: str, system_prompt: str, input_prompt: str, num_ctx: int = 4096):
    """Executes the agent task using litellm routing or direct ollama hit."""
    start_time = time.time()
    proxy_model, api_base, is_direct = resolve_endpoint(model_name)

    if is_direct:
        payload = {
            "model": proxy_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_prompt}
            ],
            "stream": False,
            "options": {
                "num_ctx": num_ctx
            }
        }
        res = requests.post(api_base, json=payload, timeout=1200)
        res.raise_for_status()
        data = res.json()
        latency = time.time() - start_time
        output = data.get("message", {}).get("content", "")
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        }
    else:
        response = litellm.completion(
            model=proxy_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_prompt}
            ],
            api_base=api_base,
            api_key="sk-dummy",
            num_ctx=num_ctx,
            timeout=1200
        )
        latency = time.time() - start_time
        output = response.choices[0].message.content or ""
        usage = response.usage.model_dump() if response.usage else {}

    # Strip <think> tags from reasoning model output before scoring
    output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
    return output, latency, usage

def multi_agent_triage_task(
    orchestrator_model: str,
    subagent_model: str,
    system_prompt: str,
    input_prompt: str,
    mock_promql_file: str = "examples/inputs/mock_promql_result.json",
    mock_logql_file: str = "examples/inputs/mock_logql_result.json",
):
    """Executes a 3-phase multi-agent triage pipeline.

    Phase 1: Subagent generates PromQL for the alert.
    Phase 2: Subagent generates LogQL using mock metric context.
    Phase 3: Orchestrator synthesizes both into a remediation plan.

    mock_promql_file / mock_logql_file point to bundled example data by default.
    Override via YAML fields of the same name for custom scenarios.
    """
    start_time = time.time()

    # Phase 1: Subagent generates PromQL
    promql_prompt = f"You are a metrics subagent. Generate a PromQL query for the container mentioned in this alert:\n{input_prompt}\nReturn ONLY the PromQL query. No markdown."
    promql_output, _, usage1 = agent_task(subagent_model, "You are a PromQL expert.", promql_prompt)

    # Phase 2: Subagent generates LogQL
    try:
        with open(mock_promql_file, "r") as f:
            mock_metrics = f.read()
    except FileNotFoundError:
        mock_metrics = "{}"

    logql_prompt = f"You are a logging subagent. The alert is:\n{input_prompt}\nThe metrics show:\n{mock_metrics}\nGenerate a Loki LogQL query to find errors for this container. Return ONLY the LogQL query. No markdown."
    logql_output, _, usage2 = agent_task(subagent_model, "You are a LogQL expert.", logql_prompt)

    # Phase 3: Orchestrator synthesizes
    try:
        with open(mock_logql_file, "r") as f:
            mock_logs = f.read()
    except FileNotFoundError:
        mock_logs = "{}"

    orchestrator_prompt = f"Alert Payload:\n{input_prompt}\n\nPromQL Query generated by subagent:\n{promql_output}\nMetrics Result:\n{mock_metrics}\n\nLogQL Query generated by subagent:\n{logql_output}\nLogs Result:\n{mock_logs}\n\nAnalyze this data and provide a remediation plan."
    final_output, _, usage3 = agent_task(orchestrator_model, system_prompt, orchestrator_prompt)

    latency = time.time() - start_time
    total_usage = {
        "completion_tokens": usage1.get("completion_tokens", 0) + usage2.get("completion_tokens", 0) + usage3.get("completion_tokens", 0),
        "prompt_tokens": usage1.get("prompt_tokens", 0) + usage2.get("prompt_tokens", 0) + usage3.get("prompt_tokens", 0),
        "total_tokens": usage1.get("total_tokens", 0) + usage2.get("total_tokens", 0) + usage3.get("total_tokens", 0)
    }

    combined_output = f"PromQL:\n{promql_output}\n\nLogQL:\n{logql_output}\n\nRemediation Plan:\n{final_output}"
    return combined_output, latency, total_usage

def multi_agent_blog_task(generator_model: str, critic_model: str, refiner_model: str, system_prompt: str, input_prompt: str):
    """Executes a Generator-Critic-Refiner pipeline for blog creation."""
    start_time = time.time()

    # Phase 1: Generator
    draft, _, usage1 = agent_task(generator_model, system_prompt, input_prompt)

    # Phase 2: Critic
    critic_prompt = f"Original Source:\n{input_prompt}\n\nDraft Blog Post:\n{draft}\n\nReview this draft and provide a bulleted list of critiques or missing information based on the source."
    critique, _, usage2 = agent_task(critic_model, "You are a strict blog editor.", critic_prompt)

    # Phase 3: Refiner
    refiner_prompt = f"Original Source:\n{input_prompt}\n\nDraft:\n{draft}\n\nCritiques:\n{critique}\n\nRewrite the final blog post incorporating these critiques."
    final_output, _, usage3 = agent_task(refiner_model, system_prompt, refiner_prompt)

    latency = time.time() - start_time
    total_usage = {
        "completion_tokens": usage1.get("completion_tokens", 0) + usage2.get("completion_tokens", 0) + usage3.get("completion_tokens", 0),
        "prompt_tokens": usage1.get("prompt_tokens", 0) + usage2.get("prompt_tokens", 0) + usage3.get("prompt_tokens", 0),
        "total_tokens": usage1.get("total_tokens", 0) + usage2.get("total_tokens", 0) + usage3.get("total_tokens", 0)
    }

    return final_output, latency, total_usage, draft, critique

def mob_of_experts_task(orchestrator_model: str, generator_model: str, critic_model: str, refiner_model: str, system_prompt: str, input_prompt: str):
    """Executes a Mob of Experts architecture: Orchestrator -> Fan-Out (Sequential) -> Synthesize."""
    start_time = time.time()
    total_usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}

    def update_usage(u):
        total_usage["completion_tokens"] += u.get("completion_tokens", 0)
        total_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        total_usage["total_tokens"] += u.get("total_tokens", 0)

    # Phase 1: Orchestrator generates distinct expert sub-prompts
    orch_prompt = f"Original Source/Instructions:\n{input_prompt}\n\nYou are the Orchestrator. Create TWO distinct 'Expert System Prompts' tailored to the requested persona. Expert A should focus on one aspect (e.g. structure/tone) and Expert B on another (e.g. depth/storytelling).\nFormat your output strictly as:\nEXPERT_A_PROMPT: <prompt>\nEXPERT_B_PROMPT: <prompt>"
    orch_out, _, u = agent_task(orchestrator_model, "You are a master AI Orchestrator.", orch_prompt)
    update_usage(u)

    expert_a_prompt = "You are an expert technical writer."
    expert_b_prompt = "You are an expert technical writer."
    if "EXPERT_A_PROMPT:" in orch_out and "EXPERT_B_PROMPT:" in orch_out:
        parts = orch_out.split("EXPERT_B_PROMPT:")
        expert_a_prompt = parts[0].replace("EXPERT_A_PROMPT:", "").strip()
        expert_b_prompt = parts[1].strip()

    # Phase 2: Sequential Fan-Out — Expert A pipeline
    draft_a, _, u_ga = agent_task(generator_model, expert_a_prompt, input_prompt)
    update_usage(u_ga)
    critic_a_prompt = f"Original Source:\n{input_prompt}\n\nDraft:\n{draft_a}\n\nProvide critiques based on this persona: {expert_a_prompt}"
    critique_a, _, u_ca = agent_task(critic_model, "You are a strict editor.", critic_a_prompt)
    update_usage(u_ca)
    refiner_a_prompt = f"Original Source:\n{input_prompt}\n\nDraft:\n{draft_a}\n\nCritiques:\n{critique_a}\n\nRewrite."
    final_a, _, u_ra = agent_task(refiner_model, expert_a_prompt, refiner_a_prompt)
    update_usage(u_ra)

    # Expert B pipeline
    draft_b, _, u_gb = agent_task(generator_model, expert_b_prompt, input_prompt)
    update_usage(u_gb)
    critic_b_prompt = f"Original Source:\n{input_prompt}\n\nDraft:\n{draft_b}\n\nProvide critiques based on this persona: {expert_b_prompt}"
    critique_b, _, u_cb = agent_task(critic_model, "You are a strict editor.", critic_b_prompt)
    update_usage(u_cb)
    refiner_b_prompt = f"Original Source:\n{input_prompt}\n\nDraft:\n{draft_b}\n\nCritiques:\n{critique_b}\n\nRewrite."
    final_b, _, u_rb = agent_task(refiner_model, expert_b_prompt, refiner_b_prompt)
    update_usage(u_rb)

    # Phase 3: Synthesis
    synth_prompt = f"Original Request:\n{input_prompt}\n\n--- EXPERT A DRAFT ---\n{final_a}\n\n--- EXPERT B DRAFT ---\n{final_b}\n\nSynthesize these two drafts into the ultimate final blog post that perfectly captures the requested persona."
    final_output, _, u_synth = agent_task(orchestrator_model, system_prompt, synth_prompt)
    update_usage(u_synth)

    latency = time.time() - start_time
    return final_output, latency, total_usage, final_a, final_b, orch_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_combo_id(combo: dict) -> str:
    return "_".join([v.replace("/", "-").replace(":", "-") for v in combo.values()])


def _load_resume(experiment_name: str) -> tuple[dict | None, set, str | None]:
    """Find the most recent results file for this experiment and return its data."""
    safe_name = "".join([c if c.isalnum() else "_" for c in experiment_name])
    results_dir = "results"
    if not os.path.isdir(results_dir):
        return None, set(), None
    matching = [
        f for f in os.listdir(results_dir)
        if f.startswith(safe_name + "_") and f.endswith(".json")
    ]
    if not matching:
        print("[*] --resume: no existing results file found, starting fresh")
        return None, set(), None
    latest_path = os.path.join(results_dir, sorted(matching)[-1])
    with open(latest_path) as f:
        data = json.load(f)
    completed: set[tuple[str, str]] = set()
    for run in data.get("runs", []):
        combo_id = _make_combo_id(run["pipeline"])
        completed.add((combo_id, run["case_name"]))
    print(f"[*] Resuming from {latest_path}: {len(completed)} cases already done")
    return data, completed, latest_path


def _dry_run_print(exp: dict, experiment_name: str, workflow: str, judge_model: str, combinations: list) -> None:
    print(f"[DRY-RUN] Experiment:  {experiment_name}")
    print(f"[DRY-RUN] Workflow:    {workflow}")
    print(f"[DRY-RUN] Judge model: {judge_model}")
    print(f"[DRY-RUN] Combos ({len(combinations)}):")
    for c in combinations:
        print(f"  - {_make_combo_id(c)}: {c}")
    test_cases = exp.get("test_cases", [])
    print(f"[DRY-RUN] Test cases ({len(test_cases)}):")
    for tc in test_cases:
        path = tc.get("input_file", "N/A")
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{status}] {tc['name']}: {path}")
    total_agent = len(combinations) * len(test_cases)
    print(f"[DRY-RUN] Estimated LLM calls: {total_agent} agent + {total_agent} judge = {total_agent * 2} total")


# ---------------------------------------------------------------------------
# Core evaluation coroutine (one combo / pipeline)
# ---------------------------------------------------------------------------

async def _eval_combo(
    combo: dict,
    exp: dict,
    judge_model: str,
    workflow: str,
    completed_keys: set,
    allow_code_execution: bool,
) -> list[dict]:
    """Run every test case for a single combo. Returns a list of run result dicts."""
    runs: list[dict] = []
    model_name = combo.get("model", combo.get("orchestrator", combo.get("generator", "pipeline")))
    combo_id = _make_combo_id(combo)
    experiment_name = exp.get("name", exp.get("experiment_name", "Unnamed Experiment"))
    print(f"\n>> Evaluating Pipeline/Model: {combo_id}")

    for case in exp.get("test_cases", []):
        if (combo_id, case["name"]) in completed_keys:
            print(f"   Skipping (already done): {case['name']}")
            continue

        try:
            if case["input_file"].lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                with open(case["input_file"], "rb") as f:
                    base64_image = base64.b64encode(f.read()).decode("utf-8")
                input_prompt = [
                    {"type": "text", "text": case.get("task_prompt", "")},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            else:
                with open(case["input_file"], "r") as f:
                    input_content = f.read()
                input_prompt = f"{case.get('task_prompt', '')}\n\nCode/Input:\n{input_content}"
        except FileNotFoundError:
            print(f"[!] Could not read input file {case['input_file']}")
            continue

        expected_output = case.get("expected_output_criteria", "")
        print(f"   Running case: {case['name']}...")

        if workflow == "multi_agent_blog_gen":
            actual_output, latency, usage, draft, critique = await asyncio.to_thread(
                multi_agent_blog_task,
                combo["generator"], combo["critic"], combo["refiner"],
                exp["system_prompt"], input_prompt,
            )
            artifact_dir = "results/artifacts"
            os.makedirs(artifact_dir, exist_ok=True)
            safe_case = case["name"].replace(" ", "_").replace("/", "-")
            artifact_path = os.path.join(artifact_dir, f"Blog_{combo_id}_{safe_case}.md")
            with open(artifact_path, "w") as af:
                af.write(f"# Pipeline: {combo_id}\n\n## Final V2 Blog Post\n\n{actual_output}\n\n---\n## Critic Feedback on V1\n\n{critique}")
            print(f"   [+] Saved artifact to {artifact_path}")

        elif workflow == "mob_of_experts":
            actual_output, latency, usage, draft_a, draft_b, orch_out = await asyncio.to_thread(
                mob_of_experts_task,
                combo["orchestrator"], combo["generator"], combo["critic"], combo["refiner"],
                exp["system_prompt"], input_prompt,
            )
            artifact_dir = "results/artifacts"
            os.makedirs(artifact_dir, exist_ok=True)
            safe_case = case["name"].replace(" ", "_").replace("/", "-")
            artifact_path = os.path.join(artifact_dir, f"Mob_{combo_id}_{safe_case}.md")
            with open(artifact_path, "w") as af:
                af.write(f"# Pipeline: {combo_id}\n\n## Final Synthesis\n\n{actual_output}\n\n---\n## Orchestrator Prompts\n\n{orch_out}\n\n---\n## Expert A Draft\n\n{draft_a}\n\n---\n## Expert B Draft\n\n{draft_b}")
            print(f"   [+] Saved artifact to {artifact_path}")

        elif workflow == "multi_agent_triage":
            subagent_model = exp.get("subagent_model", "ollama/qwen2.5-coder:7b")
            mock_promql_file = exp.get("mock_promql_file", "examples/inputs/mock_promql_result.json")
            mock_logql_file = exp.get("mock_logql_file", "examples/inputs/mock_logql_result.json")
            actual_output, latency, usage = await asyncio.to_thread(
                multi_agent_triage_task,
                model_name, subagent_model, exp["system_prompt"], input_prompt,
                mock_promql_file, mock_logql_file,
            )

        else:
            num_ctx = exp.get("num_ctx", 4096)
            actual_output, latency, usage = await asyncio.to_thread(
                agent_task, model_name, exp["system_prompt"], input_prompt, num_ctx
            )

        test_case_input = input_prompt if isinstance(input_prompt, str) else case.get("task_prompt", "")
        test_case = LLMTestCase(
            input=test_case_input,
            actual_output=actual_output,
            expected_output=expected_output,
        )

        print(f"   Grading Output...")
        if allow_code_execution:
            exec_metric = ExecutionMetric()
            exec_score = await asyncio.to_thread(exec_metric.measure, test_case)
            exec_reason = exec_metric.reason or ""
        else:
            exec_score = None
            exec_reason = "Skipped (pass --allow-code-execution to enable)"

        geval = GEval(
            name="Code Requirements Checklist",
            criteria=expected_output,
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            model=CustomLiteLLM(judge_model),
        )
        geval_score = await geval.a_measure(test_case)

        run = {
            "pipeline": combo,
            "case_name": case["name"],
            "latency_sec": round(latency, 3),
            "tokens": usage,
            "actual_output": actual_output,
            "scores": {
                "ExecutionMetric": exec_score,
                "ExecutionReason": exec_reason,
                "GEval": geval_score,
                "GEvalReason": getattr(geval, "reason", ""),
            },
        }
        runs.append(run)
        print(f"   [DONE] Latency: {latency:.2f}s | GEval: {geval_score} | Exec: {exec_score if exec_score is not None else 'skipped'}")

        push_metrics_to_prometheus(
            experiment_name, combo_id, case["name"],
            {"ExecutionMetric": exec_score, "GEval": geval_score},
            latency,
        )

    return runs


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    config_path: str = typer.Argument(..., help="Path to experiment YAML spec"),
    allow_code_execution: bool = typer.Option(
        False,
        "--allow-code-execution",
        help="Allow ExecutionMetric to run LLM-generated code. Only use with trusted/local models.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate the YAML and print what would run without calling any LLM.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Skip cases already present in the most recent results file for this experiment.",
    ),
    parallel: bool = typer.Option(
        False,
        "--parallel",
        help="Evaluate all model combos concurrently instead of sequentially.",
    ),
):
    """Run an agent experiment defined in a YAML spec."""
    asyncio.run(_run_eval(config_path, allow_code_execution, dry_run, resume, parallel))


async def _run_eval(
    config_path: str,
    allow_code_execution: bool,
    dry_run: bool,
    resume: bool,
    parallel: bool,
) -> None:
    setup_tracing()

    with open(config_path, "r") as f:
        exp = yaml.safe_load(f)

    experiment_name = exp.get("name", exp.get("experiment_name", "Unnamed Experiment"))
    print(f"[*] Starting Experiment: {experiment_name}")

    judge_model = exp.get("judge_model", "judge-model")
    models_to_test = exp.get("orchestrator_models", []) or exp.get("models_to_test", [])
    workflow = exp.get("workflow", "single_agent")

    if "pipeline_combinations" in exp:
        combinations = exp["pipeline_combinations"]
    elif "mob_combinations" in exp:
        combinations = exp["mob_combinations"]
    else:
        combinations = [{"model": m} for m in models_to_test]

    if dry_run:
        _dry_run_print(exp, experiment_name, workflow, judge_model, combinations)
        return

    completed_keys: set[tuple[str, str]] = set()
    existing_results: dict | None = None
    resume_path: str | None = None

    if resume:
        existing_results, completed_keys, resume_path = _load_resume(experiment_name)

    results: dict = existing_results or {"experiment_name": experiment_name, "runs": []}

    try:
        if parallel:
            tasks = [
                _eval_combo(combo, exp, judge_model, workflow, completed_keys, allow_code_execution)
                for combo in combinations
            ]
            all_runs = await asyncio.gather(*tasks, return_exceptions=True)
            for r in all_runs:
                if isinstance(r, Exception):
                    print(f"[!] Combo evaluation failed: {r}")
                else:
                    results["runs"].extend(r)
        else:
            for combo in combinations:
                runs = await _eval_combo(combo, exp, judge_model, workflow, completed_keys, allow_code_execution)
                results["runs"].extend(runs)
    finally:
        save_experiment_results(experiment_name, results, output_path=resume_path)


@app.command()
def compare(
    result_a: str = typer.Argument(..., help="Path to first results JSON"),
    result_b: str = typer.Argument(..., help="Path to second results JSON"),
):
    """Compare two experiment result files side by side."""
    with open(result_a) as f:
        data_a = json.load(f)
    with open(result_b) as f:
        data_b = json.load(f)

    def build_index(data: dict) -> dict:
        idx = {}
        for run in data.get("runs", []):
            combo_id = _make_combo_id(run["pipeline"])
            idx[(combo_id, run["case_name"])] = run
        return idx

    idx_a = build_index(data_a)
    idx_b = build_index(data_b)
    all_keys = sorted(set(idx_a.keys()) | set(idx_b.keys()))

    print(f"\nComparing:")
    print(f"  A: {data_a['experiment_name']} ({result_a})")
    print(f"  B: {data_b['experiment_name']} ({result_b})")
    print()

    col_combo = 38
    col_case = 28
    header = (
        f"{'Combo':<{col_combo}}  {'Case':<{col_case}}"
        f"  {'GEval A':>7}  {'GEval B':>7}  {'Δ GEval':>7}"
        f"  {'Lat A':>7}  {'Lat B':>7}  {'Δ Lat':>7}"
    )
    print(header)
    print("-" * len(header))

    for combo_id, case_name in all_keys:
        run_a = idx_a.get((combo_id, case_name))
        run_b = idx_b.get((combo_id, case_name))

        geval_a = run_a["scores"]["GEval"] if run_a else None
        geval_b = run_b["scores"]["GEval"] if run_b else None
        lat_a = run_a["latency_sec"] if run_a else None
        lat_b = run_b["latency_sec"] if run_b else None

        sa = f"{geval_a:.3f}" if geval_a is not None else "N/A"
        sb = f"{geval_b:.3f}" if geval_b is not None else "N/A"
        dg = f"{geval_b - geval_a:+.3f}" if geval_a is not None and geval_b is not None else "N/A"
        la = f"{lat_a:.1f}s" if lat_a is not None else "N/A"
        lb = f"{lat_b:.1f}s" if lat_b is not None else "N/A"
        dl = f"{lat_b - lat_a:+.1f}s" if lat_a is not None and lat_b is not None else "N/A"

        c = combo_id[:col_combo - 2] + ".." if len(combo_id) > col_combo else combo_id
        n = case_name[:col_case - 2] + ".." if len(case_name) > col_case else case_name

        print(f"{c:<{col_combo}}  {n:<{col_case}}  {sa:>7}  {sb:>7}  {dg:>7}  {la:>7}  {lb:>7}  {dl:>7}")


if __name__ == "__main__":
    app()

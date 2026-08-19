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
      anthropic/<model>   — Anthropic's own API, via ANTHROPIC_API_KEY

    The anthropic/ prefix exists because some things worth evaluating run on a hosted
    model in production and nowhere else — a routing agent, for instance. Scoring a
    local stand-in would measure a model that never sees the real traffic.
    """
    api_base = os.getenv("LITELLM_API_BASE", "http://localhost:4000/v1")
    is_direct = False

    if model_name.startswith("claude-cli/"):
        return model_name[len("claude-cli/") :], None, False

    if model_name.startswith("anthropic/"):
        # litellm resolves the endpoint and key itself for a first-class provider;
        # returning api_base=None keeps _chat from overriding it with the proxy.
        return model_name, None, False

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

class ToolCallMetric(BaseMetric):
    """Deterministic scoring of emitted tool calls against expected ones.

    Unlike GEval this asks no judge model — either the right tool came back with the right
    arguments or it did not. It also names *how* it failed, because the failure modes need
    different fixes:

      ok             — expected call(s) emitted and parsed
      no_call        — model answered in prose and never tried to call a tool
      unparsed_call  — model emitted a tool call in its content that the server never parsed
                       into structured tool_calls (template/parser mismatch). The model chose
                       correctly; the plumbing dropped it. An agent sees nothing either way.
      wrong_tool     — a tool was called, but not the expected one
      bad_arguments  — right tool, wrong or missing arguments
      unwanted_call  — a case marked `expect_no_tool_call` got one anyway (over-eager tool use)

    Expected calls come from a test case's `expected_tool_calls`:

        expected_tool_calls:
          - name: get_prometheus_metric
            arguments:            # exact match, string compare is case-insensitive
              query: "up"
            arguments_contains:   # substring match, for values that vary (queries, paths)
              query: "api-server-1"
    """

    # A bare {"name": ..., "arguments": ...} object or a <tool_call> wrapper left in the text
    _UNPARSED_PATTERNS = (
        re.compile(r"<tool_call>", re.IGNORECASE),
        re.compile(r'\{\s*"name"\s*:\s*".+?"\s*,\s*"arguments"\s*:', re.DOTALL),
        re.compile(r'\{\s*"function"\s*:\s*\{', re.DOTALL),
    )

    def __init__(
        self,
        expected_calls: list[dict],
        tool_calls: list[dict],
        raw_content: str = "",
        threshold: float = 1.0,
        expect_no_call: bool = False,
    ):
        self.expected_calls = expected_calls or []
        self.tool_calls = tool_calls or []
        self.raw_content = raw_content or ""
        self.threshold = threshold
        self.expect_no_call = expect_no_call
        self.score = 0.0
        self.reason = None
        self.failure_mode = None
        self.success = False

    def _looks_like_unparsed_call(self) -> bool:
        return any(p.search(self.raw_content) for p in self._UNPARSED_PATTERNS)

    @staticmethod
    def _args_match(expected: dict, contains: dict, actual: dict) -> tuple[bool, str]:
        for key, want in (expected or {}).items():
            if key not in actual:
                return False, f"missing argument '{key}'"
            got = actual[key]
            if isinstance(want, str) and isinstance(got, str):
                if want.strip().lower() != got.strip().lower():
                    return False, f"argument '{key}' was {got!r}, expected {want!r}"
            elif want != got:
                return False, f"argument '{key}' was {got!r}, expected {want!r}"

        for key, want in (contains or {}).items():
            if key not in actual:
                return False, f"missing argument '{key}'"
            if str(want).lower() not in str(actual[key]).lower():
                return False, f"argument '{key}' ({actual[key]!r}) does not contain {want!r}"

        return True, ""

    def measure(self, test_case: LLMTestCase):
        if self.expect_no_call:
            if self.tool_calls:
                self.score = 0.0
                self.failure_mode = "unwanted_call"
                self.reason = f"Expected no tool call; model called {[c.get('name') for c in self.tool_calls]}."
            else:
                self.score = 1.0
                self.failure_mode = "ok"
                self.reason = "No tool call emitted, as expected."
            self.success = self.score >= self.threshold
            return self.score

        if not self.expected_calls:
            self.score = 0.0
            self.failure_mode = "no_expectation"
            self.reason = "No expected_tool_calls defined for this case."
            self.success = False
            return self.score

        if not self.tool_calls:
            if self._looks_like_unparsed_call():
                self.failure_mode = "unparsed_call"
                self.reason = (
                    "Model emitted a tool call in its content but the server did not parse it into "
                    "tool_calls (chat template / parser mismatch). An agent sees no tool call. "
                    f"Content began: {self.raw_content[:160]!r}"
                )
            else:
                self.failure_mode = "no_call"
                self.reason = f"No tool call emitted. Content began: {self.raw_content[:160]!r}"
            self.score = 0.0
            self.success = False
            return self.score

        called = {c.get("name"): c.get("arguments", {}) for c in self.tool_calls}
        per_call: list[float] = []
        reasons: list[str] = []
        modes: set[str] = set()

        for expected in self.expected_calls:
            name = expected.get("name")
            if name not in called:
                per_call.append(0.0)
                modes.add("wrong_tool")
                reasons.append(f"expected '{name}', got {sorted(called) or 'nothing'}")
                continue

            ok, why = self._args_match(expected.get("arguments"), expected.get("arguments_contains"), called[name])
            if ok:
                per_call.append(1.0)
                reasons.append(f"'{name}' called correctly")
            else:
                per_call.append(0.5)
                modes.add("bad_arguments")
                reasons.append(f"'{name}' called but {why}")

        self.score = sum(per_call) / len(per_call)
        self.success = self.score >= self.threshold
        self.failure_mode = "ok" if self.success else ("wrong_tool" if "wrong_tool" in modes else "bad_arguments")

        extra = [n for n in called if n not in {e.get("name") for e in self.expected_calls}]
        if extra:
            reasons.append(f"also called unexpected {extra}")
        self.reason = "; ".join(reasons)
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Tool Call Accuracy"


class EmbeddingRetrievalMetric(BaseMetric):
    """Deterministic scoring for `workflow: embedding_quality`: does the correct passage's
    embedding rank closest to the query's, among a set of distractors?

    No judge model — cosine similarity is exact arithmetic, not something worth asking an LLM to
    grade. Score is `1 / rank` (1.0 if the correct passage comes out on top, 0.5 if second, 0.33
    if third...) rather than pass/fail, since "barely lost first place" and "buried under four
    distractors" are different quality signals worth keeping apart in the results.
    """

    def __init__(
        self,
        query_vector: list[float],
        candidate_vectors: list[list[float]],
        correct_index: int = 0,
        threshold: float = 1.0,
    ):
        self.query_vector = query_vector
        self.candidate_vectors = candidate_vectors
        self.correct_index = correct_index
        self.threshold = threshold
        self.score = 0.0
        self.reason = None
        self.success = False

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def measure(self, test_case=None):
        sims = [self._cosine(self.query_vector, v) for v in self.candidate_vectors]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        rank = ranked.index(self.correct_index) + 1  # 1-indexed
        self.score = round(1.0 / rank, 3)
        self.success = rank == 1
        self.reason = (
            f"Correct passage ranked #{rank} of {len(sims)} "
            f"(similarity {sims[self.correct_index]:.3f}, top similarity {max(sims):.3f})"
        )
        return self.score

    async def a_measure(self, test_case=None):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Embedding Retrieval Rank"


# The `claude` CLI is driven as a subprocess rather than through the API, following
# harness-bench's claude-code adapter: it authenticates against a Claude subscription,
# so measuring a model that only runs behind that subscription costs quota rather than
# metered API credits.
#
# FIDELITY TRADEOFF, stated plainly: the CLI accepts no arbitrary tool schemas -- its
# tools are its own, and extra ones arrive only over MCP. So the schemas are rendered
# into the prompt and the model is asked to answer with the call it would make. That
# measures the *decision* against the real model, which is the point; it does not
# measure whether the model reliably emits a well-formed tool call through a real
# function-calling API. A `no_call` here means it did not answer in the requested shape,
# which is a weaker claim than it would be over the API.
_CLAUDE_CLI_CONTRACT = """

You have access to the following tools:

{tools}

Answer with a single JSON object naming the ONE tool call you would make, and nothing
else -- no prose, no markdown fence:

{{"name": "<tool name>", "arguments": {{...}}}}

If you would not call any tool, answer exactly: {{"name": null, "arguments": {{}}}}
"""


def _chat_claude_cli(model: str, system_prompt: str, input_prompt: str, tools: list | None):
    """One turn through the `claude` CLI. Returns (output, usage, tool_calls)."""
    rendered = ""
    if tools:
        rendered = json.dumps(
            [t.get("function", t) for t in tools], indent=2
        )
    prompt = f"{system_prompt}\n\n{input_prompt}"
    if tools:
        prompt += _CLAUDE_CLI_CONTRACT.format(tools=rendered)

    argv = ["claude", "-p", prompt, "--output-format", "json"]
    if model and model not in ("default", "native"):
        argv += ["--model", model]
    # Its own tools are irrelevant here and would let it wander off exploring the
    # filesystem instead of answering; a routing decision needs no filesystem at all.
    argv += ["--disallowedTools", "Bash", "Read", "Write", "Edit", "Glob", "Grep", "Task", "WebFetch", "WebSearch"]
    # No MCP servers should load for a single-turn scored question, and loading none is
    # both correct and marginally cheaper. Measured per-call context: 26.5k with no flags,
    # 22.2k with the tool restrictions above. The remaining ~22k is Claude Code's own
    # system prompt and cannot be reduced through this interface -- it is the fixed price
    # of driving the CLI, paid fresh on every sample because every sample is a new process.
    argv += ["--strict-mcp-config"]

    proc = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI exited {proc.returncode}: {proc.stderr[:400]}")

    envelope = json.loads(proc.stdout)
    output = envelope.get("result") or ""
    u = envelope.get("usage") or {}

    # Cache tokens are the real cost here and must be counted. A `claude -p` call reports
    # input_tokens: 2 while actually consuming ~26.5k -- the session's system prompt, tool
    # schemas and project context arrive as cache_creation + cache_read, and every
    # invocation is a fresh process that pays that setup again before it sees the question.
    # Recording input_tokens alone understated this eval's own consumption by four orders
    # of magnitude, which is a bad failure for an instrument whose whole purpose is telling
    # you which model is worth its cost.
    cache_creation = u.get("cache_creation_input_tokens", 0)
    cache_read = u.get("cache_read_input_tokens", 0)
    prompt_tokens = u.get("input_tokens", 0) + cache_creation + cache_read
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": u.get("output_tokens", 0),
        "total_tokens": prompt_tokens + u.get("output_tokens", 0),
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
        "uncached_input_tokens": u.get("input_tokens", 0),
        "cost_usd": envelope.get("total_cost_usd"),
    }

    # Parse the declared call back into the same shape a real tool_calls array has, so
    # ToolCallMetric scores this identically to every other transport.
    tool_calls = []
    m = re.search(r"\{.*\}", output, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and obj.get("name"):
                tool_calls = [{"name": obj["name"], "arguments": obj.get("arguments") or {}}]
        except json.JSONDecodeError:
            pass
    return output, usage, tool_calls


def _chat(model_name: str, system_prompt: str, input_prompt: str, num_ctx: int = 4096, tools: list | None = None):
    """One chat completion via litellm routing or a direct ollama hit.

    Returns (output, latency, usage, tool_calls). `tool_calls` is a list of
    {"name": str, "arguments": dict} normalized across both transports, or [] if the
    model emitted none. Passing `tools` sends the schemas to the model; note that a
    model can emit a tool call as plain text without it being parsed into this list —
    that distinction is what ToolCallMetric scores.
    """
    start_time = time.time()
    proxy_model, api_base, is_direct = resolve_endpoint(model_name)

    if model_name.startswith("claude-cli/"):
        output, usage, tool_calls = _chat_claude_cli(proxy_model, system_prompt, input_prompt, tools)
        return output, time.time() - start_time, usage, tool_calls

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
        if tools:
            payload["tools"] = tools
        res = requests.post(api_base, json=payload, timeout=1200)
        res.raise_for_status()
        data = res.json()
        latency = time.time() - start_time
        message = data.get("message", {})
        output = message.get("content", "")
        raw_tool_calls = message.get("tool_calls") or []
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        }
    else:
        kwargs = {}
        if tools:
            kwargs["tools"] = tools
        if api_base is None:
            # A first-class litellm provider (anthropic/...): let it resolve its own
            # endpoint and credentials. num_ctx is an Ollama option and is not sent.
            response = litellm.completion(
                model=proxy_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_prompt},
                ],
                timeout=1200,
                **kwargs,
            )
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
                timeout=1200,
                **kwargs,
            )
        latency = time.time() - start_time
        message = response.choices[0].message
        output = message.content or ""
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        usage = response.usage.model_dump() if response.usage else {}

    # Strip <think> tags from reasoning model output before scoring
    output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()
    return output, latency, usage, _normalize_tool_calls(raw_tool_calls)


def _normalize_tool_calls(raw_tool_calls: list) -> list[dict]:
    """Flatten Ollama-native and OpenAI-compat tool call shapes into {"name", "arguments"}.

    Native /api/chat returns arguments as a dict; the OpenAI-compat /v1 path returns them
    as a JSON string. Both are normalized to a dict so scoring does not care which
    transport produced them.
    """
    normalized = []
    for call in raw_tool_calls:
        if isinstance(call, dict):
            fn = call.get("function", {})
        else:  # litellm/openai object
            fn = getattr(call, "function", None)
            fn = {"name": getattr(fn, "name", None), "arguments": getattr(fn, "arguments", None)} if fn else {}

        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {"_unparsed": args}
        normalized.append({"name": fn.get("name"), "arguments": args or {}})
    return normalized


def agent_task(model_name: str, system_prompt: str, input_prompt: str, num_ctx: int = 4096):
    """Executes the agent task using litellm routing or direct ollama hit."""
    output, latency, usage, _ = _chat(model_name, system_prompt, input_prompt, num_ctx)
    return output, latency, usage


def _load_tools(exp: dict) -> list:
    """Resolve an experiment's `tools:` field into a list of OpenAI-format tool schemas.

    Accepts either the schemas inline, or a path to a .json/.yaml file holding them (a bare
    list, or an object with a top-level `tools` key).
    """
    tools = exp.get("tools")
    if not tools:
        raise ValueError("workflow 'tool_calling' requires a 'tools' field (inline schemas or a file path)")

    if isinstance(tools, str):
        with open(tools, "r") as f:
            loaded = json.load(f) if tools.lower().endswith(".json") else yaml.safe_load(f)
        tools = loaded.get("tools") if isinstance(loaded, dict) else loaded

    if not isinstance(tools, list):
        raise ValueError("'tools' must resolve to a list of tool schemas")
    return tools


def tool_calling_task(
    model_name: str,
    system_prompt: str,
    input_prompt: str,
    tools: list,
    num_ctx: int = 4096,
):
    """Single-turn tool-calling task: offer the model real tool schemas and see what it emits.

    Returns (rendered_output, latency, usage, tool_calls). The rendered output keeps the
    model's prose plus a readable dump of any parsed calls so GEval can still score it if
    the experiment supplies `expected_output_criteria`.
    """
    output, latency, usage, tool_calls = _chat(model_name, system_prompt, input_prompt, num_ctx, tools=tools)

    if tool_calls:
        rendered = "Tool calls:\n" + "\n".join(
            f"- {c['name']}({json.dumps(c['arguments'], sort_keys=True)})" for c in tool_calls
        )
        if output:
            rendered += f"\n\nContent:\n{output}"
    else:
        rendered = output

    return rendered, latency, usage, tool_calls


def _embed(model_name: str, texts: list[str]) -> tuple[list[list[float]], float, dict]:
    """Embed a batch of texts, using the same prefix routing _chat() uses.

    Native path (`direct_ws/`, `direct_sre/`) calls Ollama's batch `/api/embed`. Non-direct path
    (`ws/`, `sre/`, `ollama/`) calls litellm.embedding() against the same `/v1` base _chat() uses
    for those prefixes, so `resolve_endpoint()` needs no embedding-specific branch.
    """
    start_time = time.time()
    proxy_model, api_base, is_direct = resolve_endpoint(model_name)

    if is_direct:
        embed_base = api_base.replace("/api/chat", "/api/embed")
        res = requests.post(embed_base, json={"model": proxy_model, "input": texts}, timeout=1200)
        res.raise_for_status()
        data = res.json()
        vectors = data.get("embeddings", [])
        usage = {"prompt_tokens": data.get("prompt_eval_count", 0)}
    else:
        response = litellm.embedding(
            model=proxy_model, input=texts, api_base=api_base, api_key="sk-dummy", timeout=1200,
        )
        vectors = [d["embedding"] for d in response.data]
        usage = response.usage.model_dump() if response.usage else {}

    latency = time.time() - start_time
    return vectors, latency, usage


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
    if workflow == "tool_calling":
        try:
            tools = _load_tools(exp)
            print(f"[DRY-RUN] Tools ({len(tools)}): {[t.get('function', {}).get('name') for t in tools]}")
        except (ValueError, OSError, yaml.YAMLError, json.JSONDecodeError) as e:
            print(f"[DRY-RUN] Tools: INVALID — {e}")

    test_cases = exp.get("test_cases", [])
    print(f"[DRY-RUN] Test cases ({len(test_cases)}):")
    judged = 0
    for tc in test_cases:
        if workflow == "embedding_quality":
            missing = [f for f in ("query", "correct") if f not in tc]
            status = "OK" if not missing else f"MISSING {missing}"
            n_distractors = len(tc.get("distractors", []))
            print(f"  [{status}] {tc['name']}: query vs. 1 correct + {n_distractors} distractor(s)")
            continue
        path = tc.get("input_file")
        if path:
            status = "OK" if os.path.exists(path) else "MISSING"
        else:
            status = "NO INPUT FILE"
            path = "(prompt only)"
        expects = []
        if tc.get("expect_no_tool_call"):
            expects.append("no tool call")
        elif tc.get("expected_tool_calls"):
            expects.append(f"{len(tc['expected_tool_calls'])} tool call(s)")
        if tc.get("expected_output_criteria"):
            expects.append("GEval criteria")
            judged += 1
        print(f"  [{status}] {tc['name']}: {path}" + (f" -> {', '.join(expects)}" if expects else ""))

    repeats = max(1, int(exp.get("repeats", 1)))
    total_agent = len(combinations) * len(test_cases) * repeats
    total_judge = len(combinations) * judged * repeats
    repeats_note = f" (repeats={repeats})" if repeats > 1 else ""
    print(f"[DRY-RUN] Estimated LLM calls{repeats_note}: {total_agent} agent + {total_judge} judge = {total_agent + total_judge} total")


# ---------------------------------------------------------------------------
# Core evaluation coroutine (one combo / pipeline)
# ---------------------------------------------------------------------------

async def _run_and_score_sample(
    combo: dict,
    combo_id: str,
    exp: dict,
    judge_model: str,
    workflow: str,
    case: dict,
    input_prompt,
    expected_output: str,
    allow_code_execution: bool,
    sample_suffix: str = "",
) -> dict:
    """Call the model once for this case and score the result. One "sample" — `repeats: N`
    in the YAML calls this N times per case and `_aggregate_samples` combines them.

    `sample_suffix` (e.g. "_s2") disambiguates multi-agent artifact filenames across repeats;
    it's empty for a single-sample run so filenames are unchanged from before `repeats` existed.
    """
    model_name = combo.get("model", combo.get("orchestrator", combo.get("generator", "pipeline")))
    tool_calls: list[dict] = []

    if workflow == "embedding_quality":
        # Shaped entirely differently from the chat workflows below — a query ranked against
        # candidate passages, not a single input/output pair — so it returns early rather than
        # flowing into the shared ExecutionMetric/ToolCallMetric/GEval scoring tail, none of
        # which apply to an embedding model.
        candidates = [case["correct"]] + list(case.get("distractors", []))
        vectors, latency, usage = await asyncio.to_thread(_embed, model_name, [case["query"]] + candidates)
        metric = EmbeddingRetrievalMetric(vectors[0], vectors[1:], correct_index=0)
        score = metric.measure()
        return {
            "latency_sec": round(latency, 3),
            "tokens": usage,
            "actual_output": metric.reason,
            "tool_calls": [],
            "scores": {
                "ExecutionMetric": None, "ExecutionReason": "N/A (embedding_quality workflow)",
                "GEval": None, "GEvalReason": "N/A (embedding_quality workflow)",
                "ToolCallMetric": None, "ToolCallReason": "N/A (embedding_quality workflow)",
                "ToolCallFailureMode": None,
                "EmbeddingRetrievalMetric": score,
                "EmbeddingRetrievalReason": metric.reason,
            },
        }

    if workflow == "multi_agent_blog_gen":
        actual_output, latency, usage, draft, critique = await asyncio.to_thread(
            multi_agent_blog_task,
            combo["generator"], combo["critic"], combo["refiner"],
            exp["system_prompt"], input_prompt,
        )
        artifact_dir = "results/artifacts"
        os.makedirs(artifact_dir, exist_ok=True)
        safe_case = case["name"].replace(" ", "_").replace("/", "-")
        artifact_path = os.path.join(artifact_dir, f"Blog_{combo_id}_{safe_case}{sample_suffix}.md")
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
        artifact_path = os.path.join(artifact_dir, f"Mob_{combo_id}_{safe_case}{sample_suffix}.md")
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

    elif workflow == "tool_calling":
        num_ctx = exp.get("num_ctx", 4096)
        actual_output, latency, usage, tool_calls = await asyncio.to_thread(
            tool_calling_task, model_name, exp["system_prompt"], input_prompt,
            _load_tools(exp), num_ctx,
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

    if allow_code_execution:
        exec_metric = ExecutionMetric()
        exec_score = await asyncio.to_thread(exec_metric.measure, test_case)
        exec_reason = exec_metric.reason or ""
    else:
        exec_score = None
        exec_reason = "Skipped (pass --allow-code-execution to enable)"

    if case.get("expected_tool_calls") or case.get("expect_no_tool_call"):
        tool_metric = ToolCallMetric(
            case.get("expected_tool_calls", []), tool_calls, raw_content=actual_output,
            expect_no_call=bool(case.get("expect_no_tool_call")),
        )
        tool_score = await asyncio.to_thread(tool_metric.measure, test_case)
        tool_reason = tool_metric.reason or ""
        tool_failure_mode = tool_metric.failure_mode
    else:
        tool_score = None
        tool_reason = "Skipped (no expected_tool_calls)"
        tool_failure_mode = None

    # A tool-calling case need not define text criteria; asking a judge to grade against an
    # empty rubric produces a meaningless score.
    if expected_output:
        geval = GEval(
            name="Code Requirements Checklist",
            criteria=expected_output,
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            model=CustomLiteLLM(judge_model),
        )
        geval_score = await geval.a_measure(test_case)
        geval_reason = getattr(geval, "reason", "")
    else:
        geval_score = None
        geval_reason = "Skipped (no expected_output_criteria)"

    return {
        "latency_sec": round(latency, 3),
        "tokens": usage,
        "actual_output": actual_output,
        "tool_calls": tool_calls,
        "scores": {
            "ExecutionMetric": exec_score,
            "ExecutionReason": exec_reason,
            "GEval": geval_score,
            "GEvalReason": geval_reason,
            "ToolCallMetric": tool_score,
            "ToolCallReason": tool_reason,
            "ToolCallFailureMode": tool_failure_mode,
        },
    }


def _aggregate_samples(combo: dict, case_name: str, samples: list[dict]) -> dict:
    """Combine repeated samples of one case into a single run record.

    Numeric scores are averaged. `ToolCallMetric` additionally gets `ToolCallPassRate` — the
    fraction of samples that hit the metric's own success threshold — because a mean score alone
    can't distinguish "always half-right" from "right half the time, wrong half the time"; the
    two need different fixes and the pass rate is what tells them apart.
    """
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    scores: dict = {}
    for key in ("ExecutionMetric", "GEval", "ToolCallMetric", "EmbeddingRetrievalMetric"):
        scores[key] = mean([s["scores"].get(key) for s in samples])

    for score_key, reason_key in (
        ("ExecutionMetric", "ExecutionReason"),
        ("GEval", "GEvalReason"),
        ("ToolCallMetric", "ToolCallReason"),
        ("EmbeddingRetrievalMetric", "EmbeddingRetrievalReason"),
    ):
        reasons = [s["scores"].get(reason_key) for s in samples if s["scores"].get(reason_key)]
        scores[reason_key] = " | ".join(f"[{i + 1}] {r}" for i, r in enumerate(reasons)) if reasons else ""

    failure_modes = [s["scores"]["ToolCallFailureMode"] for s in samples if s["scores"].get("ToolCallFailureMode")]
    if failure_modes:
        distinct = set(failure_modes)
        scores["ToolCallFailureMode"] = failure_modes[0] if len(distinct) == 1 else f"mixed({','.join(sorted(distinct))})"
        scores["ToolCallPassRate"] = round(sum(1 for m in failure_modes if m == "ok") / len(samples), 3)
    else:
        scores["ToolCallFailureMode"] = None
        scores["ToolCallPassRate"] = None

    tokens_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for s in samples:
        for k in tokens_total:
            tokens_total[k] += (s.get("tokens") or {}).get(k, 0)

    return {
        "pipeline": combo,
        "case_name": case_name,
        "repeats": len(samples),
        "latency_sec": round(sum(s["latency_sec"] for s in samples) / len(samples), 3),
        "tokens": tokens_total,
        "actual_output": samples[-1]["actual_output"],
        "tool_calls": samples[-1]["tool_calls"],
        "samples": samples,
        "scores": scores,
    }


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
    combo_id = _make_combo_id(combo)
    experiment_name = exp.get("name", exp.get("experiment_name", "Unnamed Experiment"))
    repeats = max(1, int(exp.get("repeats", 1)))
    print(f"\n>> Evaluating Pipeline/Model: {combo_id}" + (f" (repeats={repeats})" if repeats > 1 else ""))

    for case in exp.get("test_cases", []):
        if (combo_id, case["name"]) in completed_keys:
            print(f"   Skipping (already done): {case['name']}")
            continue

        try:
            if not case.get("input_file"):
                # Tool-calling and other prompt-only cases need no input file.
                input_prompt = case.get("task_prompt", "")
            elif case["input_file"].lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
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

        samples = []
        for i in range(repeats):
            if repeats > 1:
                print(f"      Sample {i + 1}/{repeats}...")
            try:
                sample = await _run_and_score_sample(
                    combo, combo_id, exp, judge_model, workflow, case, input_prompt, expected_output,
                    allow_code_execution, sample_suffix=f"_s{i + 1}" if repeats > 1 else "",
                )
            except Exception as e:
                # One candidate's incompatibility (a model that rejects a tools payload, a judge
                # that returns malformed JSON on an off sample) must not cost every model queued
                # after it in models_to_test — record the failure and keep going. A bake-off run
                # is only useful if one bad candidate can't silently truncate the rest of it.
                err = f"{type(e).__name__}: {e}"
                print(f"   [!] Sample failed: {err}")
                sample = {
                    "latency_sec": 0.0,
                    "tokens": {},
                    "actual_output": "",
                    "tool_calls": [],
                    "scores": {
                        "ExecutionMetric": None, "ExecutionReason": f"ERROR: {err}",
                        "GEval": None, "GEvalReason": f"ERROR: {err}",
                        "ToolCallMetric": None, "ToolCallReason": f"ERROR: {err}", "ToolCallFailureMode": "error",
                        "EmbeddingRetrievalMetric": None, "EmbeddingRetrievalReason": f"ERROR: {err}",
                    },
                }
            samples.append(sample)

        if repeats == 1:
            s = samples[0]
            scores = dict(s["scores"])
            # ToolCallPassRate is otherwise only produced by _aggregate_samples, so a
            # single-sample run carried a failure mode but no pass rate -- and --fail-under
            # gates on the pass rate. The gate therefore had nothing to evaluate and printed
            # PASS unconditionally on exactly the runs that are now the default. A rate over
            # one sample is just whether that sample passed.
            if scores.get("ToolCallFailureMode") is not None:
                scores["ToolCallPassRate"] = 1.0 if scores["ToolCallFailureMode"] == "ok" else 0.0
            run = {
                "pipeline": combo,
                "case_name": case["name"],
                "latency_sec": s["latency_sec"],
                "tokens": s["tokens"],
                "actual_output": s["actual_output"],
                "tool_calls": s["tool_calls"],
                "scores": scores,
            }
        else:
            run = _aggregate_samples(combo, case["name"], samples)

        runs.append(run)
        scores = run["scores"]
        summary = (
            f"   [DONE] Latency: {run['latency_sec']:.2f}s | GEval: {scores['GEval']} | "
            f"Exec: {scores['ExecutionMetric'] if scores['ExecutionMetric'] is not None else 'skipped'}"
        )
        if scores["ToolCallMetric"] is not None:
            summary += f" | ToolCall: {scores['ToolCallMetric']:.2f} ({scores['ToolCallFailureMode']})"
            if repeats > 1:
                summary += f" | PassRate: {scores['ToolCallPassRate']:.2f}"
        if scores.get("EmbeddingRetrievalMetric") is not None:
            summary += f" | EmbeddingRetrieval: {scores['EmbeddingRetrievalMetric']:.2f}"
        print(summary)

        push_metrics_to_prometheus(
            experiment_name, combo_id, case["name"],
            {
                "ExecutionMetric": scores["ExecutionMetric"],
                "GEval": scores["GEval"],
                "ToolCallMetric": scores["ToolCallMetric"],
                "EmbeddingRetrievalMetric": scores.get("EmbeddingRetrievalMetric"),
            },
            run["latency_sec"],
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
    fail_under: float = typer.Option(
        None,
        "--fail-under",
        help="Exit non-zero if any pipeline's mean ToolCallPassRate falls below this. "
             "Turns a run into a regression gate rather than something to eyeball.",
    ),
):
    """Run an agent experiment defined in a YAML spec."""
    asyncio.run(_run_eval(config_path, allow_code_execution, dry_run, resume, parallel, fail_under))


async def _run_eval(
    config_path: str,
    allow_code_execution: bool,
    dry_run: bool,
    resume: bool,
    parallel: bool,
    fail_under: float | None = None,
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

    _print_headline(results, fail_under)


def _headline_scores(results: dict) -> dict[str, dict]:
    """Per-pipeline headline: mean ToolCallMetric and mean ToolCallPassRate.

    Per-case scores answer "which case failed"; a regression gate needs one number per
    pipeline it can compare against a committed baseline. Pass rate is the honest one to
    gate on -- a mean score can sit comfortably above a threshold while half the runs
    route to the wrong place.
    """
    by_pipeline: dict[str, list[dict]] = {}
    for run in results.get("runs", []):
        by_pipeline.setdefault(str(run.get("pipeline", "?")), []).append(run)

    out: dict[str, dict] = {}
    for pipeline, runs in by_pipeline.items():
        def mean(key):
            vals = [r.get("scores", {}).get(key) for r in runs]
            vals = [v for v in vals if v is not None]
            return round(sum(vals) / len(vals), 3) if vals else None

        # Failure modes are counted, not just averaged, because a single pass rate
        # conflates two failures that need different fixes. A `no_call` means the model
        # answered in prose and the plumbing saw nothing -- a reliability problem with
        # that model. A `wrong_tool`/`bad_arguments` means it acted, incorrectly -- a
        # correctness problem with the prompt or the task. An experiment can sit at 0.7
        # entirely from the former while every decision it actually made was right.
        modes: dict[str, int] = {}
        for r in runs:
            raw = r.get("scores", {}).get("ToolCallFailureMode") or ""
            for part in raw.replace("mixed(", "").replace(")", "").split(","):
                part = part.strip()
                if part:
                    modes[part] = modes.get(part, 0) + 1

        out[pipeline] = {
            "cases": len(runs),
            "ToolCallMetric": mean("ToolCallMetric"),
            "ToolCallPassRate": mean("ToolCallPassRate"),
            "failure_modes": modes,
            "incorrect_actions": sum(
                modes.get(m, 0) for m in ("wrong_tool", "bad_arguments", "unwanted_call")
            ),
        }
    return out


def _print_headline(results: dict, fail_under: float | None) -> None:
    headline = _headline_scores(results)
    if not headline:
        return

    print("\n[*] Headline (mean across cases)")
    for pipeline, s in sorted(headline.items()):
        score = "n/a" if s["ToolCallMetric"] is None else f"{s['ToolCallMetric']:.3f}"
        rate = "n/a" if s["ToolCallPassRate"] is None else f"{s['ToolCallPassRate']:.3f}"
        print(f"    {pipeline:52} score={score}  pass_rate={rate}  n={s['cases']}")
        if s["failure_modes"]:
            detail = "  ".join(f"{k}={v}" for k, v in sorted(s["failure_modes"].items()))
            print(f"      modes: {detail}")
            print(f"      incorrect actions (wrong_tool/bad_arguments/unwanted_call): "
                  f"{s['incorrect_actions']}")

    if fail_under is None:
        return

    # A gate with nothing to measure must not report success. Passing --fail-under against
    # an experiment that produces no pass rate at all means the threshold is unenforceable,
    # and silently printing PASS is the worst available answer.
    measurable = {p: s for p, s in headline.items() if s["ToolCallPassRate"] is not None}
    if not measurable:
        print(f"\n[!] FAIL: --fail-under={fail_under} was requested but no pipeline produced a "
              "ToolCallPassRate, so the threshold could not be evaluated.")
        raise typer.Exit(code=1)

    failed = {p: s["ToolCallPassRate"] for p, s in measurable.items() if s["ToolCallPassRate"] < fail_under}
    if failed:
        print(f"\n[!] FAIL: pass rate below --fail-under={fail_under}")
        for p, rate in sorted(failed.items()):
            print(f"    {p}: {rate:.3f}")
        raise typer.Exit(code=1)
    print(f"\n[+] PASS: every pipeline at or above --fail-under={fail_under}")


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

    # Pick whichever metric these results actually scored — ToolCallMetric and
    # EmbeddingRetrievalMetric are each specific to one workflow; GEval is the fallback.
    def has_scores(data: dict, key: str) -> bool:
        return any(run.get("scores", {}).get(key) is not None for run in data.get("runs", []))

    if has_scores(data_a, "ToolCallMetric") or has_scores(data_b, "ToolCallMetric"):
        metric_key, metric_label = "ToolCallMetric", "Tool"
    elif has_scores(data_a, "EmbeddingRetrievalMetric") or has_scores(data_b, "EmbeddingRetrievalMetric"):
        metric_key, metric_label = "EmbeddingRetrievalMetric", "Embed"
    else:
        metric_key, metric_label = "GEval", "GEval"

    col_combo = 38
    col_case = 28
    header = (
        f"{'Combo':<{col_combo}}  {'Case':<{col_case}}"
        f"  {metric_label + ' A':>7}  {metric_label + ' B':>7}  {'Δ':>7}"
        f"  {'Lat A':>7}  {'Lat B':>7}  {'Δ Lat':>7}"
    )
    print(header)
    print("-" * len(header))

    for combo_id, case_name in all_keys:
        run_a = idx_a.get((combo_id, case_name))
        run_b = idx_b.get((combo_id, case_name))

        geval_a = run_a["scores"].get(metric_key) if run_a else None
        geval_b = run_b["scores"].get(metric_key) if run_b else None
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

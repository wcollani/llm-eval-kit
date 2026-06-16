import os
import json
from datetime import datetime
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

def setup_tracing(service_name="agent-testing-cli", endpoint=None):
    """
    Configure standard OpenTelemetry tracing to push to Grafana Alloy or Arize Phoenix
    via OTLP over HTTP.
    """
    if endpoint is None:
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4319/v1/traces")
        
    resource = Resource(attributes={"service.name": service_name})
    provider = TracerProvider(resource=resource)
    
    # Configure OTLP Exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=endpoint
    )
    span_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(span_processor)
    
    trace.set_tracer_provider(provider)
    print(f"[*] OpenTelemetry configured to export to {endpoint}")

def save_experiment_results(experiment_name, results):
    """
    Dumps the results into a standardized JSON file.
    """
    os.makedirs("results", exist_ok=True)
    # Sanitize experiment name for file path
    safe_name = "".join([c if c.isalnum() else "_" for c in experiment_name])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{safe_name}_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[*] Results saved to {filename}")

def push_metrics_to_prometheus(experiment_name, model_name, case_name, scores, latency_sec, gateway_url=None):
    """
    Pushes per-case score and latency gauges to a Prometheus Pushgateway.

    `scores` is a dict of metric_name -> value (e.g. {"ExecutionMetric": 1.0, "GEval": 0.85}).
    Never raises — a missing or unreachable gateway must not fail an eval run.
    """
    if gateway_url is None:
        gateway_url = os.getenv("PROMETHEUS_PUSHGATEWAY_URL")
    if not gateway_url:
        return

    try:
        registry = CollectorRegistry()

        score_gauge = Gauge(
            "llm_eval_score", "LLM eval score per case",
            ["model", "experiment", "case", "metric"], registry=registry
        )
        for metric_name, value in scores.items():
            if value is None:
                continue
            score_gauge.labels(model=model_name, experiment=experiment_name, case=case_name, metric=metric_name).set(value)

        latency_gauge = Gauge(
            "llm_eval_latency_ms", "LLM eval latency per case in milliseconds",
            ["model", "experiment", "case"], registry=registry
        )
        latency_gauge.labels(model=model_name, experiment=experiment_name, case=case_name).set(latency_sec * 1000)

        push_to_gateway(
            gateway_url, job="llm_eval", registry=registry,
            grouping_key={"experiment": experiment_name, "model": model_name, "case": case_name},
        )
    except Exception as e:
        print(f"[!] Failed to push metrics to Prometheus Pushgateway: {e}")

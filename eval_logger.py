import os
import json
from datetime import datetime
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

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

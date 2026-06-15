# Prometheus Metrics Schema

## Core Application Metrics

- `http_requests_total`
  - **Type**: Counter
  - **Description**: Total number of HTTP requests.
  - **Labels**: `method` (GET, POST, etc.), `status` (200, 404, 500, etc.), `path` (/api/v1/users, /healthz, etc.), `container` (api-server, grafana, frontend)

- `http_request_duration_seconds`
  - **Type**: Histogram
  - **Description**: Duration of HTTP requests in seconds.
  - **Labels**: `method`, `status`, `path`, `container`
  - **Buckets**: 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10

- `process_resident_memory_bytes`
  - **Type**: Gauge
  - **Description**: Resident memory size in bytes.
  - **Labels**: `container`, `namespace`

- `process_cpu_seconds_total`
  - **Type**: Counter
  - **Description**: Total user and system CPU time spent in seconds.
  - **Labels**: `container`, `namespace`

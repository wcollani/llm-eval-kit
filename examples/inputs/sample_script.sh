#!/usr/bin/env bash
# Sample deployment health check script.
# Used as input for the code-refactor experiment.
set -euo pipefail

SERVICES=("api-server" "worker" "scheduler")
BASE_URL="${BASE_URL:-http://localhost:8080}"

check_service() {
    local name="$1"
    local url="$BASE_URL/health/$name"
    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$url")
    if [ "$status" -eq 200 ]; then
        echo "[OK]   $name ($url)"
    else
        echo "[FAIL] $name ($url) returned HTTP $status"
        return 1
    fi
}

echo "=== Health Check: $(date -u) ==="
all_ok=true
for svc in "${SERVICES[@]}"; do
    check_service "$svc" || all_ok=false
done

if $all_ok; then
    echo "All services healthy."
    exit 0
else
    echo "One or more services failed health check."
    exit 1
fi

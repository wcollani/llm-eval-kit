import os

import requests


def push_metrics(payload: dict):
    """Looks like the hardcoded-secret case, but the token comes from the environment and
    the module refuses to load without it rather than falling back to a default."""
    token = os.environ["METRICS_API_TOKEN"]
    return requests.post(
        "https://metrics.example.com/v1/ingest",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )

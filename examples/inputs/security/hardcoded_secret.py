import requests

API_TOKEN = "sk_test_FAKE_NOT_A_REAL_KEY_0000000000"

def push_metrics(payload: dict):
    return requests.post(
        "https://metrics.example.com/v1/ingest",
        json=payload,
        headers={"Authorization": f"Bearer {API_TOKEN}"},
        timeout=10,
    )

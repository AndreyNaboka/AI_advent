import os

import requests


def charge(amount: float) -> dict:
    if amount <= 0:
        raise ValueError("amount must be positive")
    response = requests.post(
        "https://payments.example/charge",
        json={"amount": amount},
        timeout=5,
        headers={"Authorization": f"Bearer {os.environ['PAYMENT_API_KEY']}"},
    )
    response.raise_for_status()
    return response.json()

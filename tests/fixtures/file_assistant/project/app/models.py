"""Simple domain models used by the store demo."""

from dataclasses import dataclass


@dataclass
class Order:
    id: str
    total: float

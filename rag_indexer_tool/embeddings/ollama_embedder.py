from __future__ import annotations

from typing import Iterable

import requests


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed_one(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        items = list(texts)
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": items},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings")
        if embeddings is None and "embedding" in data:
            embeddings = [data["embedding"]]
        if not embeddings or len(embeddings) != len(items):
            raise RuntimeError(f"Unexpected Ollama embedding response: {data.keys()}")
        return embeddings

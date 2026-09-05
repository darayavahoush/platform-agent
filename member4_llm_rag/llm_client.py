"""Thin client for a locally-hosted small instruct model served by Ollama.
Prompted, not fine-tuned — gets a working MissionModule today without a
training pipeline. `format="json"` constrains Ollama's output so it's
reliably parseable.

Setup (one time):
    1. Install Ollama: https://ollama.com/download
    2. ollama pull qwen2.5:3b-instruct
    3. pip install requests
"""
from __future__ import annotations

import json
import time
from typing import Optional

import requests


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        host: str = "http://localhost:11434",
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def generate_json(self, prompt: str, system: Optional[str] = None) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2},
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return json.loads(resp.json()["response"])
            except Exception as e:  # noqa: BLE001 - retry on anything, surface last error
                last_err = e
                time.sleep(0.5 * (attempt + 1))

        raise RuntimeError(
            f"OllamaClient: failed to get valid JSON after {self.max_retries + 1} attempts: {last_err}"
        )

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False
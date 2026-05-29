"""Vertex POST client for the terka import.

Same endpoint kadar uploads to: POST /trajectories/?device_id=…
Returns the parsed JSON summary on success; raises requests
exceptions on transport / HTTP errors so the batch runner can
distinguish "skip and continue" from "stop".
"""

from __future__ import annotations

import requests


class VertexClient:
    def __init__(self, base_url: str, timeout_s: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._session = requests.Session()

    def upload(self, doc: dict, device_id: str) -> dict:
        r = self._session.post(
            f"{self.base_url}/trajectories/",
            params={"device_id": device_id},
            json=doc,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()

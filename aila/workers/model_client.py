from __future__ import annotations

import json
import urllib.request
from typing import Any


def post_for_json(
    url: str,
    *,
    data: bytes,
    content_type: str,
    timeout: float,
) -> Any | None:
    """POST ``data`` to ``url`` and return the parsed JSON response.

    Shared by the local model-backed workers (mic/whisper, camera/VLM) which
    all talk to an OpenAI-compatible HTTP endpoint. Returns ``None`` on any
    failure (unreachable host, timeout, non-JSON body) so callers can treat a
    backend outage uniformly rather than crashing the poll loop.
    """
    try:
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": content_type},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except Exception:
        return None

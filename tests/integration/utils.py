# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import json
import re
from typing import Any, Dict

# from ._support.gateway import GatewayApiClient


__all__ = [
    "REDACT_KEYS",
    "mask_secret_string",
    "redact_obj",
    "redacted_response_text",
]

# redact these common secret fields if present
REDACT_KEYS = {
    "value",
    "token",
    "access_token",
    "api_token",
    "apiKey",
    "api_key",
    "secret",
    "password",
}


def mask_secret_string(s: str) -> str:
    """Redact keys and replace api_keys with pak-****"""
    if not isinstance(s, str) or not s:
        return "<none>"
    # mask api_keys with pak-****
    if s.startswith("pak-"):
        return "pak-****"
    # mask other auth tokens (JWT-ish)
    if re.fullmatch(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", s):
        return "***TOKEN***"
    return "****"


def redact_obj(obj: Any) -> Any:
    """Redact dict/list/str recursively, preserving structure."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k in REDACT_KEYS and isinstance(v, str):
                out[k] = mask_secret_string(v)
            else:
                out[k] = redact_obj(v)
        return out
    if isinstance(obj, list):
        return [redact_obj(x) for x in obj]
    if isinstance(obj, str):
        s = obj
        # mask api_keys with pak-****
        s = re.sub(r"\bpak-[A-Za-z0-9]+\b", "pak-****", s)
        # mask other auth tokens (JWT-ish)
        s = re.sub(
            r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",
            "***TOKEN***",
            s,
        )
        return s
    return obj


def redacted_response_text(r) -> str:
    """Render a response with secrets redacted.
    Tries JSON first; falls back to plaintext masking.
    """
    try:
        data = r.json()
        return json.dumps(redact_obj(data), indent=2, ensure_ascii=False)
    except Exception:
        txt = getattr(r, "text", "") or ""
        txt = re.sub(r"\bpak-[A-Za-z0-9]+\b", "pak-****", txt)
        txt = re.sub(
            r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",
            "***TOKEN***",
            txt,
        )
        return txt

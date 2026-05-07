from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

from .constants import LLM_API_STYLE_OPTIONS


def default_llm_api_style() -> str:
    explicit = (
        os.environ.get("LLM_ROUTE_API_STYLE")
        or os.environ.get("LLM_CONTROL_API_STYLE")
        or os.environ.get("LLM_API_STYLE")
        or os.environ.get("API_STYLE")
        or ""
    ).strip()
    if explicit:
        return normalize_llm_api_style(explicit)
    if os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return "openai_chat"
    return "openai_chat"


def normalize_llm_api_style(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "openai": "openai_chat",
        "chat": "openai_chat",
        "chat_completions": "openai_chat",
        "openai_compatible": "openai_chat",
        "responses": "openai_responses",
        "openai_response": "openai_responses",
        "anthropic": "anthropic_sdk",
        "claude": "anthropic_sdk",
    }
    style = aliases.get(text, text)
    return style if style in LLM_API_STYLE_OPTIONS else "openai_chat"


def extract_json_object(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
    start = raw.find("{")
    if start < 0:
        return {}
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(raw)):
        char = raw[index]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start:index + 1]
                try:
                    parsed = json.loads(candidate)
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}
    return {}

"""Assemble the self-contained review dashboard: code-computed data + optional agent text.

The template draws everything with textContent, so agent text can never become
markup. The JSON is embedded with '<' escaped so no string can close the script tag.
"""
from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("dashboard_template.html")


def render(data: dict, ai: dict | None, meta: dict) -> str:
    payload = dict(data, ai=ai, meta=meta)
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    blob = blob.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = TEMPLATE.read_text(encoding="utf-8")
    assert html.count("/*__DATA__*/") == 1
    return html.replace("/*__DATA__*/", blob)

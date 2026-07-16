#!/usr/bin/env python3
"""Smoke tests for UI API (no live network except optional)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ui.api import (  # noqa: E402
    HUB,
    dispatch,
    handle_catalog,
    handle_health,
    handle_status,
)


def test_health_catalog_status():
    h = handle_health()
    assert h["ok"] is True
    cat = handle_catalog()
    assert cat["ok"] is True
    providers = {p["id"]: p for p in cat["catalog"]["providers"]}
    assert providers["grok"]["status"] == "active"
    assert providers["openai"]["status"] == "coming_soon"
    st = handle_status()
    assert st["ok"] is True
    assert "grok_key_present" in st
    # Never leak secrets
    blob = json.dumps(st) + json.dumps(cat)
    assert "xai-" not in blob.lower() or "xai-api" in blob.lower()
    # env key *names* are ok; values must not appear
    assert "sk-ant-" not in blob


def test_dispatch_routes():
    code, body = dispatch("GET", "/api/health")
    assert code == 200 and body["ok"]
    code, body = dispatch("GET", "/api/catalog")
    assert code == 200 and "catalog" in body
    code, body = dispatch("GET", "/api/nope")
    assert code == 404
    code, body = dispatch("POST", "/api/chat", {})
    assert code == 400
    assert "message" in body.get("error", "").lower() or body.get("ok") is False


def test_session_reset_export_empty():
    HUB.reset()
    code, body = dispatch("GET", "/api/session")
    assert code == 200 and body["session"]["active"] is False
    code, body = dispatch("POST", "/api/session/export", {})
    assert code == 400


def test_chat_rejects_client_key_injection_path():
    # Server strips api_key before dispatch in server.py; API itself never needs it from body
    code, body = dispatch(
        "POST",
        "/api/chat",
        {"message": "", "api_key": "xai-should-never-echo"},
    )
    assert code == 400
    assert "xai-should-never-echo" not in json.dumps(body)


if __name__ == "__main__":
    test_health_catalog_status()
    test_dispatch_routes()
    test_session_reset_export_empty()
    test_chat_rejects_client_key_injection_path()
    print("test_ui_api OK")

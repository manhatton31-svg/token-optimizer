"""
JSON API for the customer UI — no framework deps.

Endpoints (handled by server.py):
  GET  /api/health
  GET  /api/catalog
  GET  /api/status
  POST /api/calibrate   {profile?, persist?}
  POST /api/chat        {message, profile?, model?, effort?, system?}
  GET  /api/session
  POST /api/session/reset
  POST /api/session/export  → writes JSONL under ui/_exports/

Never returns API keys or raw .env contents.
"""
from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Project root on path when launched via ui/server.py
from token_optimizer import GrokSession, product_catalog
from token_optimizer.grok import PROFILES, _api_key, _load_dotenv
from token_optimizer.onboarding import OnboardingService, PROVIDER_BILLING

EXPORT_DIR = Path(__file__).resolve().parent / "_exports"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_error(exc: BaseException) -> dict[str, Any]:
    msg = str(exc)
    # Belt-and-suspenders: strip anything that looks like a key
    for prefix in ("xai-", "sk-", "sk-ant-", "AIza"):
        if prefix in msg.lower() or prefix in msg:
            msg = "API error (details redacted)"
            break
    return {"ok": False, "error": msg, "type": type(exc).__name__}


def _key_present_grok() -> bool:
    _load_dotenv()
    try:
        _api_key(None)
        return True
    except Exception:
        return False


def handle_health() -> dict[str, Any]:
    return {"ok": True, "service": "token-optimizer-ui", "ts": _now()}


def handle_catalog() -> dict[str, Any]:
    cat = product_catalog()
    return {"ok": True, "catalog": cat}


def handle_status() -> dict[str, Any]:
    _load_dotenv()
    svc = OnboardingService()
    providers = []
    for name, meta in PROVIDER_BILLING.items():
        ready = svc.readiness(name)
        lock = svc.state.get("providers", {}).get(name, {})
        # Never include key values — only presence + which env names to set
        providers.append(
            {
                "id": name,
                "name": meta["label"],
                "status": (
                    "active"
                    if name == "grok"
                    else "coming_soon"
                ),
                "key_present": bool(ready.get("key_present")),
                "env_keys": list(meta["env_keys"]),
                "can_live_calibrate": bool(ready.get("can_live_calibrate")),
                "locked": bool(ready.get("locked")),
                "model": lock.get("model") or meta["default_model"],
                "token_accuracy": lock.get("token_accuracy"),
                "cost_accuracy": lock.get("cost_accuracy"),
                "dashboard": meta.get("dashboard_hint"),
            }
        )
    ticks = svc.state.get("cost_usd_ticks") or {}
    return {
        "ok": True,
        "grok_key_present": _key_present_grok(),
        "providers": providers,
        "cost_usd_ticks": {
            "scale": ticks.get("scale"),
            "status": ticks.get("status"),
            "scale_note": ticks.get("scale_note"),
        },
        "profiles": list(PROFILES.keys()),
        "default_profile": "production",
        "ts": _now(),
    }


class SessionHub:
    """In-memory GrokSession for the local UI process."""

    def __init__(self) -> None:
        self.session: GrokSession | None = None
        self.profile: str = "production"

    def ensure(self, profile: str | None = None, **kwargs: Any) -> GrokSession:
        name = (profile or self.profile or "production").strip().lower()
        if name not in PROFILES:
            raise ValueError(f"Unknown profile {name!r}")
        if self.session is None or self.profile != name:
            self.session = GrokSession(profile=name, live_prices=True, **kwargs)
            self.profile = name
        return self.session

    def reset(self) -> None:
        self.session = None
        self.profile = "production"

    def summary(self) -> dict[str, Any]:
        if self.session is None:
            return {
                "active": False,
                "profile": self.profile,
                "session_calls": 0,
                "session_api_total": 0,
            }
        s = self.session.summary()
        s["active"] = True
        # Strip anything sensitive if ever nested
        return s


HUB = SessionHub()


def handle_session() -> dict[str, Any]:
    return {"ok": True, "session": HUB.summary()}


def handle_session_reset() -> dict[str, Any]:
    HUB.reset()
    return {"ok": True, "session": HUB.summary()}


def handle_calibrate(body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    profile = str(body.get("profile") or "production").strip().lower()
    persist = bool(body.get("persist", True))
    if not _key_present_grok():
        return {
            "ok": False,
            "error": "XAI_API_KEY not set. Add it to ~/.env or the process environment.",
            "env_keys": list(PROVIDER_BILLING["grok"]["env_keys"]),
        }
    try:
        # Fresh session for a clean probe
        HUB.reset()
        session = HUB.ensure(profile=profile, max_tokens_out=64)
        cal = session.calibrate(persist=persist)
        # Redact free-form text that might be long; keep short probe reply
        safe = {
            k: cal[k]
            for k in (
                "provider",
                "status",
                "model",
                "profile",
                "api_total",
                "cost_in_usd_ticks",
                "ticks_per_usd",
                "api_cost_usd",
                "est_cost_usd",
                "accuracy",
                "within_99_5",
                "persisted",
                "rates",
                "seconds",
                "cost",
            )
            if k in cal
        }
        safe["text"] = (cal.get("text") or "")[:80]
        return {
            "ok": True,
            "calibration": safe,
            "session": HUB.summary(),
            "status": handle_status(),
        }
    except Exception as e:
        return _safe_error(e)


def handle_chat(body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    message = (body.get("message") or body.get("prompt") or "").strip()
    if not message:
        return {"ok": False, "error": "message is required"}
    if not _key_present_grok():
        return {
            "ok": False,
            "error": "XAI_API_KEY not set. Add it to ~/.env or the process environment.",
            "env_keys": list(PROVIDER_BILLING["grok"]["env_keys"]),
        }
    profile = str(body.get("profile") or HUB.profile or "production").strip().lower()
    model = body.get("model") or None
    if model is not None:
        model = str(model).strip() or None
    effort_raw = body.get("effort")
    effort: str | None
    if effort_raw is None or str(effort_raw).strip() == "":
        effort = None
    else:
        effort = str(effort_raw).strip().lower()
        if effort in ("default", "null", "profile"):
            effort = None
        # "none" / "off" mean disable reasoning; passed through to GrokSession
    system = body.get("system")
    max_out = int(body.get("max_tokens_out") or 256)
    try:
        session = HUB.ensure(profile=profile, max_tokens_out=max_out)
        # If profile changed mid-request via body, ensure already handled
        if model or effort is not None:
            result = session.chat(
                message,
                model=model,
                effort=effort,
                system=system,
            )
        else:
            result = session.chat(message, system=system)
        return {
            "ok": True,
            "result": {
                "text": result.text,
                "model": result.model,
                "profile": result.profile,
                "seconds": result.seconds,
                "api_total": result.api_total,
                "usage": {
                    k: result.usage.get(k)
                    for k in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "cached_tokens",
                        "reasoning_tokens",
                        "cost_in_usd_ticks",
                    )
                },
                "cost_usd_api": result.cost_usd_api,
                "cost_usd_est": result.cost_usd_est,
                "warnings": list(result.warnings or [])[:8],
            },
            "session": HUB.summary(),
            "cost": session.compare_cost_accuracy(),
        }
    except Exception as e:
        err = _safe_error(e)
        # Include traceback only in a flag — default off
        if body.get("debug"):
            err["trace"] = traceback.format_exc()[-2000:]
        return err


def handle_export() -> dict[str, Any]:
    if HUB.session is None:
        return {"ok": False, "error": "No active session to export"}
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DIR / f"session_{_now().replace(':', '')}.jsonl"
    n = HUB.session.export_jsonl(path)
    return {
        "ok": True,
        "path": str(path),
        "calls_written": n,
        "session": HUB.summary(),
    }


def dispatch(method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    """Route API call → (http_status, json_body)."""
    path = path.rstrip("/") or "/"
    if method == "GET" and path == "/api/health":
        return 200, handle_health()
    if method == "GET" and path == "/api/catalog":
        return 200, handle_catalog()
    if method == "GET" and path == "/api/status":
        return 200, handle_status()
    if method == "GET" and path == "/api/session":
        return 200, handle_session()
    if method == "POST" and path == "/api/session/reset":
        return 200, handle_session_reset()
    if method == "POST" and path == "/api/calibrate":
        out = handle_calibrate(body)
        return (200 if out.get("ok") else 400), out
    if method == "POST" and path == "/api/chat":
        out = handle_chat(body)
        return (200 if out.get("ok") else 400), out
    if method == "POST" and path == "/api/session/export":
        out = handle_export()
        return (200 if out.get("ok") else 400), out
    return 404, {"ok": False, "error": f"Unknown route {method} {path}"}

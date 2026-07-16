"""
Multi-provider onboarding / calibration service.

Learned from Grok (100% token match once dashboard lag clears):
  1. Use LIVE usage from the completion response (never guess token counts).
  2. Use LIVE price sheets when the provider exposes them (xAI models API).
  3. Bill every component: uncached in, cached in, output, reasoning/thinking.
  4. Verify with dashboard before/after delta → target accuracy >= 99.5%.

Providers: grok | openai | anthropic | gemini
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .costing import (
    cost_within_tolerance,
    estimate_cost_usd,
    fetch_xai_model_pricing,
    normalize_usage,
)
from .tokenizers import COST_PRESETS, resolve_provider

# ---------------------------------------------------------------------------
# Billable-component map (how each frontier dashboard typically counts)
# ---------------------------------------------------------------------------

PROVIDER_BILLING: dict[str, dict[str, Any]] = {
    "grok": {
        "label": "xAI Grok",
        "env_keys": ("XAI_API_KEY", "GROK_API_KEY", "XAI_KEY"),
        "default_model": "grok-4.3",
        "base_url": "https://api.x.ai/v1",
        "components": (
            "uncached_prompt @ pin",
            "cached_prompt @ pin_cached",
            "completion @ pout",
            "reasoning @ pout  # billed as output",
        ),
        "live_prices": "xai-models-api",
        "dashboard_hint": "console.x.ai → Usage (API), not grok.com chat",
    },
    "openai": {
        "label": "OpenAI",
        "env_keys": ("OPENAI_API_KEY",),
        "default_model": "gpt-4o-mini",
        "components": (
            "prompt @ pin (minus cached if reported)",
            "cached_prompt @ pin_cached if present",
            "completion @ pout",
            "reasoning @ pout for o-series when present",
        ),
        "live_prices": "preset+usage",
        "dashboard_hint": "platform.openai.com → Usage",
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "env_keys": ("ANTHROPIC_API_KEY",),
        "default_model": "claude-sonnet-4-20250514",
        "components": (
            "input_tokens @ pin",
            "cache_read_input_tokens @ pin_cached",
            "cache_creation_input_tokens @ pin_cache_write",
            "output_tokens @ pout",
        ),
        "live_prices": "preset",
        "dashboard_hint": "console.anthropic.com → Usage",
        # Anthropic cache write is often 1.25x input; cache read 0.1x
        "pin_cache_write_mult": 1.25,
        "pin_cached_mult": 0.1,
    },
    "gemini": {
        "label": "Google Gemini",
        "env_keys": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "default_model": "gemini-2.0-flash",
        "components": (
            "prompt_token_count @ pin",
            "candidates_token_count @ pout",
            "thoughts_token_count @ pout when present",
        ),
        "live_prices": "preset",
        "dashboard_hint": "aistudio.google.com / Cloud billing",
    },
}

STATE_PATH = Path(__file__).resolve().parent.parent / "calibration_state.json"
LEDGER_PATH = Path(__file__).resolve().parent.parent / "calibration_ledger.jsonl"

TARGET_ACCURACY = 0.995


@dataclass
class ProviderLock:
    """Locked calibration for one provider after >= 99.5% verification."""

    provider: str
    model: str
    status: str  # locked | pending | unverified
    token_accuracy: float | None = None
    cost_accuracy: float | None = None
    pin: float = 0.0
    pout: float = 0.0
    pin_cached: float = 0.0
    price_source: str = ""
    last_api_total: int = 0
    last_est_cost_usd: float = 0.0
    dashboard_total: int | None = None
    notes: list[str] = field(default_factory=list)
    updated_at: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _first_env(names: tuple[str, ...]) -> str | None:
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return None


def load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {
        "target_accuracy": TARGET_ACCURACY,
        "providers": {},
        "formula": {
            "grok": "uncached*pin + cached*pin_cached + (completion+reasoning)*pout",
            "openai": "prompt*pin + completion*pout (+ cache/reasoning when present)",
            "anthropic": "input*pin + cache_read*pin_cached + cache_write*pin_write + output*pout",
            "gemini": "prompt*pin + (candidates+thoughts)*pout",
        },
    }


def save_state(state: dict[str, Any], path: Path = STATE_PATH) -> None:
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def append_ledger(row: dict[str, Any], path: Path = LEDGER_PATH) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def resolve_rates(
    provider: str,
    model: str | None = None,
    *,
    api_key: str | None = None,
    pin: float | None = None,
    pout: float | None = None,
) -> dict[str, Any]:
    """Preset rates + live xAI sheet when available."""
    meta = PROVIDER_BILLING[provider]
    preset_key = "grok" if provider == "grok" else (
        "openai-mini" if provider == "openai" else provider
    )
    if provider == "openai":
        preset_key = "openai-mini"
    elif provider == "anthropic":
        preset_key = "anthropic"
    elif provider == "gemini":
        preset_key = "gemini"

    resolved = resolve_provider(preset_key, pin=pin, pout=pout, model=model)
    rates = {
        "pin": float(resolved["pin"]),
        "pout": float(resolved["pout"]),
        "pin_cached": float(resolved.get("pin_cached") or resolved["pin"]),
        "pin_cache_write": float(resolved["pin"]) * float(
            meta.get("pin_cache_write_mult") or 1.0
        ),
        "label": resolved.get("label") or meta["label"],
        "model": model or resolved.get("model") or meta["default_model"],
        "price_source": f"preset:{preset_key}",
    }

    # Anthropic default cache multipliers if not in preset
    if provider == "anthropic":
        rates["pin_cached"] = float(resolved["pin"]) * float(
            meta.get("pin_cached_mult") or 0.1
        )
        rates["pin_cache_write"] = float(resolved["pin"]) * float(
            meta.get("pin_cache_write_mult") or 1.25
        )

    if provider == "grok":
        key = api_key or _first_env(tuple(meta["env_keys"]))
        if key:
            try:
                live = fetch_xai_model_pricing(rates["model"], api_key=key)
                if pin is None:
                    rates["pin"] = float(live["pin"])
                if pout is None:
                    rates["pout"] = float(live["pout"])
                rates["pin_cached"] = float(live["pin_cached"])
                rates["model"] = live["model"]
                rates["price_source"] = "xai-models-api"
            except Exception as e:
                rates["price_warning"] = str(e)
    return rates


def estimate_provider_cost(
    provider: str,
    usage: dict[str, Any],
    rates: dict[str, Any],
) -> dict[str, Any]:
    """
    Provider-aware cost estimate from LIVE usage + rates.
    """
    if provider == "anthropic":
        # Flatten Anthropic-style fields into estimate_cost_usd shape
        inp = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        out = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        cache_write = int(usage.get("cache_creation_input_tokens") or 0)
        # treat non-cache input as uncached
        uncached = max(0, inp - cache_read)
        flat = {
            "prompt_tokens": inp,
            "completion_tokens": out,
            "cached_tokens": cache_read,
            "reasoning_tokens": 0,
            "total_tokens": inp + out + cache_write,  # write often extra billable
        }
        base = estimate_cost_usd(
            flat,
            pin=rates["pin"],
            pout=rates["pout"],
            pin_cached=rates["pin_cached"],
        )
        write_cost = cache_write / 1e6 * float(rates.get("pin_cache_write") or rates["pin"])
        base["cost_usd"] = float(base["cost_usd"]) + write_cost
        base["breakdown"]["cache_write_usd"] = write_cost
        base["tokens"]["cache_write"] = cache_write
        base["tokens"]["uncached_prompt"] = uncached
        return base

    if provider == "gemini":
        prompt = int(
            usage.get("prompt_token_count")
            or usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or 0
        )
        candidates = int(
            usage.get("candidates_token_count")
            or usage.get("completion_tokens")
            or usage.get("output_tokens")
            or 0
        )
        thoughts = int(
            usage.get("thoughts_token_count")
            or usage.get("reasoning_tokens")
            or 0
        )
        flat = {
            "prompt_tokens": prompt,
            "completion_tokens": candidates,
            "reasoning_tokens": thoughts,
            "cached_tokens": int(usage.get("cached_content_token_count") or 0),
            "total_tokens": int(usage.get("total_token_count") or 0)
            or (prompt + candidates + thoughts),
        }
        return estimate_cost_usd(
            flat,
            pin=rates["pin"],
            pout=rates["pout"],
            pin_cached=rates.get("pin_cached"),
        )

    # grok + openai (OpenAI-compatible usage shape)
    return estimate_cost_usd(
        usage,
        pin=rates["pin"],
        pout=rates["pout"],
        pin_cached=rates.get("pin_cached"),
    )


def verify_token_delta(
    api_total: int,
    dashboard_before: int,
    dashboard_after: int,
    *,
    min_accuracy: float = TARGET_ACCURACY,
) -> dict[str, Any]:
    delta = int(dashboard_after) - int(dashboard_before)
    check = cost_within_tolerance(
        float(api_total), float(delta), min_accuracy=min_accuracy
    )
    check["kind"] = "token_delta"
    check["dashboard_before"] = dashboard_before
    check["dashboard_after"] = dashboard_after
    check["dashboard_delta"] = delta
    check["api_total"] = api_total
    # Exact match common once lag clears
    check["exact"] = api_total == delta
    return check


class OnboardingService:
    """
    Extrapolate Grok-proven calibration to all frontiers.

    Workflow per provider:
      1. ensure_key / dry-run readiness
      2. run_probe() → live usage + live rates → est cost
      3. user reports dashboard before/after
      4. verify() → lock if accuracy >= 99.5%
    """

    def __init__(self, state_path: Path = STATE_PATH):
        self.state_path = state_path
        self.state = load_state(state_path)

    def list_providers(self) -> list[dict[str, Any]]:
        rows = []
        for name, meta in PROVIDER_BILLING.items():
            key_ok = bool(_first_env(tuple(meta["env_keys"])))
            lock = self.state.get("providers", {}).get(name, {})
            rows.append(
                {
                    "provider": name,
                    "label": meta["label"],
                    "key_present": key_ok,
                    "status": lock.get("status", "unverified"),
                    "token_accuracy": lock.get("token_accuracy"),
                    "model": lock.get("model") or meta["default_model"],
                    "dashboard_hint": meta["dashboard_hint"],
                }
            )
        return rows

    def readiness(self, provider: str) -> dict[str, Any]:
        meta = PROVIDER_BILLING[provider]
        key = _first_env(tuple(meta["env_keys"]))
        return {
            "provider": provider,
            "key_present": bool(key),
            "env_keys": meta["env_keys"],
            "default_model": meta["default_model"],
            "components": meta["components"],
            "dashboard_hint": meta["dashboard_hint"],
            "can_live_calibrate": bool(key),
            "locked": self.state.get("providers", {}).get(provider, {}).get("status")
            == "locked",
        }

    def lock_provider(
        self,
        provider: str,
        *,
        model: str,
        rates: dict[str, Any],
        api_total: int,
        est_cost: float,
        token_accuracy: float | None,
        cost_accuracy: float | None = None,
        dashboard_total: int | None = None,
        notes: list[str] | None = None,
    ) -> ProviderLock:
        status = "pending"
        if token_accuracy is not None and token_accuracy >= TARGET_ACCURACY:
            status = "locked"
        lock = ProviderLock(
            provider=provider,
            model=model,
            status=status,
            token_accuracy=token_accuracy,
            cost_accuracy=cost_accuracy,
            pin=float(rates["pin"]),
            pout=float(rates["pout"]),
            pin_cached=float(rates.get("pin_cached") or rates["pin"]),
            price_source=str(rates.get("price_source") or ""),
            last_api_total=api_total,
            last_est_cost_usd=est_cost,
            dashboard_total=dashboard_total,
            notes=list(notes or []),
            updated_at=_now(),
        )
        self.state.setdefault("providers", {})[provider] = asdict(lock)
        self.state["updated_at"] = _now()
        save_state(self.state, self.state_path)
        return lock

    def apply_lock_to_presets(self) -> None:
        """Push locked rates into COST_PRESETS for TokenOptimizer(provider=...)."""
        for name, lock in self.state.get("providers", {}).items():
            if lock.get("status") != "locked":
                continue
            if name not in COST_PRESETS:
                COST_PRESETS[name] = {"label": name, "tokenizer": "approx"}
            COST_PRESETS[name]["pin"] = lock["pin"]
            COST_PRESETS[name]["pout"] = lock["pout"]
            COST_PRESETS[name]["pin_cached"] = lock.get("pin_cached", lock["pin"])
            COST_PRESETS[name]["model"] = lock.get("model")
            COST_PRESETS[name]["calibrated"] = True
            COST_PRESETS[name]["calibrated_at"] = lock.get("updated_at")

    def record_run(
        self,
        provider: str,
        *,
        model: str,
        usage: dict[str, Any],
        rates: dict[str, Any],
        seconds: float,
        mode: str,
        dashboard_before: int | None = None,
        dashboard_after: int | None = None,
    ) -> dict[str, Any]:
        est = estimate_provider_cost(provider, usage, rates)
        toks = est.get("tokens") or normalize_usage(usage)
        api_total = int(toks.get("total") or usage.get("total_tokens") or 0)
        row = {
            "ts": _now(),
            "provider": provider,
            "model": model,
            "mode": mode,
            "usage": usage,
            "tokens": toks,
            "api_total": api_total,
            "est_cost_usd": est["cost_usd"],
            "breakdown": est.get("breakdown"),
            "rates": {
                "pin": rates["pin"],
                "pout": rates["pout"],
                "pin_cached": rates.get("pin_cached"),
                "price_source": rates.get("price_source"),
            },
            "seconds": seconds,
            "dashboard_before": dashboard_before,
            "dashboard_after": dashboard_after,
        }
        if dashboard_before is not None and dashboard_after is not None:
            row["accuracy"] = verify_token_delta(
                api_total, dashboard_before, dashboard_after
            )
            if row["accuracy"].get("ok"):
                self.lock_provider(
                    provider,
                    model=model,
                    rates=rates,
                    api_total=api_total,
                    est_cost=float(est["cost_usd"]),
                    token_accuracy=float(row["accuracy"]["accuracy"]),
                    dashboard_total=dashboard_after,
                    notes=["Verified via dashboard before/after delta"],
                )
                self.apply_lock_to_presets()
        append_ledger(row)
        return row

    def status_report(self) -> str:
        lines = [
            "=" * 72,
            "ONBOARDING / CALIBRATION STATUS",
            "=" * 72,
            f"target_accuracy: {self.state.get('target_accuracy', TARGET_ACCURACY)}",
            f"state_file: {self.state_path}",
            "-" * 72,
        ]
        for row in self.list_providers():
            acc = row.get("token_accuracy")
            acc_s = f"{acc*100:.2f}%" if isinstance(acc, (int, float)) else "—"
            lines.append(
                f"{row['provider']:10} key={'yes' if row['key_present'] else 'no ':3} "
                f"status={row['status']:10} acc={acc_s:8} model={row['model']}"
            )
            lines.append(f"           dashboard: {row['dashboard_hint']}")
        lines.append("=" * 72)
        lines.append(
            "Grok path is proven: use LIVE usage total_tokens vs console delta."
        )
        lines.append(
            "Other frontiers: same loop — run probe, report before/after, lock at 99.5%+."
        )
        lines.append("=" * 72)
        return "\n".join(lines)


def bootstrap_grok_lock_from_session() -> None:
    """
    Persist the Grok calibration we already proved in-session:
    exact token deltas 1074, 875, 1007 vs dashboard.
    """
    svc = OnboardingService()
    rates = resolve_rates("grok", "grok-4.3")
    # 100% token accuracy proven across three sequential dashboard jumps
    svc.lock_provider(
        "grok",
        model="grok-4.3",
        rates=rates,
        api_total=1007,
        est_cost=0.0015593,
        token_accuracy=1.0,
        dashboard_total=6787,
        notes=[
            "Session-verified exact matches:",
            "3831+1074=4905, 4905+875=5780, 5780+1007=6787",
            "Formula: uncached*pin + cached*pin_cached + (out+reasoning)*pout",
            "Prices from xAI models API (pin=1.25, cached=0.20, pout=2.50)",
        ],
    )
    svc.apply_lock_to_presets()
    # Also mark others as ready-for-key
    for p in ("openai", "anthropic", "gemini"):
        if p not in svc.state.get("providers", {}):
            meta = PROVIDER_BILLING[p]
            r = resolve_rates(p, meta["default_model"])
            svc.state["providers"][p] = asdict(
                ProviderLock(
                    provider=p,
                    model=meta["default_model"],
                    status="pending_key" if not _first_env(tuple(meta["env_keys"])) else "unverified",
                    pin=r["pin"],
                    pout=r["pout"],
                    pin_cached=r.get("pin_cached", r["pin"]),
                    price_source=r.get("price_source", "preset"),
                    notes=[
                        "Same onboarding loop as Grok once API key is set.",
                        f"Components: {', '.join(meta['components'])}",
                    ],
                    updated_at=_now(),
                )
            )
    save_state(svc.state)

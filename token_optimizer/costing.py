"""
Accurate cost estimation from provider usage + live price sheets.

xAI encodes token prices on the models API as integers where:
    usd_per_1M = price_field / 10_000

Billable components (Grok):
  - uncached prompt tokens  @ prompt_text_token_price
  - cached prompt tokens    @ cached_prompt_text_token_price
  - completion tokens       @ completion_text_token_price
  - reasoning tokens        @ completion_text_token_price  (billed as output)
"""
from __future__ import annotations

import os
import re
from typing import Any, Callable

# xAI models.list() price fields are in 1e-4 USD per 1M? 
# Empirically: 12500 → $1.25/1M ⇒ divide by 10_000
XAI_PRICE_SCALE = 10_000.0


def xai_field_to_usd_per_m(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / XAI_PRICE_SCALE


def fetch_xai_model_pricing(
    model: str,
    *,
    api_key: str | None = None,
    base_url: str = "https://api.x.ai/v1",
) -> dict[str, Any]:
    """
    Pull live rates for a Grok model from the xAI models API.
    Returns pin, pout, pin_cached (USD / 1M) plus raw fields.
    """
    from openai import OpenAI  # type: ignore

    key = api_key or os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not key:
        raise RuntimeError("XAI_API_KEY required to fetch live Grok pricing")

    client = OpenAI(api_key=key, base_url=base_url)
    target = model.strip()
    hit = None
    for m in client.models.list().data:
        if m.id == target or m.id.replace("_", "-") == target:
            hit = m
            break
    if hit is None:
        # prefix match
        for m in client.models.list().data:
            if target in m.id:
                hit = m
                break
    if hit is None:
        raise RuntimeError(f"Model not found on xAI: {model}")

    d = hit.model_dump() if hasattr(hit, "model_dump") else dict(getattr(hit, "__dict__", {}))
    pin = xai_field_to_usd_per_m(d.get("prompt_text_token_price"))
    pout = xai_field_to_usd_per_m(d.get("completion_text_token_price"))
    pin_cached = xai_field_to_usd_per_m(d.get("cached_prompt_text_token_price"))
    pin_long = xai_field_to_usd_per_m(d.get("prompt_text_token_price_long_context"))
    pout_long = xai_field_to_usd_per_m(d.get("completion_text_token_price_long_context"))

    return {
        "model": hit.id,
        "pin": pin if pin is not None else 1.25,
        "pout": pout if pout is not None else 2.50,
        "pin_cached": pin_cached if pin_cached is not None else (pin or 1.25) * 0.16,
        "pin_long": pin_long,
        "pout_long": pout_long,
        "raw": {
            k: d.get(k)
            for k in (
                "prompt_text_token_price",
                "cached_prompt_text_token_price",
                "completion_text_token_price",
                "prompt_text_token_price_long_context",
                "completion_text_token_price_long_context",
            )
        },
        "source": "xai-models-api",
    }


def parse_int_field(obj: Any, *names: str) -> int:
    if obj is None:
        return 0
    if isinstance(obj, dict):
        for n in names:
            if n in obj and obj[n] is not None:
                return int(obj[n])
    for n in names:
        try:
            v = getattr(obj, n, None)
            if v is not None:
                return int(v)
        except Exception:
            pass
    # stringified details
    s = str(obj)
    for n in names:
        m = re.search(rf"{n}=(\d+)", s)
        if m:
            return int(m.group(1))
    return 0


def normalize_usage(usage: dict[str, Any]) -> dict[str, int]:
    """Flatten provider usage into billable buckets."""
    prompt = int(
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or 0
    )
    completion = int(
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or 0
    )
    total = int(usage.get("total_tokens") or 0)
    reasoning = int(usage.get("reasoning_tokens") or 0)
    if not reasoning:
        reasoning = parse_int_field(
            usage.get("completion_tokens_details"), "reasoning_tokens"
        )
    cached = int(usage.get("cached_tokens") or 0)
    if not cached:
        cached = parse_int_field(
            usage.get("prompt_tokens_details"), "cached_tokens"
        )
    # cached cannot exceed prompt
    cached = min(cached, prompt) if prompt else cached
    uncached = max(0, prompt - cached)

    # If API total > prompt+completion, attribute remainder to reasoning
    if total > prompt + completion and reasoning == 0:
        reasoning = total - prompt - completion

    return {
        "prompt": prompt,
        "uncached_prompt": uncached,
        "cached_prompt": cached,
        "completion": completion,
        "reasoning": reasoning,
        "output_billable": completion + reasoning,
        "total": total or (prompt + completion + reasoning),
    }


def estimate_cost_usd(
    usage: dict[str, Any],
    *,
    pin: float,
    pout: float,
    pin_cached: float | None = None,
) -> dict[str, Any]:
    """
    Estimate USD cost from usage + rates (per 1M tokens).

    Returns breakdown suitable for 99.5%+ match when rates come from the
    provider price sheet and usage is the live API usage object.
    """
    u = normalize_usage(usage)
    pin_c = pin if pin_cached is None else pin_cached

    cost_uncached = u["uncached_prompt"] / 1e6 * pin
    cost_cached = u["cached_prompt"] / 1e6 * pin_c
    cost_completion = u["completion"] / 1e6 * pout
    cost_reasoning = u["reasoning"] / 1e6 * pout
    total = cost_uncached + cost_cached + cost_completion + cost_reasoning

    return {
        "cost_usd": total,
        "breakdown": {
            "uncached_input_usd": cost_uncached,
            "cached_input_usd": cost_cached,
            "completion_usd": cost_completion,
            "reasoning_usd": cost_reasoning,
        },
        "tokens": u,
        "rates": {
            "pin": pin,
            "pin_cached": pin_c,
            "pout": pout,
        },
    }


def cost_within_tolerance(
    estimated: float,
    actual: float,
    *,
    min_accuracy: float = 0.995,
) -> dict[str, Any]:
    """
    Check estimated is within min_accuracy of actual
    (e.g. 0.995 ⇒ |est-actual|/actual ≤ 0.5%).
    """
    if actual <= 0:
        return {
            "ok": estimated == 0,
            "accuracy": 1.0 if estimated == 0 else 0.0,
            "rel_error": 0.0 if estimated == 0 else float("inf"),
            "estimated": estimated,
            "actual": actual,
            "min_accuracy": min_accuracy,
        }
    rel = abs(estimated - actual) / actual
    accuracy = 1.0 - rel
    return {
        "ok": accuracy + 1e-15 >= min_accuracy,
        "accuracy": accuracy,
        "rel_error": rel,
        "estimated": estimated,
        "actual": actual,
        "min_accuracy": min_accuracy,
    }


# xAI reports cost_in_usd_ticks on usage. Empirically (2026-07):
#   usd = ticks / 10_000_000_000
# Verified: ticks=6546500 ≈ $0.00065465 matches sheet estimate for same usage.
XAI_USD_TICKS_PER_DOLLAR = 10_000_000_000  # 1e10


def ticks_to_usd(
    ticks: int | float | None,
    *,
    scale: int | float = XAI_USD_TICKS_PER_DOLLAR,
) -> float | None:
    """Convert xAI cost_in_usd_ticks → USD. Returns None if ticks missing."""
    if ticks is None:
        return None
    try:
        t = float(ticks)
    except (TypeError, ValueError):
        return None
    if t < 0 or scale <= 0:
        return None
    return t / float(scale)


def parse_cost_in_usd_ticks(usage: Any) -> int | None:
    """Extract cost_in_usd_ticks from a usage object or dict."""
    if usage is None:
        return None
    if isinstance(usage, dict):
        v = usage.get("cost_in_usd_ticks")
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
        return None
    v = getattr(usage, "cost_in_usd_ticks", None)
    if v is not None:
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    # stringified
    m = re.search(r"cost_in_usd_ticks[=:]\s*(\d+)", str(usage))
    if m:
        return int(m.group(1))
    return None


def compare_cost_accuracy(
    api_cost: float | None,
    est_cost: float | None,
    *,
    min_accuracy: float = 0.995,
    dashboard_cost: float | None = None,
) -> dict[str, Any]:
    """
    Compare provider-reported API cost (from ticks) vs our sheet estimate,
    and optionally vs console dashboard USD.
    """
    out: dict[str, Any] = {
        "api_cost": api_cost,
        "est_cost": est_cost,
        "dashboard_cost": dashboard_cost,
        "min_accuracy": min_accuracy,
        "api_vs_est": None,
        "api_vs_dashboard": None,
        "est_vs_dashboard": None,
        "within_99_5": False,
        "primary": None,
    }
    if api_cost is not None and est_cost is not None:
        out["api_vs_est"] = cost_within_tolerance(
            float(est_cost), float(api_cost), min_accuracy=min_accuracy
        )
        # Note: cost_within_tolerance(estimated, actual) — here est is estimate, api is truth
        out["primary"] = "api_vs_est"
        out["within_99_5"] = bool(out["api_vs_est"].get("ok"))
        out["accuracy"] = out["api_vs_est"].get("accuracy")
    if api_cost is not None and dashboard_cost is not None:
        out["api_vs_dashboard"] = cost_within_tolerance(
            float(api_cost), float(dashboard_cost), min_accuracy=min_accuracy
        )
        out["primary"] = "api_vs_dashboard"
        out["within_99_5"] = bool(out["api_vs_dashboard"].get("ok"))
        out["accuracy"] = out["api_vs_dashboard"].get("accuracy")
    elif est_cost is not None and dashboard_cost is not None and out["primary"] is None:
        out["est_vs_dashboard"] = cost_within_tolerance(
            float(est_cost), float(dashboard_cost), min_accuracy=min_accuracy
        )
        out["primary"] = "est_vs_dashboard"
        out["within_99_5"] = bool(out["est_vs_dashboard"].get("ok"))
        out["accuracy"] = out["est_vs_dashboard"].get("accuracy")
    return out

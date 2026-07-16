#!/usr/bin/env python3
r"""
==============================================================================
 token-optimizer — Provider calibration (onboarding baseline)
==============================================================================

PURPOSE
  Run ONE standardized, agent-style prompt against a real frontier provider
  and print the numbers that match (or closely match) the provider dashboard:
    • input tokens
    • output tokens
    • wall time
    • estimated USD cost (current preset rates in token_optimizer)

SUPPORTED PROVIDERS
  openai | anthropic | grok | gemini

-----------------------------------------------------------------------------
 HOW TO SET API KEYS
-----------------------------------------------------------------------------

  Windows PowerShell:
    $env:OPENAI_API_KEY    = "sk-..."
    $env:ANTHROPIC_API_KEY = "sk-ant-..."
    $env:XAI_API_KEY       = "xai-..."      # Grok (also accepts GROK_API_KEY)
    $env:GOOGLE_API_KEY    = "AIza..."      # Gemini (also accepts GEMINI_API_KEY)

  macOS / Linux (bash):
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    export XAI_API_KEY=xai-...
    export GOOGLE_API_KEY=AIza...

  Optional model override:
    $env:CALIBRATE_MODEL = "gpt-4o-mini"
    export CALIBRATE_MODEL=claude-sonnet-4-20250514

-----------------------------------------------------------------------------
 HOW TO RUN
-----------------------------------------------------------------------------

  # Provider via CLI
  python calibrate.py --provider openai
  python calibrate.py -p anthropic
  python calibrate.py -p grok
  python calibrate.py -p gemini

  # Provider via environment
  $env:CALIBRATE_PROVIDER = "openai"   # or export CALIBRATE_PROVIDER=openai
  python calibrate.py

  # Local dry-run (no network; estimates tokens only)
  python calibrate.py -p openai --dry-run

  # Optional packages for live calibration
  pip install openai anthropic tiktoken google-genai

-----------------------------------------------------------------------------
 NOTES
-----------------------------------------------------------------------------
  • Live runs require the matching SDK + API key.
  • If the SDK or key is missing, the script falls back gracefully and still
    prints a baseline using local token estimates (clearly labeled).
  • Pricing comes from token_optimizer COST_PRESETS (USD per 1M tokens).
    Override with --pin / --pout if your dashboard rates differ.
==============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Load .env (no python-dotenv required)
# ---------------------------------------------------------------------------

def _load_dotenv() -> list[str]:
    """
    Load KEY=VALUE pairs from common .env locations into os.environ
    (does not override existing env vars). Returns paths that were read.
    """
    from pathlib import Path

    here = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / ".env",
        here / ".env",
        here.parent / ".env",
        Path.home() / ".env",
    ]
    loaded: list[str] = []
    seen: set[Path] = set()
    for path in candidates:
        try:
            path = path.resolve()
        except Exception:
            continue
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.lower().startswith("export "):
                line = line[7:].strip()
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if not key:
                continue
            # Prefer explicit env; only fill missing
            if key not in os.environ or not os.environ.get(key, "").strip():
                os.environ[key] = val
        loaded.append(str(path))
    return loaded


_DOTENV_LOADED = _load_dotenv()

# ---------------------------------------------------------------------------
# Optional package imports (graceful)
# ---------------------------------------------------------------------------

try:
    from token_optimizer.tokenizers import (
        COST_PRESETS,
        approx_tokens,
        make_anthropic_tokenizer,
        make_gemini_tokenizer,
        make_tiktoken_tokenizer,
        resolve_provider,
    )
    from token_optimizer.costing import (
        XAI_USD_TICKS_PER_DOLLAR,
        compare_cost_accuracy,
        cost_within_tolerance,
        estimate_cost_usd,
        fetch_xai_model_pricing,
        normalize_usage,
        parse_cost_in_usd_ticks,
        ticks_to_usd,
    )
except ImportError:  # running without package layout
    COST_PRESETS = {}
    approx_tokens = lambda t: max(1, (len(t) + 3) // 4) if t else 0  # type: ignore

    def resolve_provider(provider, **kw):  # type: ignore
        return {
            "pin": kw.get("pin") or 1.0,
            "pout": kw.get("pout") or 3.0,
            "pin_cached": kw.get("pin") or 1.0,
            "label": provider or "unknown",
            "model": kw.get("model"),
            "tokenizer_name": "approx",
        }

    def make_tiktoken_tokenizer(encoding="cl100k_base"):  # type: ignore
        raise ImportError("tiktoken not available")

    def make_anthropic_tokenizer(model=""):  # type: ignore
        return approx_tokens

    def make_gemini_tokenizer(model=""):  # type: ignore
        return approx_tokens

    def estimate_cost_usd(usage, *, pin, pout, pin_cached=None):  # type: ignore
        tin = int(usage.get("prompt_tokens") or 0)
        tout = int(usage.get("completion_tokens") or 0)
        return {
            "cost_usd": tin / 1e6 * pin + tout / 1e6 * pout,
            "breakdown": {},
            "tokens": {},
            "rates": {"pin": pin, "pout": pout},
        }

    def fetch_xai_model_pricing(model, **kw):  # type: ignore
        raise RuntimeError("costing module unavailable")

    def normalize_usage(usage):  # type: ignore
        return {}

    def cost_within_tolerance(est, act, min_accuracy=0.995):  # type: ignore
        return {"ok": False, "accuracy": 0.0}


# ---------------------------------------------------------------------------
# Standardized agent-style calibration prompt (realistic multi-turn / tools)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a careful coding agent. Use tools when needed. "
    "Prefer short, actionable answers. If blocked, say what is missing."
)

# Mirrors typical agent context: task + file snippet + tool trace + ask
USER_PROMPT = """# Task
Refactor the checkout total so tax is applied once, and the API field
`items` is renamed to `lines` after the catalog v3 migration.

# Current code (excerpt)
```python
def grand_total(order, tax_rate=0.10):
    sub = 0
    for line in order.items:  # legacy name
        sub += line.price * line.qty
    return int(sub * (1 + tax_rate) * (1 + tax_rate))  # double tax

def checkout(order):
    amount = legacy_cart_total(order)  # rename incomplete
    return {"status": "ok", "total": amount}
```

# Last tool result
TOOL read_file path=services/pricing.py
OK 1842 bytes
(truncated) … def subtotal_items(order): return sum(l.price for l in order.lines)

# Question
List the three concrete bugs, the correct total for
LineItem(10,1)+LineItem(20,2) at 10% tax, and a minimal patch plan.
Reply in under 200 words.
"""

# Max completion size — keeps cost predictable for onboarding
MAX_OUTPUT_TOKENS = 256


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------

PROVIDERS = {
    "openai": {
        "env_keys": ("OPENAI_API_KEY",),
        "default_model": "gpt-4o-mini",
        "preset": "openai-mini",
        "sdk": "openai",
    },
    "anthropic": {
        "env_keys": ("ANTHROPIC_API_KEY",),
        "default_model": "claude-sonnet-4-20250514",
        "preset": "anthropic",
        "sdk": "anthropic",
    },
    "grok": {
        "env_keys": ("XAI_API_KEY", "GROK_API_KEY", "XAI_KEY"),
        "default_model": "grok-4.3",
        "preset": "grok",
        "sdk": "openai",  # OpenAI-compatible client → api.x.ai
        "base_url": "https://api.x.ai/v1",
    },
    "gemini": {
        "env_keys": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "default_model": "gemini-2.0-flash",
        "preset": "gemini",
        "sdk": "gemini",
    },
}


@dataclass
class CalibResult:
    provider: str
    label: str
    model: str
    mode: str  # live | dry-run | fallback
    input_tokens: int
    output_tokens: int
    seconds: float
    pin: float
    pout: float
    cost_usd: float
    tokenizer_note: str
    response_preview: str = ""
    warnings: list[str] = field(default_factory=list)
    raw_usage: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str = ""
    pin_cached: float = 0.0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    api_total_tokens: int = 0
    cost_breakdown: dict[str, Any] = field(default_factory=dict)
    price_source: str = "preset"
    accuracy_check: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.api_total_tokens or (
            self.input_tokens + self.output_tokens + self.reasoning_tokens
        )


def _first_env(names: tuple[str, ...]) -> str | None:
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return None


def _pricing(
    provider_key: str,
    pin: float | None,
    pout: float | None,
    model: str | None,
    *,
    api_key: str | None = None,
    live_prices: bool = True,
) -> dict[str, Any]:
    """Resolve rates; for Grok prefer live models API price sheet."""
    preset_name = PROVIDERS.get(provider_key, {}).get("preset", provider_key)
    resolved = resolve_provider(preset_name, pin=pin, pout=pout, model=model)
    out = {
        "pin": float(resolved["pin"]),
        "pout": float(resolved["pout"]),
        "pin_cached": float(resolved.get("pin_cached") or resolved["pin"]),
        "label": str(resolved.get("label") or provider_key),
        "model": resolved.get("model") or model,
        "price_source": f"preset:{preset_name}",
    }
    if live_prices and provider_key == "grok" and (api_key or _first_env(("XAI_API_KEY", "GROK_API_KEY", "XAI_KEY"))):
        try:
            live = fetch_xai_model_pricing(
                out["model"] or "grok-4.3",
                api_key=api_key or _first_env(("XAI_API_KEY", "GROK_API_KEY", "XAI_KEY")),
            )
            if pin is None:
                out["pin"] = float(live["pin"])
            if pout is None:
                out["pout"] = float(live["pout"])
            out["pin_cached"] = float(live["pin_cached"])
            out["model"] = live["model"]
            out["price_source"] = live["source"]
            out["live_raw"] = live.get("raw")
        except Exception as e:
            out["price_warning"] = f"live price fetch failed: {e}"
    return out


def _local_count(text: str, provider: str) -> tuple[int, str]:
    """Best-effort local count when API usage is unavailable."""
    notes = []
    if provider in ("openai", "grok"):
        for enc in ("o200k_base", "cl100k_base"):
            try:
                fn = make_tiktoken_tokenizer(enc)
                return fn(text), f"tiktoken:{enc}"
            except Exception:
                notes.append(f"tiktoken/{enc} unavailable")
    if provider == "anthropic":
        try:
            fn = make_anthropic_tokenizer()
            n = fn(text)
            return n, "anthropic-or-approx"
        except Exception:
            notes.append("anthropic tokenizer unavailable")
    if provider == "gemini":
        try:
            fn = make_gemini_tokenizer()
            return fn(text), "gemini-or-approx"
        except Exception:
            notes.append("gemini tokenizer unavailable")
    return approx_tokens(text), "approx(~4 chars/tok)" + (
        f" [{'; '.join(notes)}]" if notes else ""
    )


# ---------------------------------------------------------------------------
# Live provider calls
# ---------------------------------------------------------------------------

def run_openai_compatible(
    *,
    api_key: str,
    model: str,
    base_url: str | None = None,
    provider_name: str = "openai",
) -> tuple[str, dict[str, Any], list[str]]:
    """OpenAI SDK — also used for xAI Grok (OpenAI-compatible base_url)."""
    warnings: list[str] = []
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Package 'openai' is not installed. Run: pip install openai"
        ) from e

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.2,
    )
    elapsed = time.perf_counter() - t0

    text = ""
    if resp.choices:
        text = (resp.choices[0].message.content or "").strip()

    usage: dict[str, Any] = {"_elapsed": elapsed}
    u = getattr(resp, "usage", None)
    if u is not None:
        usage["prompt_tokens"] = int(getattr(u, "prompt_tokens", 0) or 0)
        usage["completion_tokens"] = int(getattr(u, "completion_tokens", 0) or 0)
        usage["total_tokens"] = int(getattr(u, "total_tokens", 0) or 0)
        # Dollar truth when present (xAI)
        ticks = parse_cost_in_usd_ticks(u)
        if ticks is not None:
            usage["cost_in_usd_ticks"] = ticks
            usd = ticks_to_usd(ticks)
            if usd is not None:
                usage["cost_usd_api"] = usd
        # Some gateways expose cached / reasoning fields
        for attr in ("prompt_tokens_details", "completion_tokens_details"):
            if hasattr(u, attr):
                det = getattr(u, attr)
                usage[attr] = str(det)
                # Pull reasoning_tokens into a top-level field for reports
                try:
                    rt = int(getattr(det, "reasoning_tokens", 0) or 0)
                    if rt:
                        usage["reasoning_tokens"] = rt
                except Exception:
                    m = re.search(r"reasoning_tokens=(\d+)", str(det))
                    if m:
                        usage["reasoning_tokens"] = int(m.group(1))
                try:
                    ct = int(getattr(det, "cached_tokens", 0) or 0)
                    if ct:
                        usage["cached_tokens"] = ct
                except Exception:
                    m = re.search(r"cached_tokens=(\d+)", str(det))
                    if m:
                        usage["cached_tokens"] = int(m.group(1))
    else:
        warnings.append("API response had no usage object; using local token estimate")
        tin, _ = _local_count(SYSTEM_PROMPT + USER_PROMPT, provider_name)
        tout, _ = _local_count(text, provider_name)
        usage["prompt_tokens"] = tin
        usage["completion_tokens"] = tout
        usage["total_tokens"] = tin + tout

    return text, usage, warnings


def run_anthropic(*, api_key: str, model: str) -> tuple[str, dict[str, Any], list[str]]:
    warnings: list[str] = []
    try:
        import anthropic  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Package 'anthropic' is not installed. Run: pip install anthropic"
        ) from e

    client = anthropic.Anthropic(api_key=api_key)
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_PROMPT}],
        temperature=0.2,
    )
    elapsed = time.perf_counter() - t0

    parts = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    text = "\n".join(parts).strip()

    usage: dict[str, Any] = {"_elapsed": elapsed}
    u = getattr(resp, "usage", None)
    if u is not None:
        usage["input_tokens"] = int(getattr(u, "input_tokens", 0) or 0)
        usage["output_tokens"] = int(getattr(u, "output_tokens", 0) or 0)
        # dashboard aliases
        usage["prompt_tokens"] = usage["input_tokens"]
        usage["completion_tokens"] = usage["output_tokens"]
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    else:
        warnings.append("Anthropic response had no usage; using local estimate")
        tin, _ = _local_count(SYSTEM_PROMPT + USER_PROMPT, "anthropic")
        tout, _ = _local_count(text, "anthropic")
        usage.update(
            prompt_tokens=tin,
            completion_tokens=tout,
            input_tokens=tin,
            output_tokens=tout,
            total_tokens=tin + tout,
        )
    return text, usage, warnings


def run_gemini(*, api_key: str, model: str) -> tuple[str, dict[str, Any], list[str]]:
    warnings: list[str] = []
    text = ""
    usage: dict[str, Any] = {}

    # Prefer new google-genai
    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        contents = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"
        t0 = time.perf_counter()
        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config={
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "temperature": 0.2,
            },
        )
        elapsed = time.perf_counter() - t0
        text = (getattr(resp, "text", None) or str(resp)).strip()
        usage["_elapsed"] = elapsed
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            tin = int(
                getattr(meta, "prompt_token_count", None)
                or getattr(meta, "input_tokens", 0)
                or 0
            )
            tout = int(
                getattr(meta, "candidates_token_count", None)
                or getattr(meta, "output_tokens", 0)
                or 0
            )
            usage["prompt_tokens"] = tin
            usage["completion_tokens"] = tout
            usage["total_tokens"] = tin + tout
        else:
            warnings.append("Gemini response missing usage_metadata; local estimate")
            tin, _ = _local_count(contents, "gemini")
            tout, _ = _local_count(text, "gemini")
            usage.update(
                prompt_tokens=tin, completion_tokens=tout, total_tokens=tin + tout
            )
        return text, usage, warnings
    except ImportError:
        pass
    except Exception as e:
        warnings.append(f"google-genai path failed: {e}")

    # Fallback: google.generativeai
    try:
        import google.generativeai as genai_old  # type: ignore

        genai_old.configure(api_key=api_key)
        m = genai_old.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT,
        )
        t0 = time.perf_counter()
        resp = m.generate_content(
            USER_PROMPT,
            generation_config={
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "temperature": 0.2,
            },
        )
        elapsed = time.perf_counter() - t0
        text = (getattr(resp, "text", None) or "").strip()
        usage["_elapsed"] = elapsed
        meta = getattr(resp, "usage_metadata", None)
        if meta is not None:
            tin = int(getattr(meta, "prompt_token_count", 0) or 0)
            tout = int(getattr(meta, "candidates_token_count", 0) or 0)
            usage["prompt_tokens"] = tin
            usage["completion_tokens"] = tout
            usage["total_tokens"] = tin + tout
        else:
            tin, _ = _local_count(SYSTEM_PROMPT + USER_PROMPT, "gemini")
            tout, _ = _local_count(text, "gemini")
            usage.update(
                prompt_tokens=tin, completion_tokens=tout, total_tokens=tin + tout
            )
            warnings.append("generativeai usage_metadata missing; local estimate")
        return text, usage, warnings
    except ImportError as e:
        raise RuntimeError(
            "Install a Gemini SDK: pip install google-genai  "
            "(or: pip install google-generativeai)"
        ) from e


def dry_run(provider: str, model: str) -> tuple[str, dict[str, Any], list[str]]:
    """No network — estimate tokens for the standard prompt + a stub reply."""
    stub = (
        "Bugs: (1) order.items should be order.lines; "
        "(2) tax applied twice; (3) legacy_cart_total missing. "
        "Correct total: 55. Plan: rename field, single tax factor, call grand_total."
    )
    tin, note_in = _local_count(SYSTEM_PROMPT + "\n" + USER_PROMPT, provider)
    tout, note_out = _local_count(stub, provider)
    usage = {
        "prompt_tokens": tin,
        "completion_tokens": tout,
        "total_tokens": tin + tout,
        "_elapsed": 0.0,
        "_dry_run": True,
        "_tokenizer_in": note_in,
        "_tokenizer_out": note_out,
    }
    return stub, usage, [
        "DRY-RUN: no API call; tokens are local estimates (install SDK + set key for live dashboard numbers)"
    ]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def calibrate(
    provider: str,
    *,
    model: str | None = None,
    dry_run_mode: bool = False,
    pin: float | None = None,
    pout: float | None = None,
    api_key: str | None = None,
    dashboard_tokens_before: int | None = None,
    dashboard_tokens_after: int | None = None,
    dashboard_cost_usd: float | None = None,
    min_accuracy: float = 0.995,
) -> CalibResult:
    provider = provider.strip().lower()
    if provider not in PROVIDERS:
        raise SystemExit(
            f"Unknown provider {provider!r}. Choose: {', '.join(PROVIDERS)}"
        )

    meta = PROVIDERS[provider]
    model = (
        model
        or os.environ.get("CALIBRATE_MODEL", "").strip()
        or meta["default_model"]
    )
    key = (api_key or "").strip() or _first_env(tuple(meta["env_keys"]))
    rates = _pricing(
        provider, pin, pout, model, api_key=key, live_prices=not dry_run_mode
    )
    pin_r = float(rates["pin"])
    pout_r = float(rates["pout"])
    pin_c = float(rates["pin_cached"])
    label = str(rates["label"])
    model = rates.get("model") or model

    warnings: list[str] = []
    if rates.get("price_warning"):
        warnings.append(str(rates["price_warning"]))
    mode = "live"
    text = ""
    usage: dict[str, Any] = {}
    err = ""
    ok = True

    if dry_run_mode:
        mode = "dry-run"
        text, usage, w2 = dry_run(provider, model)
        warnings.extend(w2)
    else:
        if not key:
            mode = "fallback"
            warnings.append(
                f"No API key found for {provider}. Checked: {', '.join(meta['env_keys'])}. "
                "Running local estimate fallback (not a live dashboard reading)."
            )
            text, usage, w2 = dry_run(provider, model)
            warnings.extend(w2)
        else:
            try:
                if provider == "openai":
                    text, usage, w2 = run_openai_compatible(
                        api_key=key, model=model, provider_name="openai"
                    )
                elif provider == "grok":
                    text, usage, w2 = run_openai_compatible(
                        api_key=key,
                        model=model,
                        base_url=meta.get("base_url"),
                        provider_name="grok",
                    )
                elif provider == "anthropic":
                    text, usage, w2 = run_anthropic(api_key=key, model=model)
                elif provider == "gemini":
                    text, usage, w2 = run_gemini(api_key=key, model=model)
                else:
                    raise RuntimeError(f"No runner for {provider}")
                warnings.extend(w2)
            except Exception as e:
                ok = False
                err = f"{type(e).__name__}: {e}"
                mode = "fallback"
                warnings.append(
                    f"Live call failed ({err}). Falling back to local estimate."
                )
                text, usage, w2 = dry_run(provider, model)
                warnings.extend(w2)

    tin = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    tout = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    seconds = float(usage.get("_elapsed") or 0.0)

    est = estimate_cost_usd(usage, pin=pin_r, pout=pout_r, pin_cached=pin_c)
    toks = est.get("tokens") or {}
    reasoning = int(toks.get("reasoning") or usage.get("reasoning_tokens") or 0)
    cached = int(toks.get("cached_prompt") or usage.get("cached_tokens") or 0)
    api_total = int(
        toks.get("total")
        or usage.get("total_tokens")
        or (tin + tout + reasoning)
    )
    if reasoning:
        usage["reasoning_tokens"] = reasoning
        warnings.append(
            f"reasoning_tokens={reasoning} billed at output rate ${pout_r}/1M; "
            f"API total_tokens={api_total}"
        )
    if cached:
        usage["cached_tokens"] = cached
        warnings.append(
            f"cached_tokens={cached} billed at cached input rate ${pin_c}/1M"
        )

    tok_note = "provider-usage"
    if usage.get("_dry_run") or mode != "live":
        tok_note = str(usage.get("_tokenizer_in") or "local-estimate")
    elif "prompt_tokens" not in usage and "input_tokens" not in usage:
        tok_note = "estimated"

    cost = float(est["cost_usd"])
    cost_api = usage.get("cost_usd_api")
    if cost_api is None and usage.get("cost_in_usd_ticks") is not None:
        cost_api = ticks_to_usd(usage.get("cost_in_usd_ticks"))
    cost_cmp = compare_cost_accuracy(
        float(cost_api) if cost_api is not None else None,
        cost,
        min_accuracy=min_accuracy,
        dashboard_cost=dashboard_cost_usd,
    )

    accuracy_check: dict[str, Any] = {"cost": cost_cmp, "est_cost_usd": cost}
    if cost_api is not None:
        accuracy_check["api_cost_usd"] = float(cost_api)
        accuracy_check["cost_in_usd_ticks"] = usage.get("cost_in_usd_ticks")
        accuracy_check["ticks_per_usd"] = XAI_USD_TICKS_PER_DOLLAR
    if dashboard_cost_usd is not None:
        accuracy_check["kind"] = "cost_usd"
        accuracy_check["within_99_5"] = cost_cmp.get("within_99_5")
        accuracy_check["accuracy"] = cost_cmp.get("accuracy")
    elif dashboard_tokens_before is not None and dashboard_tokens_after is not None:
        delta = int(dashboard_tokens_after) - int(dashboard_tokens_before)
        tok_check = cost_within_tolerance(
            float(api_total), float(delta), min_accuracy=min_accuracy
        )
        accuracy_check["kind"] = "token_delta"
        accuracy_check["dashboard_delta"] = delta
        accuracy_check["api_total"] = api_total
        accuracy_check["est_cost_usd"] = cost
        accuracy_check.update(tok_check)
        if cost_api is not None:
            accuracy_check["within_99_5_cost_est"] = cost_cmp.get("within_99_5")

    preview = (text or "").replace("\r", "")
    if len(preview) > 400:
        preview = preview[:400] + "…"

    return CalibResult(
        provider=provider,
        label=label,
        model=model or "",
        mode=mode,
        input_tokens=tin,
        output_tokens=tout,
        seconds=seconds,
        pin=pin_r,
        pout=pout_r,
        cost_usd=cost,
        tokenizer_note=str(tok_note),
        response_preview=preview,
        warnings=warnings,
        raw_usage={k: v for k, v in usage.items() if not str(k).startswith("_")},
        ok=ok,
        error=err,
        pin_cached=pin_c,
        reasoning_tokens=reasoning,
        cached_tokens=cached,
        api_total_tokens=api_total,
        cost_breakdown=dict(est.get("breakdown") or {}),
        price_source=str(rates.get("price_source") or "preset"),
        accuracy_check=accuracy_check,
    )


def format_report(r: CalibResult) -> str:
    """Clean, copyable baseline report for humans and agents."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "=" * 72,
        "TOKEN-OPTIMIZER CALIBRATION BASELINE",
        "=" * 72,
        f"timestamp:        {ts}",
        f"provider:         {r.provider} ({r.label})",
        f"model:            {r.model}",
        f"mode:             {r.mode}",
        f"token_source:     {r.tokenizer_note}",
        f"price_source:     {r.price_source}",
        "-" * 72,
        "DASHBOARD METRICS (copy these)",
        "-" * 72,
        f"input_tokens:     {r.input_tokens}",
        f"cached_input:     {r.cached_tokens}",
        f"uncached_input:   {max(0, r.input_tokens - r.cached_tokens)}",
        f"output_tokens:    {r.output_tokens}",
        f"reasoning_tokens: {r.reasoning_tokens}",
        f"api_total:        {r.api_total_tokens}",
        f"time_seconds:     {r.seconds:.3f}",
        f"price_in_$/M:     {r.pin:.4f}",
        f"price_cached_$/M: {r.pin_cached:.4f}",
        f"price_out_$/M:    {r.pout:.4f}",
        f"est_cost_usd:     ${r.cost_usd:.8f}",
    ]
    ac0 = r.accuracy_check or {}
    if ac0.get("api_cost_usd") is not None:
        lines.append(f"api_cost_usd:     ${float(ac0['api_cost_usd']):.8f}  (from cost_in_usd_ticks)")
        lines.append(f"ticks_per_usd:    {ac0.get('ticks_per_usd', XAI_USD_TICKS_PER_DOLLAR)}")
    if r.cost_breakdown:
        lines.append("cost_breakdown:")
        for k, v in r.cost_breakdown.items():
            lines.append(f"  {k}: ${float(v):.10f}")
    lines += [
        "-" * 72,
        "ONE-LINE SUMMARY",
        "-" * 72,
        (
            f"{r.provider}/{r.model} | in={r.input_tokens} "
            f"(cached={r.cached_tokens}) out={r.output_tokens} "
            f"reason={r.reasoning_tokens} api_total={r.api_total_tokens} | "
            f"{r.seconds:.3f}s | est=${r.cost_usd:.8f}"
            + (
                f" api=${float(ac0['api_cost_usd']):.8f}"
                if ac0.get("api_cost_usd") is not None
                else ""
            )
            + f" | mode={r.mode} | prices={r.price_source}"
        ),
    ]
    if r.accuracy_check:
        ac = r.accuracy_check
        acc = float(ac.get("accuracy") or 0) * 100
        lines += [
            "-" * 72,
            "ACCURACY (target >= 99.5%)",
            "-" * 72,
            f"kind:            {ac.get('kind')}",
            f"estimated:       {ac.get('estimated')}",
            f"actual:          {ac.get('actual')}",
            f"accuracy:        {acc:.4f}%",
            f"target:          {float(ac.get('min_accuracy') or 0.995) * 100:.2f}%",
            f"within_target:   {ac.get('ok')}",
        ]
        cost_c = ac.get("cost") or {}
        if cost_c.get("api_vs_est"):
            ae = cost_c["api_vs_est"]
            lines.append(
                f"api_vs_est:      acc={float(ae.get('accuracy') or 0)*100:.4f}% "
                f"ok={ae.get('ok')} (est={ae.get('estimated')} api={ae.get('actual')})"
            )
    lines += [
        "-" * 72,
        "RESPONSE PREVIEW",
        "-" * 72,
        r.response_preview or "(empty)",
    ]
    if r.raw_usage:
        lines += [
            "-" * 72,
            "RAW USAGE FIELDS",
            "-" * 72,
            json.dumps(r.raw_usage, indent=2, default=str),
        ]
    if r.warnings:
        lines += ["-" * 72, "WARNINGS"]
        for w in r.warnings:
            lines.append(f"  • {w}")
    if r.error:
        lines += ["-" * 72, f"ERROR: {r.error}"]
    lines += [
        "-" * 72,
        "HOW TO LOCK 99.5% ACCURACY",
        "-" * 72,
        "  1. Note console tokens BEFORE run (e.g. 3831).",
        "  2. python calibrate.py -p grok --before 3831 --after <new_total>",
        "  3. Or pass dashboard USD: --dashboard-cost 0.001234",
        "  4. Live Grok rates come from xAI models API (not hardcoded guesses).",
        "  5. Cost = uncached_in*pin + cached_in*pin_cached + (out+reasoning)*pout",
        "=" * 72,
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate token-optimizer against a real frontier provider.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Env: CALIBRATE_PROVIDER, CALIBRATE_MODEL, OPENAI_API_KEY, "
        "ANTHROPIC_API_KEY, XAI_API_KEY, GOOGLE_API_KEY",
    )
    p.add_argument(
        "-p",
        "--provider",
        default=os.environ.get("CALIBRATE_PROVIDER", "").strip() or None,
        choices=sorted(PROVIDERS.keys()),
        help="Provider (or set CALIBRATE_PROVIDER)",
    )
    p.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model id override (or CALIBRATE_MODEL)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Local token estimate only (no network)",
    )
    p.add_argument(
        "--pin",
        type=float,
        default=None,
        help="Override input USD per 1M tokens",
    )
    p.add_argument(
        "--pout",
        type=float,
        default=None,
        help="Override output USD per 1M tokens",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Also print machine-readable JSON after the report",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API key override (else XAI_API_KEY / GROK_API_KEY / OPENAI_…)",
    )
    p.add_argument(
        "--before",
        type=int,
        default=None,
        help="Dashboard token total BEFORE this run (for 99.5%% accuracy check)",
    )
    p.add_argument(
        "--after",
        type=int,
        default=None,
        help="Dashboard token total AFTER this run",
    )
    p.add_argument(
        "--dashboard-cost",
        type=float,
        default=None,
        help="Dashboard USD cost for this run (accuracy check)",
    )
    p.add_argument(
        "--min-accuracy",
        type=float,
        default=0.995,
        help="Required accuracy vs dashboard (default 0.995 = 99.5%%)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if _DOTENV_LOADED:
        # Paths only — never print secret values
        print("loaded .env from: " + ", ".join(_DOTENV_LOADED))
    provider = args.provider
    if not provider:
        # Default to grok when only xAI key is present
        if _first_env(("XAI_API_KEY", "GROK_API_KEY", "XAI_KEY")):
            provider = "grok"
            print("no --provider set; defaulting to grok (found xAI key in env)")
        else:
            print(
                "ERROR: pass --provider openai|anthropic|grok|gemini "
                "or set CALIBRATE_PROVIDER.\n"
                "See the docstring at the top of calibrate.py for API key setup.",
                file=sys.stderr,
            )
            return 2

    try:
        result = calibrate(
            provider,
            model=args.model,
            dry_run_mode=args.dry_run,
            pin=args.pin,
            pout=args.pout,
            api_key=args.api_key,
            dashboard_tokens_before=args.before,
            dashboard_tokens_after=args.after,
            dashboard_cost_usd=args.dashboard_cost,
            min_accuracy=args.min_accuracy,
        )
    except Exception as e:
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print(format_report(result))
    if args.json:
        payload = {
            "provider": result.provider,
            "label": result.label,
            "model": result.model,
            "mode": result.mode,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
            "time_seconds": result.seconds,
            "est_cost_usd": result.cost_usd,
            "pin": result.pin,
            "pout": result.pout,
            "tokenizer_note": result.tokenizer_note,
            "warnings": result.warnings,
            "error": result.error,
            "raw_usage": result.raw_usage,
        }
        print("\n--- JSON ---")
        print(json.dumps(payload, indent=2))

    # exit 0 for successful live or intentional dry-run; 1 if forced fallback error
    if result.mode == "fallback" and result.error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Provider token counting + dashboard cost presets.

Default counting is fast character-approx. Optional backends:
  - OpenAI tiktoken (cl100k_base / o200k_base)
  - Anthropic SDK count_tokens when installed
  - Gemini count via google-genai / generativeai when installed
  - Any callable registered with register_tokenizer()

Cost presets align with common product dashboards (USD / 1M tokens).
Override pin/pout anytime — providers change prices.
"""
from __future__ import annotations

from typing import Callable, Dict, Any

TokenizerFn = Callable[[str], int]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, TokenizerFn] = {}


def approx_tokens(text: str) -> int:
    """Fast default: ~4 characters per token."""
    return max(1, (len(text) + 3) // 4) if text else 0


def register_tokenizer(name: str, fn: TokenizerFn) -> None:
    """Register a custom tokenizer: fn(text) -> token_count (int)."""
    if not callable(fn):
        raise TypeError("tokenizer must be callable(text) -> int")
    _REGISTRY[name.strip().lower()] = fn


def list_tokenizers() -> list[str]:
    return sorted(set(_REGISTRY) | {"approx", "tiktoken", "anthropic", "gemini", "cl100k_base", "o200k_base"})


# ---------------------------------------------------------------------------
# Built-in backends (lazy; optional deps)
# ---------------------------------------------------------------------------

_tiktoken_cache: dict[str, TokenizerFn] = {}


def make_tiktoken_tokenizer(encoding: str = "cl100k_base") -> TokenizerFn:
    """OpenAI tiktoken encoder wrapper (raises ImportError if missing)."""
    key = encoding or "cl100k_base"
    if key in _tiktoken_cache:
        return _tiktoken_cache[key]
    import tiktoken  # type: ignore

    enc = tiktoken.get_encoding(key)

    def count(text: str) -> int:
        if not text:
            return 0
        return len(enc.encode(text))

    _tiktoken_cache[key] = count
    return count


def make_anthropic_tokenizer(model: str = "claude-sonnet-4-20250514") -> TokenizerFn:
    """
    Anthropic token counting via SDK when available.
    Falls back to approx_tokens if the SDK/API is unavailable.
    """
    model_name = model

    def count(text: str) -> int:
        if not text:
            return 0
        try:
            import anthropic  # type: ignore

            client = anthropic.Anthropic()
            # Prefer messages.count_tokens (newer SDK)
            if hasattr(client, "messages") and hasattr(client.messages, "count_tokens"):
                res = client.messages.count_tokens(
                    model=model_name,
                    messages=[{"role": "user", "content": text}],
                )
                n = getattr(res, "input_tokens", None) or getattr(res, "token_count", None)
                if n is not None:
                    return int(n)
            # Older helper
            if hasattr(client, "count_tokens"):
                return int(client.count_tokens(text))
        except Exception:
            pass
        # Published Claude approx often ~3.5–4 chars/token for English
        return approx_tokens(text)

    return count


def make_gemini_tokenizer(model: str = "gemini-2.0-flash") -> TokenizerFn:
    """Gemini count_tokens when google-genai / generativeai is installed."""
    model_name = model

    def count(text: str) -> int:
        if not text:
            return 0
        try:
            from google import genai  # type: ignore

            client = genai.Client()
            res = client.models.count_tokens(model=model_name, contents=text)
            n = getattr(res, "total_tokens", None)
            if n is not None:
                return int(n)
        except Exception:
            pass
        try:
            import google.generativeai as genai_old  # type: ignore

            m = genai_old.GenerativeModel(model_name)
            res = m.count_tokens(text)
            n = getattr(res, "total_tokens", None)
            if n is not None:
                return int(n)
        except Exception:
            pass
        return approx_tokens(text)

    return count


def resolve_tokenizer(
    tokenizer: str | TokenizerFn | None = None,
    *,
    encoding: str | None = None,
    model: str | None = None,
) -> tuple[TokenizerFn, str]:
    """
    Resolve a tokenizer callable and a short name for stats.

    tokenizer: None/'approx' | 'tiktoken' | 'cl100k_base' | 'anthropic' |
               'gemini' | registered name | custom callable
    """
    if callable(tokenizer) and not isinstance(tokenizer, str):
        return tokenizer, "custom"

    name = (tokenizer or "approx").strip().lower()
    if name in _REGISTRY:
        return _REGISTRY[name], name

    if name in ("approx", "char", "fast", "default"):
        return approx_tokens, "approx"

    if name in ("tiktoken", "openai", "cl100k_base", "o200k_base"):
        enc = encoding or ("o200k_base" if name == "o200k_base" else "cl100k_base")
        if name in ("cl100k_base", "o200k_base"):
            enc = name
        try:
            return make_tiktoken_tokenizer(enc), f"tiktoken:{enc}"
        except ImportError:
            return approx_tokens, "approx(no-tiktoken)"

    if name in ("anthropic", "claude"):
        try:
            return make_anthropic_tokenizer(model or "claude-sonnet-4-20250514"), "anthropic"
        except Exception:
            return approx_tokens, "approx(no-anthropic)"

    if name in ("gemini", "google"):
        return make_gemini_tokenizer(model or "gemini-2.0-flash"), "gemini"

    # unknown string → approx (safe default)
    return approx_tokens, "approx"


# ---------------------------------------------------------------------------
# Dashboard cost presets (USD per 1M tokens) — match common billing UIs
# pin = input, pout = output. Update as vendors change price sheets.
# ---------------------------------------------------------------------------

COST_PRESETS: Dict[str, Dict[str, Any]] = {
    # xAI Grok — rates from models API (usd/1M = field/10000). Prefer live fetch in calibrate.
    # grok-4.3 short context: $1.25 in / $0.20 cached / $2.50 out (reasoning billed as out)
    # Grok tiers — live sheet: usd/1M = models API field/10000
    "grok": {
        "label": "xAI Grok 4.3 (production)",
        "pin": 1.25,
        "pout": 2.50,
        "pin_cached": 0.20,
        "tokenizer": "approx",
        "model": "grok-4.3",
        "tier": "standard",
    },
    "xai": {
        "label": "xAI Grok 4.3 (production)",
        "pin": 1.25,
        "pout": 2.50,
        "pin_cached": 0.20,
        "tokenizer": "approx",
        "model": "grok-4.3",
        "tier": "standard",
    },
    "grok-4.3": {
        "label": "xAI Grok 4.3 (production)",
        "pin": 1.25,
        "pout": 2.50,
        "pin_cached": 0.20,
        "tokenizer": "approx",
        "model": "grok-4.3",
        "tier": "standard",
    },
    "grok-4.5": {
        "label": "xAI Grok 4.5 (frontier)",
        "pin": 2.00,
        "pout": 6.00,
        "pin_cached": 0.50,
        "tokenizer": "approx",
        "model": "grok-4.5",
        "tier": "frontier",
    },
    "grok-frontier": {
        "label": "xAI Grok 4.5 (frontier)",
        "pin": 2.00,
        "pout": 6.00,
        "pin_cached": 0.50,
        "tokenizer": "approx",
        "model": "grok-4.5",
        "tier": "frontier",
    },
    "grok-build": {
        "label": "xAI Grok Build 0.1 (bulk)",
        "pin": 1.00,
        "pout": 2.00,
        "pin_cached": 0.20,
        "tokenizer": "approx",
        "model": "grok-build-0.1",
        "tier": "fast",
    },
    "grok-fast": {
        "label": "xAI Grok non-reasoning bulk",
        "pin": 1.25,
        "pout": 2.50,
        "pin_cached": 0.20,
        "tokenizer": "approx",
        "model": "grok-4.20-0309-non-reasoning",
        "tier": "fast",
    },
    # OpenAI
    "openai": {
        "label": "OpenAI GPT-4o",
        "pin": 2.50,
        "pout": 10.00,
        "tokenizer": "tiktoken",
        "encoding": "o200k_base",
        "model": "gpt-4o",
    },
    "openai-mini": {
        "label": "OpenAI GPT-4o-mini",
        "pin": 0.15,
        "pout": 0.60,
        "tokenizer": "tiktoken",
        "encoding": "o200k_base",
        "model": "gpt-4o-mini",
    },
    "openai-4o": {
        "label": "OpenAI GPT-4o",
        "pin": 2.50,
        "pout": 10.00,
        "tokenizer": "tiktoken",
        "encoding": "o200k_base",
        "model": "gpt-4o",
    },
    # Anthropic
    "anthropic": {
        "label": "Anthropic Claude Sonnet",
        "pin": 3.00,
        "pout": 15.00,
        "tokenizer": "anthropic",
        "model": "claude-sonnet-4-20250514",
    },
    "claude": {
        "label": "Anthropic Claude Sonnet",
        "pin": 3.00,
        "pout": 15.00,
        "tokenizer": "anthropic",
        "model": "claude-sonnet-4-20250514",
    },
    "anthropic-haiku": {
        "label": "Anthropic Claude Haiku",
        "pin": 0.80,
        "pout": 4.00,
        "tokenizer": "anthropic",
        "model": "claude-haiku-4-20250414",
    },
    # Google Gemini
    "gemini": {
        "label": "Google Gemini Flash",
        "pin": 0.10,
        "pout": 0.40,
        "tokenizer": "gemini",
        "model": "gemini-2.0-flash",
    },
    "gemini-pro": {
        "label": "Google Gemini Pro",
        "pin": 1.25,
        "pout": 5.00,
        "tokenizer": "gemini",
        "model": "gemini-1.5-pro",
    },
    # Cursor IDE agent dashboard (blended multi-model usage estimate)
    "cursor": {
        "label": "Cursor Agent (blended)",
        "pin": 2.00,
        "pout": 8.00,
        "tokenizer": "tiktoken",
        "encoding": "cl100k_base",
        "model": "cursor-agent",
    },
    # Spark — Cursor/product "Spark" tier or similar agent SKU in dashboards
    "spark": {
        "label": "Spark Agent",
        "pin": 1.00,
        "pout": 4.00,
        "tokenizer": "tiktoken",
        "encoding": "cl100k_base",
        "model": "spark",
    },
    # Explicit fast approx (no vendor SDK)
    "approx": {
        "label": "Approx (~4 chars/tok)",
        "pin": 0.20,
        "pout": 0.50,
        "tokenizer": "approx",
    },
}


def resolve_provider(
    provider: str | None = None,
    *,
    pin: float | None = None,
    pout: float | None = None,
    tokenizer: str | TokenizerFn | None = None,
    encoding: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Merge a named dashboard preset with explicit overrides.
    Returns pin, pout, tokenizer fn, tokenizer_name, provider label.
    """
    key = (provider or "approx").strip().lower()
    preset = dict(COST_PRESETS.get(key, COST_PRESETS["approx"]))
    if provider and key not in COST_PRESETS and key not in ("approx",):
        # unknown provider name still usable as label
        preset["label"] = provider

    use_pin = preset.get("pin", 0.20) if pin is None else pin
    use_pout = preset.get("pout", 0.50) if pout is None else pout
    pin_cached = preset.get("pin_cached")
    tok_spec = tokenizer if tokenizer is not None else preset.get("tokenizer", "approx")
    enc = encoding or preset.get("encoding")
    mod = model or preset.get("model")

    fn, tok_name = resolve_tokenizer(tok_spec, encoding=enc, model=mod)
    return {
        "pin": float(use_pin),
        "pout": float(use_pout),
        "pin_cached": float(pin_cached) if pin_cached is not None else float(use_pin),
        "tokenizer": fn,
        "tokenizer_name": tok_name,
        "provider": key,
        "label": preset.get("label", key),
        "model": mod,
    }


# Register built-in aliases so register_tokenizer consumers see them
register_tokenizer("approx", approx_tokens)

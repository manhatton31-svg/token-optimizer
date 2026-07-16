"""
Drop-in Grok client for heavy builders (customer-ready).

Replace:
    client.chat.completions.create(messages=[...])

With:
    session = GrokSession(profile="production")  # frontier | innovate | production | bulk
    resp = session.chat("fix the tax bug", history=hist)
    # per-call: session.chat(..., model="grok-4.5", effort="high")

Always:
  - compacts context (90%+ savings vs naive full dumps)
  - applies profile reasoning / model
  - observes real API usage + cost_in_usd_ticks for $ truth
  - warns on fat prompts (does not freeze soft mode)

Heavy-user stack
----------------
  frontier   — grok-4.5 design / hard problems
  innovate   — grok-4.3 full reasoning + max compression
  production — grok-4.3 volume (default)
  bulk       — non-reasoning max savings after the idea works
"""
from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from .core import TokenOptimizer, tool_compact
from .costing import (
    XAI_USD_TICKS_PER_DOLLAR,
    compare_cost_accuracy,
    estimate_cost_usd,
    normalize_usage,
    parse_cost_in_usd_ticks,
    ticks_to_usd,
)
from .models import get_model
from .onboarding import TARGET_ACCURACY, verify_token_delta

BASE_URL = "https://api.x.ai/v1"

# Profiles: quality vs scale. Savings always from compact context.
PROFILES: dict[str, dict[str, Any]] = {
    "frontier": {
        "model": "grok-4.5",
        "reasoning_mode": "innovate",
        "system": "Frontier expert. Think carefully. Precise and brief.",
        "err_tail": 64,
        "ctx_max": 140,
        "hist_keep": 2,
        "max_tokens": 500_000,
        "soft": True,
        "warn_chars": 2_000,
        "notes": "Design / hard debug on grok-4.5. Aggressive compact; full reasoning.",
    },
    "innovate": {
        "model": "grok-4.3",
        "reasoning_mode": "innovate",
        "system": "Expert agent. Think carefully. Be precise and brief.",
        "err_tail": 64,
        "ctx_max": 160,
        "hist_keep": 2,
        "max_tokens": 500_000,
        "soft": True,
        "warn_chars": 2_500,
        "notes": "Full reasoning + max compression on 4.3. Invent / hard debug at lower $.",
    },
    "production": {
        "model": "grok-4.3",
        "reasoning_mode": "balanced",
        "system": "Agent. Be brief and correct.",
        "err_tail": 48,
        "ctx_max": 120,
        "hist_keep": 1,
        "max_tokens": 500_000,
        "soft": True,
        "warn_chars": 2_000,
        "notes": "Calibrated production path. Default for customer volume.",
    },
    "bulk": {
        "model": "grok-4.20-0309-non-reasoning",
        "reasoning_mode": "off",
        "system": "fix. short.",
        "err_tail": 40,
        "ctx_max": 100,
        "hist_keep": 1,
        "max_tokens": 500_000,
        "soft": True,
        "warn_chars": 1_500,
        "notes": "Max savings; no reasoning tax. Use after the idea works.",
    },
}

# Per-call effort override → reasoning_mode (None keeps profile default)
_EFFORT_TO_MODE: dict[str, str] = {
    "high": "high",
    "medium": "balanced",
    "low": "low",
    "off": "off",
    "none": "off",
}
_VALID_EFFORTS = frozenset({"high", "medium", "low", "off", "none"})


def _load_dotenv() -> None:
    from pathlib import Path

    for candidate in (Path.cwd() / ".env", Path.home() / ".env"):
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.lower().startswith("export "):
                line = line[7:].strip()
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip("'").strip('"')
            if k and not os.environ.get(k):
                os.environ[k] = v


def _api_key(explicit: str | None = None) -> str:
    _load_dotenv()
    key = (
        (explicit or "").strip()
        or os.environ.get("XAI_API_KEY", "").strip()
        or os.environ.get("GROK_API_KEY", "").strip()
        or os.environ.get("XAI_KEY", "").strip()
    )
    if not key:
        raise RuntimeError(
            "Grok API key missing. Set XAI_API_KEY in .env or pass api_key=..."
        )
    return key


def warn_fat_prompt(
    text: str,
    *,
    warn_chars: int = 2_000,
    label: str = "prompt",
) -> list[str]:
    """
    Advisory warnings for builders who accidentally ship monorepos.
    Never raises — complexity-proof for DIY loops.
    """
    msgs: list[str] = []
    n = len(text or "")
    if n >= warn_chars:
        msgs.append(
            f"[token-optimizer] fat {label}: {n} chars "
            f"(threshold {warn_chars}). Compact or trim before send."
        )
    # Heuristics: looks like full file dumps
    if text.count("\n") > 80 and (
        "def " in text or "class " in text or "COMPLETE SOURCE" in text
    ):
        msgs.append(
            f"[token-optimizer] {label} looks like a full source dump "
            f"({text.count(chr(10))} lines). Prefer error tails + paths."
        )
    for m in msgs:
        warnings.warn(m, UserWarning, stacklevel=2)
    return msgs


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    """Normalize usage; int token fields + optional cost_in_usd_ticks / cost_usd_api."""
    if usage is None:
        return {}
    if isinstance(usage, dict):
        d = dict(usage)
    else:
        d = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        pdet = getattr(usage, "prompt_tokens_details", None)
        cdet = getattr(usage, "completion_tokens_details", None)
        if pdet is not None:
            try:
                d["cached_tokens"] = int(getattr(pdet, "cached_tokens", 0) or 0)
            except Exception:
                m = re.search(r"cached_tokens=(\d+)", str(pdet))
                if m:
                    d["cached_tokens"] = int(m.group(1))
        if cdet is not None:
            try:
                d["reasoning_tokens"] = int(getattr(cdet, "reasoning_tokens", 0) or 0)
            except Exception:
                m = re.search(r"reasoning_tokens=(\d+)", str(cdet))
                if m:
                    d["reasoning_tokens"] = int(m.group(1))
        ticks = parse_cost_in_usd_ticks(usage)
        if ticks is not None:
            d["cost_in_usd_ticks"] = ticks

    out: dict[str, Any] = {}
    for k, v in d.items():
        if k in ("cost_in_usd_ticks", "cost_usd_api", "cost_usd_est"):
            try:
                out[k] = int(v) if k == "cost_in_usd_ticks" else float(v)
            except (TypeError, ValueError):
                continue
            continue
        try:
            out[k] = int(v or 0)
        except (TypeError, ValueError):
            continue

    # Prefer explicit ticks on object even if dict path missed it
    if "cost_in_usd_ticks" not in out:
        ticks = parse_cost_in_usd_ticks(usage)
        if ticks is not None:
            out["cost_in_usd_ticks"] = ticks
    if "cost_in_usd_ticks" in out:
        usd = ticks_to_usd(out["cost_in_usd_ticks"])
        if usd is not None:
            out["cost_usd_api"] = usd
    return out


@dataclass
class ChatResult:
    """Normalized result from GrokSession.chat / .chat_messages / chat_stream."""

    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    model: str = ""
    profile: str = ""
    warnings: list[str] = field(default_factory=list)
    raw: Any = None
    stopped: bool = False
    reason: str = ""
    cost_usd_api: float | None = None
    cost_usd_est: float | None = None

    @property
    def api_total(self) -> int:
        return int(
            self.usage.get("total_tokens")
            or (
                int(self.usage.get("prompt_tokens", 0) or 0)
                + int(self.usage.get("completion_tokens", 0) or 0)
                + int(self.usage.get("reasoning_tokens", 0) or 0)
            )
        )


class GrokSession:
    """
    Drop-in Grok path for builders and customers.

    Profiles
    --------
    frontier   — grok-4.5, full reasoning, aggressive compact (design)
    innovate   — grok-4.3, full reasoning, max compression
    production — grok-4.3, balanced (default volume)
    bulk       — non-reasoning model, max savings after the idea works

    Example
    -------
    session = GrokSession(profile="production")
    hist = []
    r = session.chat("Fix tax double-apply", history=hist)
    # design turn without leaving the session:
    r2 = session.chat("Architect the fix", model="grok-4.5", effort="high")
    print(r.text, r.api_total)
    session.calibrate()  # lock $ to this key's live ticks
    session.verify_dashboard(before=251203, after=251900)
    """

    def __init__(
        self,
        profile: str = "production",
        *,
        model: str | None = None,
        system: str | None = None,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        soft: bool | None = None,
        max_tokens_out: int = 256,
        temperature: float = 0.2,
        live_prices: bool = True,
        **optimizer_kwargs: Any,
    ):
        name = (profile or "production").strip().lower()
        if name not in PROFILES:
            raise ValueError(
                f"Unknown profile {profile!r}. Choose: {', '.join(PROFILES)}"
            )
        cfg = dict(PROFILES[name])
        self.profile = name
        self.max_tokens_out = max_tokens_out
        self.temperature = temperature
        self.warn_chars = int(cfg.get("warn_chars") or 2000)
        self.soft = cfg["soft"] if soft is None else soft
        self._api_key = api_key
        self.base_url = base_url
        self._client = None
        self._live_prices = live_prices

        model_id = model or cfg["model"]
        sys_msg = system if system is not None else cfg["system"]

        opt_kw = {
            "system": sys_msg,
            "model": model_id,
            "provider": "grok",
            "reasoning_mode": cfg["reasoning_mode"],
            "err_tail": cfg["err_tail"],
            "ctx_max": cfg["ctx_max"],
            "hist_keep": cfg["hist_keep"],
            "max_tokens": cfg["max_tokens"],
            "max_retries": 50,
        }
        opt_kw.update(optimizer_kwargs)
        self.opt = TokenOptimizer(**opt_kw)
        if name in ("innovate", "frontier"):
            self.opt.innovator_profile()
            self.opt.reasoning_mode = "innovate"
            if name == "frontier":
                # tighter than innovate catalog defaults
                self.opt.ctx_max = min(self.opt.ctx_max, int(cfg["ctx_max"]))
                self.opt.err_tail = min(self.opt.err_tail, int(cfg["err_tail"]))

        # Apply catalog / live rates
        self._apply_model_rates(model_id, live=live_prices, api_key=api_key)

        self.opt.charge_system()
        self.history: list[str] = []
        self.session_api_total = 0
        self.session_calls = 0
        self.session_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }
        # Dollar meters: API ticks (truth) vs sheet estimate
        self.session_cost_usd_api: float = 0.0
        self.session_cost_usd_est: float = 0.0
        self.session_cost_ticks: int = 0
        self.ticks_per_usd: int = XAI_USD_TICKS_PER_DOLLAR
        self.last_warnings: list[str] = []
        self.last_result: ChatResult | None = None
        self._events: list[dict[str, Any]] = []
        self.last_calibration: dict[str, Any] | None = None

    def _apply_model_rates(
        self,
        model_id: str,
        *,
        live: bool | None = None,
        api_key: str | None = None,
    ) -> None:
        """Set model id + catalog rates; optionally refresh from xAI models API."""
        use_live = self._live_prices if live is None else live
        spec = get_model(model_id)
        if spec:
            self.opt.use_model(spec.id)
        else:
            self.opt.model = model_id
        if use_live:
            try:
                from .costing import fetch_xai_model_pricing

                live_row = fetch_xai_model_pricing(
                    self.opt.model or model_id,
                    api_key=_api_key(api_key if api_key is not None else self._api_key),
                )
                self.opt.pin = float(live_row["pin"])
                self.opt.pout = float(live_row["pout"])
                self.opt.pin_cached = float(live_row["pin_cached"])
                self.opt.model = live_row["model"]
                self.opt.provider_label = f"{live_row['model']} (live sheet)"
            except Exception:
                pass  # keep catalog rates

    # --- client ---
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as e:
                raise RuntimeError("pip install openai") from e
            self._client = OpenAI(api_key=_api_key(self._api_key), base_url=self.base_url)
        return self._client

    def use_profile(self, profile: str) -> None:
        """Switch profile mid-session (keeps history + cumulative usage)."""
        cfg = PROFILES.get((profile or "").lower())
        if not cfg:
            raise ValueError(f"Unknown profile {profile!r}. Choose: {', '.join(PROFILES)}")
        self.profile = profile.lower()
        self.opt.set_reasoning_mode(cfg["reasoning_mode"])
        self.opt.err_tail = cfg["err_tail"]
        self.opt.ctx_max = cfg["ctx_max"]
        self.opt.hist_keep = cfg["hist_keep"]
        self.warn_chars = int(cfg["warn_chars"])
        self._apply_model_rates(cfg["model"], live=False)
        if self.profile in ("innovate", "frontier"):
            self.opt.innovator_profile()
            self.opt.reasoning_mode = "innovate"
            self.opt.ctx_max = min(self.opt.ctx_max, int(cfg["ctx_max"]))
            self.opt.err_tail = min(self.opt.err_tail, int(cfg["err_tail"]))
        if self._live_prices:
            self._apply_model_rates(self.opt.model or cfg["model"], live=True)

    def _push_call_overrides(
        self,
        *,
        model: str | None = None,
        effort: str | None = None,
    ) -> dict[str, Any]:
        """
        Temporarily apply per-call model / effort. Returns snapshot to restore.
        effort: high | medium | low | off | None (None = keep profile mode)
        """
        snap = {
            "model": self.opt.model,
            "pin": self.opt.pin,
            "pout": self.opt.pout,
            "pin_cached": getattr(self.opt, "pin_cached", self.opt.pin),
            "provider_label": getattr(self.opt, "provider_label", None),
            "reasoning_mode": self.opt.reasoning_mode,
        }
        if model:
            # Catalog rates for override (avoid live fetch on every turn);
            # live sheet still applied if model matches current live label.
            self._apply_model_rates(model, live=False)
            if self._live_prices and model == snap["model"]:
                self._apply_model_rates(model, live=True)
            elif self._live_prices:
                # Best-effort live rates for the override model
                self._apply_model_rates(model, live=True)
        if effort is not None:
            e = str(effort).strip().lower()
            if e not in _VALID_EFFORTS:
                raise ValueError(
                    f"effort must be one of {sorted(_VALID_EFFORTS)} or None; got {effort!r}"
                )
            self.opt.set_reasoning_mode(_EFFORT_TO_MODE[e])
        return snap

    def _pop_call_overrides(self, snap: dict[str, Any]) -> None:
        self.opt.model = snap["model"]
        self.opt.pin = snap["pin"]
        self.opt.pout = snap["pout"]
        self.opt.pin_cached = snap["pin_cached"]
        if snap.get("provider_label") is not None:
            self.opt.provider_label = snap["provider_label"]
        self.opt.reasoning_mode = snap["reasoning_mode"]
    # --- main entrypoints ---
    def _prepare_chat_turn(
        self,
        user: str,
        *,
        history: list[str] | None = None,
        tool_result: str | None = None,
        system: str | None = None,
        bill_prepare: bool = False,
    ) -> tuple[list[str], dict[str, Any], ChatResult | None]:
        """
        Shared compact + warn path for chat / chat_stream.
        Returns (hist, turn, early_result_or_None).
        """
        hist = self.history if history is None else history
        fat = warn_fat_prompt(user, warn_chars=self.warn_chars, label="user")
        tool_for_model = tool_result
        if tool_result:
            # Warn on ORIGINAL fat payload so builders learn, then compact for send
            fat += warn_fat_prompt(
                tool_result, warn_chars=self.warn_chars, label="tool_result"
            )
            tool_cap = max(120, min(800, int(self.opt.ctx_max) * 3))
            tool_for_model = tool_compact(tool_result, max_chars=tool_cap)
            if len(str(tool_result)) > len(str(tool_for_model)):
                fat.append(
                    f"[token-optimizer] tool_result compacted "
                    f"{len(str(tool_result))}→{len(str(tool_for_model))} chars"
                )
        self.last_warnings = fat

        turn = self.opt.prepare_turn(
            user,
            history=hist,
            tool_result=tool_for_model,
            system=system,
            bill=bill_prepare,
            soft=self.soft,
        )
        fat2 = warn_fat_prompt(
            turn["user"], warn_chars=self.warn_chars, label="compact_user"
        )
        self.last_warnings.extend(fat2)

        if turn["stopped"] and not self.soft:
            early = ChatResult(
                text="",
                stopped=True,
                reason=turn.get("reason") or "budget",
                profile=self.profile,
                model=str(self.opt.model or ""),
                warnings=list(self.last_warnings),
            )
            self.last_result = early
            return hist, turn, early
        return hist, turn, None

    def _record_usage_and_finish(
        self,
        text: str,
        u: dict[str, Any],
        *,
        hist: list[str],
        turn: dict[str, Any],
        history: list[str] | None,
        seconds: float,
        raw: Any = None,
        extra_warnings: list[str] | None = None,
    ) -> ChatResult:
        """Shared post-response bookkeeping for chat / chat_stream."""
        warns = list(self.last_warnings)
        if extra_warnings:
            warns.extend(extra_warnings)

        # Token meters (do not pass cost fields into observe as tokens)
        token_u = {
            k: int(u.get(k, 0) or 0)
            for k in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cached_tokens",
                "reasoning_tokens",
            )
        }
        if u:
            self.opt.observe_api_usage(token_u)
        self.opt.finish_turn(
            text,
            history=hist,
            user_blob=turn.get("blob") or "",
            emit_status=False,
        )
        if history is None:
            self.history = hist

        if not token_u.get("total_tokens"):
            token_u["total_tokens"] = (
                int(token_u.get("prompt_tokens", 0) or 0)
                + int(token_u.get("completion_tokens", 0) or 0)
                + int(token_u.get("reasoning_tokens", 0) or 0)
            )
            u["total_tokens"] = token_u["total_tokens"]

        for k in self.session_usage:
            self.session_usage[k] = self.session_usage.get(k, 0) + int(
                token_u.get(k, 0) or 0
            )
        self.session_api_total += int(token_u.get("total_tokens") or 0)
        self.session_calls += 1

        # Dollar meters: API ticks (truth) + sheet estimate
        est = estimate_cost_usd(
            token_u,
            pin=self.opt.pin,
            pout=self.opt.pout,
            pin_cached=getattr(self.opt, "pin_cached", self.opt.pin),
        )
        cost_est = float(est["cost_usd"])
        self.session_cost_usd_est += cost_est

        ticks = u.get("cost_in_usd_ticks")
        if ticks is None:
            ticks = parse_cost_in_usd_ticks(raw)
        cost_api = None
        if ticks is not None:
            try:
                ticks_i = int(ticks)
                self.session_cost_ticks += ticks_i
                cost_api = ticks_to_usd(ticks_i, scale=self.ticks_per_usd)
                if cost_api is not None:
                    self.session_cost_usd_api += cost_api
                    u["cost_in_usd_ticks"] = ticks_i
                    u["cost_usd_api"] = cost_api
            except (TypeError, ValueError):
                pass
        u["cost_usd_est"] = cost_est

        result = ChatResult(
            text=text,
            usage=u,
            seconds=seconds,
            model=str(self.opt.model or ""),
            profile=self.profile,
            warnings=warns,
            raw=raw,
            stopped=bool(turn.get("stopped")),
            reason=str(turn.get("reason") or ""),
            cost_usd_api=cost_api,
            cost_usd_est=cost_est,
        )
        self.last_result = result
        self._events.append(
            {
                "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "profile": self.profile,
                "model": result.model,
                "seconds": seconds,
                "usage": {
                    k: int(token_u.get(k, 0) or 0)
                    for k in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "cached_tokens",
                        "reasoning_tokens",
                    )
                },
                "cost_in_usd_ticks": u.get("cost_in_usd_ticks"),
                "cost_usd_api": cost_api,
                "cost_usd_est": cost_est,
                "rates": {
                    "pin": self.opt.pin,
                    "pout": self.opt.pout,
                    "pin_cached": getattr(self.opt, "pin_cached", self.opt.pin),
                },
                "stopped": result.stopped,
                "reason": result.reason,
            }
        )
        return result

    def chat(
        self,
        user: str,
        *,
        history: list[str] | None = None,
        tool_result: str | None = None,
        system: str | None = None,
        bill_prepare: bool = False,
        model: str | None = None,
        effort: str | None = None,
    ) -> ChatResult:
        """
        One compacted, metered Grok turn.

        history: optional list you own; defaults to session.history
        bill_prepare: if True, also meter compacted blob via add_in
                      (usually False when observe_api_usage is used)
        model: optional per-call model override (e.g. grok-4.5)
        effort: optional per-call reasoning effort: high | medium | low | off
        """
        snap = self._push_call_overrides(model=model, effort=effort)
        try:
            hist, turn, early = self._prepare_chat_turn(
                user,
                history=history,
                tool_result=tool_result,
                system=system,
                bill_prepare=bill_prepare,
            )
            if early is not None:
                return early

            import time

            client = self._get_client()
            kwargs: dict[str, Any] = {
                "model": self.opt.model,
                "messages": turn["messages"],
                "max_tokens": self.max_tokens_out,
                "temperature": self.temperature,
            }
            api_extra = turn.get("api_kwargs") or self.opt.api_kwargs()
            kwargs.update(api_extra)

            t0 = time.perf_counter()
            resp = client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - t0

            text = ""
            if resp.choices:
                text = (resp.choices[0].message.content or "").strip()

            u = _usage_to_dict(getattr(resp, "usage", None))
            return self._record_usage_and_finish(
                text,
                u,
                hist=hist,
                turn=turn,
                history=history,
                seconds=elapsed,
                raw=resp,
            )
        finally:
            self._pop_call_overrides(snap)

    def chat_stream(
        self,
        user: str,
        *,
        history: list[str] | None = None,
        tool_result: str | None = None,
        system: str | None = None,
        bill_prepare: bool = False,
        model: str | None = None,
        effort: str | None = None,
    ) -> Iterator[str]:
        """
        Streaming compacted Grok turn.

        Yields text deltas. After exhaustion, ``session.last_result`` is a full
        :class:`ChatResult`. Optional per-call ``model`` / ``effort`` same as chat.

        Usage is taken from the final chunk when the API provides it
        (``stream_options={"include_usage": True}``). If missing, falls back
        to a local estimate and appends a warning.
        """
        import time

        snap = self._push_call_overrides(model=model, effort=effort)
        try:
            hist, turn, early = self._prepare_chat_turn(
                user,
                history=history,
                tool_result=tool_result,
                system=system,
                bill_prepare=bill_prepare,
            )
            if early is not None:
                return

            client = self._get_client()
            kwargs: dict[str, Any] = {
                "model": self.opt.model,
                "messages": turn["messages"],
                "max_tokens": self.max_tokens_out,
                "temperature": self.temperature,
                "stream": True,
            }
            api_extra = turn.get("api_kwargs") or self.opt.api_kwargs()
            if "extra_body" in api_extra and "extra_body" in kwargs:
                merged = dict(kwargs.get("extra_body") or {})
                merged.update(api_extra["extra_body"] or {})
                kwargs["extra_body"] = merged
                api_extra = {k: v for k, v in api_extra.items() if k != "extra_body"}
            kwargs.update(api_extra)

            stream_kwargs = dict(kwargs)
            try:
                stream_kwargs["stream_options"] = {"include_usage": True}
                stream = client.chat.completions.create(**stream_kwargs)
            except TypeError:
                stream = client.chat.completions.create(**kwargs)
            except Exception:
                stream = client.chat.completions.create(**kwargs)

            t0 = time.perf_counter()
            pieces: list[str] = []
            raw_usage = None
            extra_warns: list[str] = []

            try:
                for chunk in stream:
                    chunk_usage = getattr(chunk, "usage", None)
                    if chunk_usage is not None:
                        raw_usage = chunk_usage
                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue
                    delta = getattr(choices[0], "delta", None)
                    if delta is None:
                        continue
                    part = getattr(delta, "content", None) or ""
                    if part:
                        pieces.append(part)
                        yield part
            finally:
                elapsed = time.perf_counter() - t0
                text = "".join(pieces).strip()
                u = _usage_to_dict(raw_usage) if raw_usage is not None else {}
                if not u or (
                    not u.get("prompt_tokens")
                    and not u.get("completion_tokens")
                    and not u.get("total_tokens")
                ):
                    # Graceful fallback: estimate from compacted prompt + full text
                    extra_warns.append(
                        "[token-optimizer] stream usage missing; "
                        "using local token estimate for this call"
                    )
                    prompt_blob = turn.get("blob") or turn.get("user") or ""
                    sys_blob = turn.get("system") or self.opt.system or ""
                    pin = self.opt.count_tokens(sys_blob + "\n" + prompt_blob)
                    pout = self.opt.count_tokens(text)
                    u = {
                        "prompt_tokens": pin,
                        "completion_tokens": pout,
                        "reasoning_tokens": 0,
                        "cached_tokens": 0,
                        "total_tokens": pin + pout,
                    }

                self._record_usage_and_finish(
                    text,
                    u,
                    hist=hist,
                    turn=turn,
                    history=history,
                    seconds=elapsed,
                    raw=raw_usage,
                    extra_warnings=extra_warns,
                )
        finally:
            self._pop_call_overrides(snap)

    def chat_messages(
        self,
        messages: Sequence[dict[str, str]],
        *,
        compact_user: bool = True,
    ) -> ChatResult:
        """
        Drop-in for existing code that already built messages=[system, user, ...].

        Compacts the last user message by default; keeps your system message.
        """
        msgs = [dict(m) for m in messages]
        system = None
        user = ""
        for m in msgs:
            if m.get("role") == "system" and system is None:
                system = m.get("content") or ""
            if m.get("role") == "user":
                user = m.get("content") or user
        if not user and msgs:
            user = msgs[-1].get("content") or ""
        if compact_user:
            return self.chat(user, system=system)
        # passthrough (still observe usage)
        fat = warn_fat_prompt(user, warn_chars=self.warn_chars)
        self.last_warnings = fat
        import time
        from openai import OpenAI  # noqa: F401

        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.opt.model,
            "messages": msgs,
            "max_tokens": self.max_tokens_out,
            "temperature": self.temperature,
        }
        kwargs.update(self.opt.api_kwargs())
        t0 = time.perf_counter()
        resp = client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - t0
        text = ""
        if resp.choices:
            text = (resp.choices[0].message.content or "").strip()
        u = _usage_to_dict(getattr(resp, "usage", None))
        self.opt.observe_api_usage(u)
        for k in self.session_usage:
            self.session_usage[k] = self.session_usage.get(k, 0) + int(u.get(k, 0) or 0)
        self.session_api_total += int(
            u.get("total_tokens")
            or (
                u.get("prompt_tokens", 0)
                + u.get("completion_tokens", 0)
                + u.get("reasoning_tokens", 0)
            )
        )
        self.session_calls += 1
        return ChatResult(
            text=text,
            usage=u,
            seconds=elapsed,
            model=str(self.opt.model or ""),
            profile=self.profile,
            warnings=list(self.last_warnings),
            raw=resp,
        )

    # --- session reporting ---
    def print_stats(self) -> None:
        print(self.stats_line())
        if self.session_usage:
            print(
                "  usage:",
                {k: v for k, v in self.session_usage.items() if v},
            )
        if self.session_cost_ticks:
            print(
                f"  api_cost=${self.session_cost_usd_api:.8f} "
                f"est_cost=${self.session_cost_usd_est:.8f} "
                f"ticks={self.session_cost_ticks}"
            )

    def summary(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "model": self.opt.model,
            "session_calls": self.session_calls,
            "session_api_total": self.session_api_total,
            "session_usage": dict(self.session_usage),
            "session_cost_usd_api": self.session_cost_usd_api,
            "session_cost_usd_est": self.session_cost_usd_est,
            "session_cost_ticks": self.session_cost_ticks,
            "ticks_per_usd": self.ticks_per_usd,
            "optimizer": self.opt.summary(),
            "rates": {
                "pin": self.opt.pin,
                "pout": self.opt.pout,
                "pin_cached": getattr(self.opt, "pin_cached", self.opt.pin),
            },
        }

    def estimate_session_cost(self) -> float:
        """Sheet-based estimate from accumulated session_usage."""
        return float(
            estimate_cost_usd(
                self.session_usage,
                pin=self.opt.pin,
                pout=self.opt.pout,
                pin_cached=getattr(self.opt, "pin_cached", self.opt.pin),
            )["cost_usd"]
        )

    def compare_cost_accuracy(
        self,
        *,
        dashboard_cost: float | None = None,
        min_accuracy: float = TARGET_ACCURACY,
    ) -> dict[str, Any]:
        """
        Dollar-level accuracy: API ticks cost vs sheet estimate (and optional dashboard $).

        Returns api_cost, est_cost, accuracy, within_99_5, …
        """
        api = self.session_cost_usd_api if self.session_cost_ticks else None
        # Prefer sum of observed API costs; if no ticks, api is None
        if self.session_cost_ticks <= 0:
            api = None
        est = self.session_cost_usd_est or self.estimate_session_cost()
        out = compare_cost_accuracy(
            api,
            est,
            min_accuracy=min_accuracy,
            dashboard_cost=dashboard_cost,
        )
        out["session_cost_ticks"] = self.session_cost_ticks
        out["ticks_per_usd"] = self.ticks_per_usd
        out["session_api_total"] = self.session_api_total
        return out

    def verify_dashboard(
        self,
        before: int,
        after: int,
        *,
        min_accuracy: float = TARGET_ACCURACY,
        use_session_total: bool = True,
        api_total: int | None = None,
        dashboard_cost: float | None = None,
    ) -> dict[str, Any]:
        """
        Compare console before/after tokens (and optional USD) to this session.

        Example:
            before = 251203  # from console
            # ... session.chat calls ...
            after = 252000   # from console
            print(session.verify_dashboard(before, after, dashboard_cost=0.01))
        """
        total = (
            api_total
            if api_total is not None
            else (self.session_api_total if use_session_total else self.opt.total_tokens)
        )
        check = verify_token_delta(
            int(total), int(before), int(after), min_accuracy=min_accuracy
        )
        check["session_api_total"] = self.session_api_total
        check["est_cost_usd"] = self.session_cost_usd_est or self.estimate_session_cost()
        check["api_cost_usd"] = (
            self.session_cost_usd_api if self.session_cost_ticks else None
        )
        check["cost"] = self.compare_cost_accuracy(
            dashboard_cost=dashboard_cost, min_accuracy=min_accuracy
        )
        check["profile"] = self.profile
        check["model"] = self.opt.model
        # Overall ok if tokens ok and (if cost data exists) cost ok
        cost_ok = check["cost"].get("within_99_5")
        if check["cost"].get("api_cost") is not None or dashboard_cost is not None:
            check["within_99_5_tokens"] = check.get("ok")
            check["within_99_5_cost"] = cost_ok
            check["ok"] = bool(check.get("ok")) and bool(cost_ok)
        return check

    def expected_after(self, before: int) -> int:
        """Dashboard total if console matches API exactly."""
        return int(before) + int(self.session_api_total)

    def stats_line(self) -> str:
        base = (
            f"GrokSession profile={self.profile} model={self.opt.model} "
            f"calls={self.session_calls} api_total={self.session_api_total}"
        )
        if self.session_cost_ticks:
            base += f" api_cost=${self.session_cost_usd_api:.8f}"
        base += f" est_cost=${self.session_cost_usd_est or self.estimate_session_cost():.8f}"
        return base + " | " + self.opt.stats_line()

    def export_jsonl(
        self,
        path: str | Path,
        *,
        include_summary: bool = True,
    ) -> int:
        """
        Write per-call audit log as JSONL (for billing / customer support).
        Never includes API keys. Returns number of call lines written.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with p.open("w", encoding="utf-8") as f:
            if include_summary:
                f.write(
                    json.dumps(
                        {
                            "type": "session_summary",
                            **self.summary(),
                            "ticks_per_usd": self.ticks_per_usd,
                            "cost_scale_note": "usd = cost_in_usd_ticks / ticks_per_usd",
                        },
                        default=str,
                    )
                    + "\n"
                )
            for ev in self._events:
                f.write(json.dumps({"type": "call", **ev}, default=str) + "\n")
                n += 1
        return n

    def calibrate(
        self,
        *,
        probe: str = "Reply with exactly: pong",
        dashboard_before: int | None = None,
        dashboard_after: int | None = None,
        dashboard_cost_usd: float | None = None,
        persist: bool = True,
        state_path: str | Path | None = None,
        min_accuracy: float = TARGET_ACCURACY,
    ) -> dict[str, Any]:
        """
        Live one-shot calibration against **this customer's** key and rates.

        Ground truth for dollars: provider ``cost_in_usd_ticks`` (scale 1e10).
        Our sheet estimate is compared; optionally verify console token/USD deltas.

        Persist locks rates + accuracy into calibration_state.json so the UI
        can show Grok as active for this account.
        """
        from .onboarding import OnboardingService, append_ledger, save_state

        # Snapshot meters so calibrate doesn't pollute long-running sessions
        before_calls = self.session_calls
        before_api = self.session_api_total
        before_ticks = self.session_cost_ticks
        before_api_usd = self.session_cost_usd_api
        before_est_usd = self.session_cost_usd_est
        usage_snap = dict(self.session_usage)

        result = self.chat(probe)
        ticks = result.usage.get("cost_in_usd_ticks")
        api_cost = result.cost_usd_api
        est_cost = result.cost_usd_est
        cost_cmp = compare_cost_accuracy(
            api_cost,
            est_cost,
            min_accuracy=min_accuracy,
            dashboard_cost=dashboard_cost_usd,
        )

        out: dict[str, Any] = {
            "provider": "grok",
            "status": "ok" if cost_cmp.get("within_99_5") else "drift",
            "model": result.model or self.opt.model,
            "profile": self.profile,
            "probe": probe,
            "text": (result.text or "")[:80],
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
            "api_total": result.api_total,
            "cost_in_usd_ticks": ticks,
            "ticks_per_usd": self.ticks_per_usd,
            "api_cost_usd": api_cost,
            "est_cost_usd": est_cost,
            "cost": cost_cmp,
            "rates": {
                "pin": self.opt.pin,
                "pout": self.opt.pout,
                "pin_cached": getattr(self.opt, "pin_cached", self.opt.pin),
                "price_source": getattr(self.opt, "provider_label", "catalog"),
            },
            "seconds": result.seconds,
            "within_99_5": bool(cost_cmp.get("within_99_5")),
            "accuracy": cost_cmp.get("accuracy"),
        }

        if dashboard_before is not None and dashboard_after is not None:
            tok = verify_token_delta(
                int(result.api_total),
                int(dashboard_before),
                int(dashboard_after),
                min_accuracy=min_accuracy,
            )
            out["token_delta"] = tok
            out["within_99_5_tokens"] = bool(tok.get("ok"))
            if not tok.get("ok"):
                out["status"] = "token_drift"
            elif out["status"] == "ok":
                out["status"] = "locked"

        if persist:
            path = Path(state_path) if state_path else None
            svc = OnboardingService(state_path=path) if path else OnboardingService()
            tok_acc = None
            if out.get("token_delta"):
                tok_acc = float(out["token_delta"].get("accuracy") or 0)
            # Prefer dollar accuracy from ticks when present
            cost_acc = cost_cmp.get("accuracy")
            if isinstance(cost_acc, (int, float)) and cost_acc >= min_accuracy:
                if tok_acc is None:
                    tok_acc = 1.0  # no token dashboard check; cost ok
            rates = {
                "pin": self.opt.pin,
                "pout": self.opt.pout,
                "pin_cached": getattr(self.opt, "pin_cached", self.opt.pin),
                "price_source": "xai-models-api"
                if "live" in str(getattr(self.opt, "provider_label", ""))
                else "catalog",
            }
            lock_status_acc = float(cost_acc) if isinstance(cost_acc, (int, float)) else None
            # lock when cost matches; token dashboard optional
            effective_tok = tok_acc if tok_acc is not None else (
                1.0 if lock_status_acc is not None and lock_status_acc >= min_accuracy else 0.0
            )
            svc.lock_provider(
                "grok",
                model=str(result.model or self.opt.model),
                rates=rates,
                api_total=int(result.api_total),
                est_cost=float(est_cost or 0),
                token_accuracy=effective_tok,
                cost_accuracy=float(lock_status_acc) if lock_status_acc is not None else None,
                dashboard_total=dashboard_after,
                notes=[
                    f"Customer calibrate profile={self.profile}",
                    f"cost_in_usd_ticks={ticks} scale={self.ticks_per_usd}",
                    f"api_cost={api_cost} est_cost={est_cost}",
                    f"within_99_5={cost_cmp.get('within_99_5')}",
                ],
            )
            # Always record ticks scale for UI / support
            st = svc.state
            st["cost_usd_ticks"] = {
                "scale": int(self.ticks_per_usd),
                "scale_note": f"usd = cost_in_usd_ticks / {int(self.ticks_per_usd)}",
                "status": "locked" if cost_cmp.get("within_99_5") else "drift",
                "verified_model": result.model or self.opt.model,
                "customer_probe": {
                    "cost_in_usd_ticks": ticks,
                    "api_cost_usd": api_cost,
                    "est_cost_usd": est_cost,
                    "accuracy": cost_cmp.get("accuracy"),
                    "within_99_5": cost_cmp.get("within_99_5"),
                },
            }
            st["product"] = product_catalog()
            save_state(st, svc.state_path)
            append_ledger(
                {
                    "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "kind": "customer_calibrate",
                    "provider": "grok",
                    **{k: out[k] for k in out if k not in ("text",)},
                }
            )
            out["persisted"] = True
            out["state_path"] = str(svc.state_path)
        else:
            out["persisted"] = False

        self.last_calibration = out
        return out


def list_profiles() -> dict[str, dict[str, Any]]:
    """Public profile catalog for docs / UIs."""
    roles = {
        "frontier": "design",
        "innovate": "design",
        "production": "volume",
        "bulk": "volume",
    }
    return {
        k: {
            "model": v["model"],
            "reasoning_mode": v["reasoning_mode"],
            "notes": v["notes"],
            "role": roles.get(k, "volume"),
            "ctx_max": v["ctx_max"],
            "hist_keep": v["hist_keep"],
        }
        for k, v in PROFILES.items()
    }


def product_catalog() -> dict[str, Any]:
    """
    UI-facing catalog: Grok active, other providers coming soon.

    Safe for frontends — no keys, no secrets.
    """
    return {
        "version": "0.1.0",
        "ready": True,
        "default_provider": "grok",
        "default_profile": "production",
        "design_profile": "frontier",
        "volume_profiles": ["production", "bulk"],
        "cost_truth": {
            "field": "cost_in_usd_ticks",
            "ticks_per_usd": XAI_USD_TICKS_PER_DOLLAR,
            "formula": "usd = cost_in_usd_ticks / 1e10",
            "est_formula": "uncached*pin + cached*pin_cached + (completion+reasoning)*pout",
            "target_accuracy": TARGET_ACCURACY,
        },
        "profiles": list_profiles(),
        "providers": [
            {
                "id": "grok",
                "name": "xAI Grok",
                "status": "active",
                "profiles": list(PROFILES.keys()),
                "models": ["grok-4.5", "grok-4.3", "grok-4.20-0309-non-reasoning"],
                "calibrate": "GrokSession.calibrate() — live ticks vs sheet $",
                "dashboard": "console.x.ai → Usage (API)",
            },
            {
                "id": "openai",
                "name": "OpenAI",
                "status": "coming_soon",
                "profiles": [],
                "note": "Same compact loop; calibration pending customer keys.",
            },
            {
                "id": "anthropic",
                "name": "Anthropic Claude",
                "status": "coming_soon",
                "profiles": [],
                "note": "Same compact loop; calibration pending customer keys.",
            },
            {
                "id": "gemini",
                "name": "Google Gemini",
                "status": "coming_soon",
                "profiles": [],
                "note": "Same compact loop; calibration pending customer keys.",
            },
        ],
        "playbook": {
            "design": "profile=frontier (or chat(model='grok-4.5', effort='high'))",
            "ship": "profile=production",
            "bulk": "profile=bulk after the idea works",
            "customer_onboard": [
                "Set XAI_API_KEY",
                "session = GrokSession(profile='production')",
                "session.calibrate()  # locks to their live $ ticks",
                "optional: session.verify_dashboard(before, after)",
            ],
        },
    }

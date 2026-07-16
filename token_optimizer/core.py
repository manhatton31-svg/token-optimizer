"""
TokenOptimizer — model-agnostic token budget + context compaction for any agent.

Works with Claude, Gemini, Grok, OpenAI, local models, or rule-based loops.
No vendor SDKs and no language-specific fix logic — only metering, compaction,
retry/token budgets, and compact status emissions.
"""
from __future__ import annotations

import re
import time
from typing import Callable, Iterable, Sequence

from .tokenizers import (
    COST_PRESETS,
    approx_tokens,
    register_tokenizer,
    resolve_provider,
    resolve_tokenizer,
)
from .models import get_model

TokenizerFn = Callable[[str], int]


def tool_compact(text: str | None, max_chars: int = 600) -> str:
    """
    Compact huge tool payloads (file dumps, JSON, logs) for agent turns.

    Keeps high-signal lines (errors, paths, keys) plus head/tail.
    Never raises on bad input.
    """
    try:
        if text is None:
            return ""
        raw = str(text)
        if not raw.strip():
            return ""
        max_chars = max(80, int(max_chars) if max_chars else 600)

        # Normalize newlines / drop NULs
        raw = raw.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
        lines = [ln.rstrip() for ln in raw.split("\n")]

        def is_signal(ln: str) -> bool:
            s = ln.strip()
            if not s:
                return False
            low = s.lower()
            if re.search(
                r"(error|exception|traceback|fail|warning|denied|timeout|"
                r"status|total|price|tax|legacy|undefined|not found)",
                low,
            ):
                return True
            if re.search(r"[\\/][\w.\-]+(?:\.[\w]+)+", s):  # path-like
                return True
            if re.search(r"['\"][\w.\-]+['\"]\s*:", s):  # json key
                return True
            if re.match(r"^(def |class |function |FILE |TOOL |GET |POST )", s):
                return True
            if ":" in s and len(s) < 120 and not s.startswith(" " * 8):
                return True
            return False

        # Drop pure filler / repeated xxxx lines
        cleaned: list[str] = []
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if re.fullmatch(r"[xX.\-\s]{20,}", s):
                continue
            if re.fullmatch(r"line \d+:\s*x+", s, re.I):
                continue
            cleaned.append(ln[:240] if len(ln) > 240 else ln)

        if not cleaned:
            return raw[:max_chars]

        signal = [ln for ln in cleaned if is_signal(ln)]
        # Unique preserve order
        seen: set[str] = set()
        signal_u: list[str] = []
        for ln in signal:
            if ln not in seen:
                seen.add(ln)
                signal_u.append(ln)

        head_n = max(3, min(8, max_chars // 80))
        tail_n = max(3, min(8, max_chars // 80))
        head = cleaned[:head_n]
        tail = cleaned[-tail_n:] if len(cleaned) > head_n else []

        parts: list[str] = []
        parts.extend(head)
        mid = [ln for ln in signal_u[:12] if ln not in head and ln not in tail]
        if mid:
            parts.append("…")
            parts.extend(mid)
        if tail and tail != head:
            parts.append("…")
            parts.extend(tail)

        out = "\n".join(parts)
        if len(cleaned) > head_n + tail_n:
            out = f"[tool compact {len(raw)} chars → keep signal/head/tail]\n" + out
        if len(out) > max_chars:
            half = max(40, (max_chars - 5) // 2)
            out = out[:half] + "\n…\n" + out[-half:]
        return out
    except Exception:
        try:
            s = str(text or "")
            return s[: max(80, int(max_chars) if max_chars else 600)]
        except Exception:
            return ""


class TokenOptimizer:
    """
    Drop-in session budgeter for multi-turn / tool-using agents.

    Features
    --------
    - Token counts: fast char-approx (default) or provider tokenizers
      (tiktoken, Anthropic, Gemini, custom via register_tokenizer / tokenizer=)
    - Dashboard cost presets: grok, openai, anthropic, gemini, cursor, spark
    - compact_context, retry/token budgets, stagnation, compact emits
    """

    DEFAULT_PIN = 0.20
    DEFAULT_POUT = 0.50

    def __init__(
        self,
        system: str = "Agent.",
        pin: float | None = None,
        pout: float | None = None,
        err_tail: int = 120,
        ctx_max: int = 280,
        hist_keep: int = 2,
        max_retries: int = 5,
        max_tokens: int = 10_000,
        emit_max: int = 8,
        stagnate_after: int = 3,
        reasoning_mode: str = "full",
        *,
        provider: str | None = None,
        tokenizer: str | TokenizerFn | None = None,
        encoding: str | None = None,
        model: str | None = None,
    ):
        """
        provider : dashboard preset — grok|grok-4.5|openai|anthropic|gemini|…
        model    : concrete id (e.g. grok-4.5, grok-4.3) — sets tier rates when known
        tokenizer : None/'approx' | 'tiktoken' | 'anthropic' | 'gemini' | callable
        pin/pout : USD per 1M tokens (override preset / model sheet)
        """
        # Model id can imply provider preset (heavy-user tiering)
        if model and provider is None and get_model(model):
            provider = get_model(model).provider  # type: ignore[union-attr]
        if model and provider in (None, "grok", "xai") and model in COST_PRESETS:
            provider = model  # e.g. provider path grok-4.5

        resolved = resolve_provider(
            provider,
            pin=pin,
            pout=pout,
            tokenizer=tokenizer,
            encoding=encoding,
            model=model,
        )
        # If no provider and no tokenizer, keep classic defaults
        if provider is None and tokenizer is None and pin is None and pout is None and model is None:
            self.pin = self.DEFAULT_PIN
            self.pout = self.DEFAULT_POUT
            self.pin_cached = self.DEFAULT_PIN
            self._tokenizer: TokenizerFn = approx_tokens
            self.tokenizer_name = "approx"
            self.provider = "approx"
            self.provider_label = "Approx (~4 chars/tok)"
            self.model = model
            self.tier = "approx"
        else:
            self.pin = resolved["pin"]
            self.pout = resolved["pout"]
            self.pin_cached = float(resolved.get("pin_cached") or resolved["pin"])
            self._tokenizer = resolved["tokenizer"]
            self.tokenizer_name = resolved["tokenizer_name"]
            self.provider = resolved["provider"]
            self.provider_label = resolved["label"]
            self.model = resolved.get("model") or model
            self.tier = str(COST_PRESETS.get(provider or "", {}).get("tier") or "standard")

        # Concrete ModelSpec overrides rates (frontier vs production)
        spec = get_model(model or "") or get_model(str(self.model or ""))
        if spec is not None:
            if pin is None:
                self.pin = spec.pin
            if pout is None:
                self.pout = spec.pout
            self.pin_cached = spec.pin_cached
            self.model = spec.id
            self.tier = spec.tier
            self.provider_label = spec.label
            if provider is None:
                self.provider = spec.provider

        # Explicit tokenizer= without provider still honored
        if tokenizer is not None and provider is None and spec is None:
            fn, tname = resolve_tokenizer(
                tokenizer, encoding=encoding, model=model
            )
            self._tokenizer = fn
            self.tokenizer_name = tname
            if pin is not None:
                self.pin = pin
            if pout is not None:
                self.pout = pout

        if not hasattr(self, "pin_cached"):
            self.pin_cached = self.pin

        self.system = system
        self.err_tail = err_tail
        self.ctx_max = ctx_max
        self.hist_keep = hist_keep
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.emit_max = emit_max
        self.stagnate_after = max(2, stagnate_after)
        # full|innovate|balanced|low|off — innovators keep full reasoning quality;
        # 90%+ savings come from compact context, not from dumbing the model down.
        self.reasoning_mode = (reasoning_mode or "full").strip().lower()
        self.tin = 0
        self.tout = 0
        self.retries = 0
        self.budget_error = ""
        self._attempts: list[dict] = []
        self.t0 = time.perf_counter()

    # --- counting ---
    @staticmethod
    def tok(text: str) -> int:
        """Module-level fast estimate (always char-approx). Prefer count_tokens()."""
        return approx_tokens(text)

    def count_tokens(self, text: str) -> int:
        """Count tokens with the configured tokenizer (billing-accurate when set)."""
        try:
            n = self._tokenizer(text or "")
            return int(n) if n else 0
        except Exception:
            return approx_tokens(text or "")

    def set_tokenizer(
        self,
        tokenizer: str | TokenizerFn,
        *,
        encoding: str | None = None,
        model: str | None = None,
    ) -> None:
        """Swap tokenizer at runtime (e.g. after installing tiktoken)."""
        fn, name = resolve_tokenizer(tokenizer, encoding=encoding, model=model)
        self._tokenizer = fn
        self.tokenizer_name = name

    def use_provider(self, provider: str, **kwargs) -> None:
        """Apply a dashboard cost + tokenizer preset (grok, openai, …)."""
        resolved = resolve_provider(provider, **kwargs)
        self.pin = resolved["pin"]
        self.pout = resolved["pout"]
        self._tokenizer = resolved["tokenizer"]
        self.tokenizer_name = resolved["tokenizer_name"]
        self.provider = resolved["provider"]
        self.provider_label = resolved["label"]
        self.model = resolved.get("model")

    def add_in(self, s: str) -> None:
        self.tin += self.count_tokens(s)

    def add_out(self, s: str) -> None:
        self.tout += self.count_tokens(s)

    def charge_system(self) -> None:
        """Bill the system prompt once (keep it short)."""
        self.add_in(self.system)

    def charge_tick(self, tag: str = "t") -> None:
        """Minimal per-task / per-case marker on the input meter."""
        self.add_in(tag)

    # --- retry / token budget ---
    @property
    def total_tokens(self) -> int:
        return self.tin + self.tout

    def begin_task(self) -> None:
        """Reset per-task retry counter and stagnation window (keeps token totals)."""
        self.retries = 0
        self.budget_error = ""
        self._attempts.clear()

    def bump_retry(self) -> int:
        """Count one attempt; return the new retry count."""
        self.retries += 1
        return self.retries

    @staticmethod
    def _fingerprint(action: str, outcome: str) -> str:
        """Normalize action/outcome into a short comparable key."""
        def norm(s: str) -> str:
            s = (s or "").lower().strip()
            s = re.sub(r"\s+", " ", s)
            # drop volatile numbers/ids so the same error class matches
            s = re.sub(r"\b\d+\b", "#", s)
            s = re.sub(r"0x[0-9a-f]+", "0x#", s)
            return s[:120]

        a, o = norm(action), norm(outcome)
        # prefer error class token if present
        m = re.search(
            r"\b([a-z_]*error|[a-z_]*exception|logic|timeout|denied)\b", o
        )
        kind = m.group(1) if m else (o[:40] if o else "unk")
        return f"{a}|{kind}|{o[:60]}"

    def record_attempt(
        self,
        action: str = "",
        outcome: str = "",
        *,
        ok: bool = False,
        fingerprint: str | None = None,
    ) -> str:
        """
        Track one step's action and outcome for stagnation detection.

        Call after each tool call / fix / model step. Failures with the same
        fingerprint stacked `stagnate_after` times trip is_stuck().
        """
        fp = fingerprint or self._fingerprint(action, outcome)
        rec = {
            "fp": fp,
            "ok": bool(ok),
            "action": (action or "")[:80],
            "outcome": (outcome or "")[:120],
        }
        self._attempts.append(rec)
        # keep a small ring buffer
        if len(self._attempts) > max(8, self.stagnate_after * 3):
            self._attempts = self._attempts[-self.stagnate_after * 3 :]
        return fp

    def is_stuck(self) -> bool:
        """
        True if the last N attempts all failed with the same fingerprint
        (repeating similar failing pattern).
        """
        n = self.stagnate_after
        if len(self._attempts) < n:
            return False
        window = self._attempts[-n:]
        if any(a.get("ok") for a in window):
            return False
        fps = [a.get("fp") for a in window]
        if not fps[0]:
            return False
        return all(fp == fps[0] for fp in fps)

    def can_continue(self) -> bool:
        """
        True if under retry budget, token budget, and not stagnating.
        On False, budget_error holds a clear stop reason.
        """
        if self.retries >= self.max_retries:
            self.budget_error = (
                f"Retry budget exhausted: {self.retries}/{self.max_retries} "
                f"retries used. Stop cleanly — no further attempts."
            )
            return False
        tot = self.total_tokens
        if tot >= self.max_tokens:
            self.budget_error = (
                f"Token budget exhausted: {tot}/{self.max_tokens} tokens used "
                f"(in={self.tin}, out={self.tout}). Stop cleanly — no further attempts."
            )
            return False
        if self.is_stuck():
            last = self._attempts[-1] if self._attempts else {}
            self.budget_error = (
                f"Stagnation detected: {self.stagnate_after} similar failing "
                f"attempts in a row (pattern={last.get('fp', '?')!r}). "
                f"Stop cleanly — no further attempts."
            )
            return False
        self.budget_error = ""
        return True

    # --- compaction (language-agnostic) ---
    def compact_err(self, err: str) -> str:
        """Keep a short tail / last signal line from a note, tool error, or log."""
        if not err:
            return ""
        lines = [ln.strip() for ln in err.strip().splitlines() if ln.strip()]
        for ln in reversed(lines):
            if re.search(
                r"(error|exception|fail|warn|timeout|denied|logic|want=|got=|status)",
                ln,
                re.I,
            ) and not re.match(r"^(file |at |in |#\d)", ln, re.I):
                return ln[: self.err_tail]
        if len(err) <= self.err_tail:
            return err.strip()
        return err[-self.err_tail :].strip()

    def _compact_structured(self, text: str, budget: int) -> str:
        """
        Compress structured text (source, JSON-ish, configs, logs):
        drop blank/comment-only lines, collapse spaces, keep head + key lines + tail.
        """
        lines: list[str] = []
        for ln in text.splitlines():
            s = ln.rstrip()
            if not s:
                continue
            stripped = s.lstrip()
            if (
                stripped.startswith("#")
                or stripped.startswith("//")
                or stripped.startswith("<!--")
            ):
                continue
            indent = len(s) - len(stripped)
            body = re.sub(r"\s+", " ", stripped)
            lines.append((" " if indent else "") + body)
        if not lines:
            return ""
        joined = "\n".join(lines)
        if len(joined) <= budget:
            return joined
        keys = [
            ln
            for ln in lines
            if re.search(
                r"\b(def|class|function|fn|func|import|export|package|module|"
                r"SELECT|FROM|CREATE|interface|struct|impl|pub|async)\b|"
                r"[{}\[\]=]|=>|::",
                ln,
                re.I,
            )
        ]
        head_n = max(2, budget // 40)
        tail_n = max(3, budget // 30)
        head, tail, mid = lines[:head_n], lines[-tail_n:], keys[:4]
        parts = head + (["…"] if mid or tail else []) + mid
        if tail and tail != head:
            parts += ["…"] + tail
        out = "\n".join(parts)
        return out if len(out) <= budget else out[: budget - 1] + "…"

    def _compact_prose(self, text: str, budget: int) -> str:
        t = re.sub(r"[ \t]+", " ", text)
        t = re.sub(r"\n{3,}", "\n\n", t).strip()
        if len(t) <= budget:
            return t
        half = max(40, budget // 2 - 1)
        return t[:half] + "…" + t[-half:]

    def _looks_structured(self, text: str) -> bool:
        return bool(
            re.search(
                r"^\s*(def |class |function |fn |import |export |package |SELECT )|"
                r"[{}\[\]]|Traceback|Error:|Exception|\"\w+\":\s",
                text,
                re.M | re.I,
            )
        )

    def compact_context(
        self,
        context: str | Sequence[str] | None = None,
        *,
        content: str | None = None,
        note: str | None = None,
        history: Sequence[str] | None = None,
        max_chars: int | None = None,
        code: str | None = None,
        err: str | None = None,
    ) -> str:
        """
        Compress conversation history and/or large documents + short notes.

        Parameters
        ----------
        context : raw string or list of turns
        content : large document / tool payload / source (alias: code)
        note    : error, observation, or tool stderr tail (alias: err)
        history : prior turns; only the last hist_keep are kept
        max_chars : hard cap (default ctx_max)
        """
        budget = self.ctx_max if max_chars is None else max_chars
        doc = content if content is not None else code
        signal = note if note is not None else err
        chunks: list[str] = []

        turns: list[str] = []
        if history:
            turns.extend(str(h) for h in history if h)
        if isinstance(context, (list, tuple)):
            turns.extend(str(h) for h in context if h)
        elif (
            isinstance(context, str)
            and context.strip()
            and doc is None
            and signal is None
        ):
            blob = context
            if self._looks_structured(blob):
                chunks.append(self._compact_structured(blob, budget))
            else:
                chunks.append(self._compact_prose(blob, budget))

        if turns:
            keep = turns[-self.hist_keep :]
            per = max(40, budget // max(1, len(keep)))
            for t in keep:
                if self._looks_structured(t):
                    chunks.append(self._compact_structured(t, per))
                else:
                    chunks.append(self._compact_prose(t, per))

        if doc is not None and str(doc).strip():
            cb = int(budget * 0.6) if signal else budget
            chunks.append(self._compact_structured(str(doc), cb))

        if signal is not None and str(signal).strip():
            chunks.append(self.compact_err(str(signal)))

        out = "\n".join(c for c in chunks if c)
        if len(out) > budget:
            out = out[: budget - 1] + "…"
        return out or "t"

    def bill_context(
        self,
        content: str | None = None,
        note: str | None = None,
        history: Sequence[str] | None = None,
        context: str | Sequence[str] | None = None,
        *,
        code: str | None = None,
        err: str | None = None,
    ) -> str:
        """Compact then charge the result as input tokens."""
        blob = self.compact_context(
            context,
            content=content,
            note=note,
            history=history,
            code=code,
            err=err,
        )
        self.add_in(blob)
        return blob

    # --- compact output ---
    def emit(self, status: str) -> str:
        """
        Record a short model/agent status on the output meter.
        Prefer tags over essays when you only need internal control signals.
        """
        s = status if len(status) <= self.emit_max else status[: self.emit_max]
        self.add_out(s)
        return s

    def ok(self) -> str:
        return self.emit("ok")

    def fixed(self) -> str:
        """Generic 'changed state / applied step' signal."""
        return self.emit("f")

    def fail(self) -> str:
        return self.emit("x")

    def emit_many(self, tags: Iterable[str]) -> None:
        for t in tags:
            self.emit(t)

    # --- report ---
    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    @property
    def cost_usd(self) -> float:
        return self.tin / 1e6 * self.pin + self.tout / 1e6 * self.pout

    def stats_line(self, passed: int | None = None, total: int | None = None) -> str:
        head = ""
        if passed is not None and total is not None:
            head = f"{passed}/{total} | "
        return (
            f"{head}Input tokens: {self.tin}, Output tokens: {self.tout}, "
            f"Total time: {self.elapsed:.3f} seconds | cost ${self.cost_usd:.8f} "
            f"| {self.provider_label} tok={self.tokenizer_name} "
            f"(${self.pin}/M in, ${self.pout}/M out)"
        )

    def print_stats(self, passed: int | None = None, total: int | None = None) -> None:
        print(self.stats_line(passed, total))

    def reset(self) -> None:
        self.tin = self.tout = 0
        self.retries = 0
        self.budget_error = ""
        self._attempts.clear()
        self.t0 = time.perf_counter()

    def summary(self) -> dict:
        return {
            "input_tokens": self.tin,
            "output_tokens": self.tout,
            "total_tokens": self.total_tokens,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "max_tokens": self.max_tokens,
            "stagnate_after": self.stagnate_after,
            "stuck": self.is_stuck(),
            "recent_attempts": len(self._attempts),
            "budget_error": self.budget_error,
            "seconds": round(self.elapsed, 6),
            "cost_usd": self.cost_usd,
            "provider": self.provider,
            "provider_label": self.provider_label,
            "tokenizer": self.tokenizer_name,
            "pin_per_m": self.pin,
            "pout_per_m": self.pout,
            "pin_cached_per_m": getattr(self, "pin_cached", self.pin),
            "model": self.model,
            "tier": getattr(self, "tier", "standard"),
            "reasoning_mode": getattr(self, "reasoning_mode", "full"),
            "reasoning_effort": self.reasoning_effort()
            if hasattr(self, "reasoning_mode")
            else None,
        }

    def use_model(self, model_id: str) -> None:
        """Switch rates/label to a catalog model (e.g. grok-4.5 vs grok-4.3)."""
        spec = get_model(model_id)
        if spec is None:
            self.model = model_id
            return
        self.model = spec.id
        self.pin = spec.pin
        self.pout = spec.pout
        self.pin_cached = spec.pin_cached
        self.tier = spec.tier
        self.provider = spec.provider
        self.provider_label = spec.label

    # --- reasoning (quality) vs context (savings) ---
    _EFFORT_MAP = {
        "full": "high",
        "innovate": "high",
        "high": "high",
        "balanced": "medium",
        "medium": "medium",
        "low": "low",
        "off": None,
        "none": None,
    }

    def set_reasoning_mode(self, mode: str) -> None:
        """full/innovate (max quality) | balanced | low | off."""
        self.reasoning_mode = (mode or "full").strip().lower()

    def reasoning_effort(self) -> str | None:
        """Provider effort flag for xAI/OpenAI-style APIs (None = omit / model default)."""
        return self._EFFORT_MAP.get(self.reasoning_mode, "high")

    def api_kwargs(self) -> dict:
        """
        Extra kwargs to merge into chat.completions.create(...).

        Savings do NOT require lowering reasoning — pass full effort while
        prepare_turn() still ships a tiny user blob.
        """
        effort = self.reasoning_effort()
        if effort is None:
            return {}
        # xAI accepts reasoning_effort via extra_body
        return {"extra_body": {"reasoning_effort": effort}}

    def innovator_profile(self) -> None:
        """
        Max reasoning quality + max context compression.
        Use while inventing/debugging; still targets 90%+ vs naive full dumps.
        """
        self.reasoning_mode = "innovate"
        self.err_tail = min(self.err_tail, 64)
        self.ctx_max = min(self.ctx_max, 140)
        self.hist_keep = min(self.hist_keep, 2)
        self.emit_max = min(self.emit_max, 4)

    # ------------------------------------------------------------------
    # DIY / custom-loop API (no OptimizedLoop required)
    # Heavy users keep their own control flow; we only compact + meter.
    # ------------------------------------------------------------------

    def prepare_turn(
        self,
        user: str,
        *,
        history: Sequence[str] | None = None,
        tool_result: str | None = None,
        system: str | None = None,
        bill: bool = True,
        bump: bool = True,
        soft: bool = True,
    ) -> dict:
        """
        Prepare one model turn for a *custom* agent loop.

        Does **not** take over control flow. Always returns a usable payload
        so builders can keep iterating even near budget limits.

        Parameters
        ----------
        user : current user / task text (may be long — will be compacted)
        history : prior turns (only last hist_keep kept after compact)
        tool_result : optional tool output to fold into note/context
                      (auto-compacted via tool_compact)
        system : override system string for this turn's messages
        bill : if True, charge compacted blob as input
        bump : if True, count a retry attempt
        soft : if True (default), never hard-block — set stopped=True instead

        Returns
        -------
        dict with keys:
          system, user, messages, stopped, reason, blob, tool_result_compact
        """
        stopped = False
        reason = ""
        # Always compact tool payloads (never ship full file dumps)
        tool_cap = max(120, min(800, int(self.ctx_max) * 3))
        tool_compacted = (
            tool_compact(tool_result, max_chars=tool_cap)
            if tool_result is not None
            else None
        )
        if bump:
            # only bump when under hard limits if not soft
            if soft or self.can_continue():
                if self.can_continue():
                    self.bump_retry()
            if not self.can_continue():
                stopped = True
                reason = self.budget_error or "budget"
                if not soft:
                    return {
                        "system": system if system is not None else self.system,
                        "user": "",
                        "messages": [],
                        "stopped": True,
                        "reason": reason,
                        "blob": "",
                    }

        note = tool_compacted
        blob = self.compact_context(
            context=user,
            note=note,
            history=history,
        )
        if bill:
            self.add_in(blob)

        sys_msg = system if system is not None else self.system
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": blob},
        ]
        # Re-check after billing (token cap)
        if not self.can_continue():
            stopped = True
            reason = self.budget_error or reason or "budget"

        return {
            "system": sys_msg,
            "user": blob,
            "messages": messages,
            "stopped": stopped,
            "reason": reason,
            "blob": blob,
            "tool_result_compact": tool_compacted,
            "api_kwargs": self.api_kwargs(),
            "reasoning_mode": self.reasoning_mode,
            "reasoning_effort": self.reasoning_effort(),
        }

    def finish_turn(
        self,
        response: object = "",
        *,
        ok: bool | None = None,
        action: str = "step",
        emit_status: bool = True,
        history: list[str] | None = None,
        user_blob: str = "",
    ) -> dict:
        """
        Optional post-model hook for DIY loops. Safe no-op style: never raises.

        - records stagnation fingerprint
        - optionally emits compact output tag
        - optionally appends compacted snippets onto a history list you own
        """
        text = "" if response is None else str(response)
        if ok is None:
            ok = not bool(
                re.search(
                    r"\b(error|exception|fail|traceback|timeout|denied)\b",
                    text,
                    re.I,
                )
            )
        self.record_attempt(action=action, outcome=text, ok=ok)
        if emit_status:
            self.emit("ok" if ok else "x")

        obs = self.compact_context(context=text, max_chars=min(120, self.ctx_max))
        if history is not None:
            if user_blob:
                history.append(
                    self.compact_context(context=user_blob, max_chars=80)
                )
            history.append(obs)
            keep = max(4, self.hist_keep * 4)
            del history[: max(0, len(history) - keep)]

        return {
            "ok": ok,
            "stopped": not self.can_continue(),
            "reason": self.budget_error,
            "observation": obs,
        }

    def observe_api_usage(self, usage: dict | None) -> dict:
        """
        Align local meters with provider usage when the DIY loop has it.

        Call after each API response: observe_api_usage(response.usage).
        Overwrites tin/tout for this session snapshot (additive when possible).
        """
        if not usage:
            return {"applied": False}
        # Accept OpenAI / Anthropic / Gemini-ish shapes
        pin = int(
            getattr(usage, "prompt_tokens", None)
            or usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or 0
        )
        pout = int(
            getattr(usage, "completion_tokens", None)
            or usage.get("completion_tokens")
            or usage.get("output_tokens")
            or 0
        )
        # reasoning often billed as output — include if present
        reason = 0
        det = None
        if isinstance(usage, dict):
            reason = int(usage.get("reasoning_tokens") or 0)
            det = usage.get("completion_tokens_details")
        else:
            det = getattr(usage, "completion_tokens_details", None)
        if not reason and det is not None:
            try:
                reason = int(getattr(det, "reasoning_tokens", 0) or 0)
            except Exception:
                m = re.search(r"reasoning_tokens=(\d+)", str(det))
                if m:
                    reason = int(m.group(1))
        # Additive observe (DIY loops often call once per turn)
        self.tin += pin
        self.tout += pout + reason
        return {
            "applied": True,
            "prompt_tokens": pin,
            "completion_tokens": pout,
            "reasoning_tokens": reason,
            "session_tin": self.tin,
            "session_tout": self.tout,
        }


# ---------------------------------------------------------------------------
# OptimizedLoop — optional wrapper (DIY loops do not need this)
# ---------------------------------------------------------------------------


class LoopStep:
    """
    One task/step inside an OptimizedLoop.

    On enter: begin_task (optional), bill compacted prompt + history, bump retry.
    record(): track stagnation, compact observation, emit short status.
    """

    __slots__ = (
        "_loop",
        "prompt",
        "label",
        "active",
        "stopped",
        "reason",
        "ok",
        "_recorded",
        "billed_prompt",
    )

    def __init__(self, loop: "OptimizedLoop", prompt: str, label: str = ""):
        self._loop = loop
        self.prompt = prompt
        self.label = label or "task"
        self.active = True
        self.stopped = False
        self.reason = ""
        self.ok: bool | None = None
        self._recorded = False
        self.billed_prompt = ""

    @property
    def opt(self) -> TokenOptimizer:
        return self._loop.opt

    @property
    def context(self) -> str:
        """Compact prompt + history — safe to pass to any model."""
        return self.opt.compact_context(
            context=self.prompt,
            history=self._loop.history,
        )

    def __enter__(self) -> "LoopStep":
        opt = self.opt
        if self._loop._reset_each_task:
            opt.begin_task()
        if not opt.can_continue():
            self.active = False
            self.stopped = True
            self.reason = opt.budget_error or "budget exhausted"
            return self
        opt.bump_retry()
        # bill compacted user/task context (+ rolling history)
        self.billed_prompt = opt.bill_context(
            context=self.prompt,
            history=self._loop.history,
        )
        if not opt.can_continue():
            self.active = False
            self.stopped = True
            self.reason = opt.budget_error or "budget exhausted"
        return self

    @property
    def messages(self) -> list[dict[str, str]]:
        """OpenAI-style messages for this step (compact user content)."""
        return [
            {"role": "system", "content": self.opt.system},
            {"role": "user", "content": self.context},
        ]

    @property
    def api_kwargs(self) -> dict:
        """Merge into chat.completions.create(..., **step.api_kwargs)."""
        return self.opt.api_kwargs()

    def record(
        self,
        result: object,
        *,
        ok: bool | None = None,
        action: str = "step",
        emit: str | None = None,
    ) -> object:
        """
        Record model/tool result: stagnation tracking + compact history + short out.

        ok=None → treat as success unless result text looks like an error/failure.
        """
        opt = self.opt
        text = "" if result is None else str(result)
        if ok is None:
            ok = not bool(
                re.search(
                    r"\b(error|exception|fail|traceback|timeout|denied)\b",
                    text,
                    re.I,
                )
            )
        self.ok = ok
        opt.record_attempt(action=action or self.label, outcome=text, ok=ok)

        # compact observation for next turns (not full dump)
        obs = opt.compact_context(context=text, max_chars=min(160, opt.ctx_max))
        opt.add_in(f"obs:{obs}")

        tag = emit if emit is not None else ("ok" if ok else "x")
        opt.emit(tag)

        self._loop.history.append(
            opt.compact_context(context=f"{self.label}:{self.prompt}", max_chars=80)
        )
        self._loop.history.append(obs)
        # cap history length outside TokenOptimizer hist_keep (session-level)
        keep = max(4, opt.hist_keep * 4)
        if len(self._loop.history) > keep:
            self._loop.history = self._loop.history[-keep:]

        self._recorded = True
        if not opt.can_continue():
            self.stopped = True
            self.reason = opt.budget_error
            self.active = False
        return result

    def __exit__(self, exc_type, exc, tb) -> bool:
        opt = self.opt
        if exc_type is not None:
            opt.record_attempt(
                action=self.label,
                outcome=f"{exc_type.__name__}: {exc}",
                ok=False,
            )
            opt.fail()
            self.ok = False
            self._recorded = True
            self.stopped = True
            self.reason = str(exc)
            return False  # do not suppress
        if not self._recorded and self.active:
            # no record() call — count as empty failed step
            opt.record_attempt(action=self.label, outcome="no_record", ok=False)
            opt.fail()
            self.ok = False
        return False


class OptimizedLoop:
    """
    Thin wrapper so agents adopt TokenOptimizer with almost no boilerplate.

    Heavy-user pattern
    ------------------
    design = OptimizedLoop(system="Architect.", model="grok-4.5")   # frontier
    prod   = OptimizedLoop(system="fix.", model="grok-4.3")       # calibrated volume

    Example
    -------
    from token_optimizer import OptimizedLoop

    opt = OptimizedLoop(system="You are a helpful assistant.", model="grok-4.3")

    for task in tasks:
        with opt.task("Fix this bug") as step:
            if not step.active:
                break
            result = model.generate(step.context)
            step.record(result)

    opt.print_stats()
    """

    def __init__(
        self,
        system: str = "You are a helpful assistant.",
        *,
        reset_each_task: bool = True,
        charge_system_once: bool = True,
        model: str | None = None,
        innovate: bool = False,
        reasoning_mode: str | None = None,
        **optimizer_kwargs,
    ):
        """
        Parameters
        ----------
        system : system prompt (kept short; billed once by default)
        model : e.g. grok-4.5 (frontier) or grok-4.3 (production)
        innovate : True → full reasoning + aggressive context compression
        reasoning_mode : full|innovate|balanced|low|off (default full)
        reset_each_task : call begin_task() on each `with opt.task(...)`
        charge_system_once : bill system prompt at construction
        optimizer_kwargs : forwarded to TokenOptimizer (pin, pout, max_retries, …)
        """
        if model is not None:
            optimizer_kwargs.setdefault("model", model)
        if reasoning_mode is not None:
            optimizer_kwargs["reasoning_mode"] = reasoning_mode
        elif innovate:
            optimizer_kwargs.setdefault("reasoning_mode", "innovate")
        self.opt = TokenOptimizer(system=system, **optimizer_kwargs)
        if innovate:
            self.opt.innovator_profile()
        self.history: list[str] = []
        self._reset_each_task = reset_each_task
        self.model = self.opt.model
        if charge_system_once:
            self.opt.charge_system()

    def task(self, prompt: str, label: str = "") -> LoopStep:
        """Start a metered task step (use as context manager)."""
        return LoopStep(self, prompt=str(prompt), label=label)

    def can_continue(self) -> bool:
        return self.opt.can_continue()

    @property
    def budget_error(self) -> str:
        return self.opt.budget_error

    def print_stats(self, passed: int | None = None, total: int | None = None) -> None:
        self.opt.print_stats(passed, total)

    def stats_line(self, passed: int | None = None, total: int | None = None) -> str:
        return self.opt.stats_line(passed, total)

    def summary(self) -> dict:
        return self.opt.summary()

    def reset(self) -> None:
        self.opt.reset()
        self.history.clear()
        if self.opt.system:
            self.opt.charge_system()

    def feedback(self, text: str) -> str:
        """
        Print a consistent, collectible feedback line for logs/telemetry.

        Format:
          [token-optimizer] <message> | Input tokens: … | cost $… | <provider> …
        """
        msg = (text or "").strip().replace("\n", " ")
        line = f"[token-optimizer] {msg} | {self.stats_line()}"
        print(line)
        return line

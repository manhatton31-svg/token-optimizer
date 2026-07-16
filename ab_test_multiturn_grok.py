#!/usr/bin/env python3
"""
Multi-turn live Grok A/B: naive full-history agent vs compact OptimizedLoop.

Confirms high savings the way the 21-case suite does, but with real API usage
you can match on console.x.ai.

Baseline expected start: 9506 (user-confirmed).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

for candidate in (Path.cwd() / ".env", Path.home() / ".env"):
    if candidate.is_file():
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip("'").strip('"')
            if k and not os.environ.get(k):
                os.environ[k] = v

from openai import OpenAI

from token_optimizer import OptimizedLoop
from token_optimizer.costing import estimate_cost_usd, fetch_xai_model_pricing
from test_suite import SUITE

MODEL = "grok-4.3"
BASE_URL = "https://api.x.ai/v1"
MAX_OUT = 180
ROUNDS = 3
DASHBOARD_BEFORE = 9506

# Use the heavy multi-file case + one medium case for multi-turn realism
CASES = []
for cat, name, src, exp in SUITE:
    if name in (
        "checkout_multi_file_rename_and_tax",
        "typo_deep_in_module",
        "indent_and_arity",
    ):
        CASES.append((name, src, exp))

SYSTEM = (
    "You are a debugging agent in a multi-turn loop. "
    "Each turn: identify the current error, propose one fix, "
    "output only a short patch description (under 120 words)."
)


def client() -> OpenAI:
    key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not key:
        raise SystemExit("XAI_API_KEY required")
    return OpenAI(api_key=key, base_url=BASE_URL)


def usage_dict(resp) -> dict:
    u = getattr(resp, "usage", None)
    out = {
        "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(u, "total_tokens", 0) or 0),
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
    if not u:
        return out
    pdet = getattr(u, "prompt_tokens_details", None)
    cdet = getattr(u, "completion_tokens_details", None)
    if pdet is not None:
        try:
            out["cached_tokens"] = int(getattr(pdet, "cached_tokens", 0) or 0)
        except Exception:
            m = re.search(r"cached_tokens=(\d+)", str(pdet))
            if m:
                out["cached_tokens"] = int(m.group(1))
    if cdet is not None:
        try:
            out["reasoning_tokens"] = int(getattr(cdet, "reasoning_tokens", 0) or 0)
        except Exception:
            m = re.search(r"reasoning_tokens=(\d+)", str(cdet))
            if m:
                out["reasoning_tokens"] = int(m.group(1))
    return out


def add_usage(agg: dict, u: dict) -> None:
    for k in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
    ):
        agg[k] = agg.get(k, 0) + int(u.get(k, 0) or 0)


def call(system: str, user: str) -> tuple[str, dict, float]:
    c = client()
    t0 = time.perf_counter()
    resp = c.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=MAX_OUT,
        temperature=0.2,
    )
    elapsed = time.perf_counter() - t0
    text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
    return text, usage_dict(resp), elapsed


def run_src(src: str) -> tuple[bool, str]:
    import traceback
    from io import StringIO
    import sys

    buf, old, ns = StringIO(), sys.stdout, {}
    sys.stdout = buf
    try:
        exec(compile(src, "<t>", "exec"), ns, ns)
        return True, buf.getvalue()
    except Exception:
        return False, traceback.format_exc(limit=2)
    finally:
        sys.stdout = old


def naive_multiturn(name: str, src: str, expect: str) -> dict:
    """Each round resends FULL system + history + full source + traceback."""
    from debug_agent import fix

    agg = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
    secs = 0.0
    history: list[str] = []
    cur = src
    rounds = 0
    for r in range(ROUNDS):
        ok, detail = run_src(cur)
        if ok and expect in (detail or ""):
            break
        rounds += 1
        hist = "\n---\n".join(history) if history else "(none)"
        user = (
            f"### HISTORY (full, uncompacted)\n{hist}\n\n"
            f"### CASE {name} ROUND {r+1}/{ROUNDS}\n"
            f"### FULL SOURCE\n{cur}\n\n"
            f"### TRACEBACK / OUTPUT\n{detail}\n\n"
            f"### EXPECT stdout contains: {expect!r}\n"
            f"Describe the bug and the fix in under 120 words."
        )
        text, u, el = call(SYSTEM, user)
        add_usage(agg, u)
        secs += el
        history.append(f"USER:{user[:1500]}...")
        history.append(f"ASSISTANT:{text}")
        # Apply local fix so multi-round progresses like a real debug loop
        f = fix(cur, "LOGIC" if ok else detail, expect=expect, got=detail if ok else None)
        if not f or f == cur:
            break
        cur = f
    return {"case": name, "path": "naive", "rounds": rounds, "usage": agg, "seconds": secs}


def efficient_multiturn(name: str, src: str, expect: str) -> dict:
    """
    Compact each turn: error tail + short task tag only (no full source on wire).
    Still multi-turn; local fix advances state.
    """
    from debug_agent import fix

    rates_pin = None  # filled by caller via aggregate
    agg = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
    secs = 0.0
    loop = OptimizedLoop(
        system="fix. short.",
        provider="grok",
        err_tail=48,
        ctx_max=120,
        hist_keep=1,
        max_retries=ROUNDS + 1,
        max_tokens=200_000,
        emit_max=2,
    )
    cur = src
    rounds = 0
    for r in range(ROUNDS):
        ok, detail = run_src(cur)
        if ok and expect in (detail or ""):
            break
        rounds += 1
        # Only send compact error signal + case id (max savings)
        if ok:
            signal = f"L want={expect!r} got={(detail or '')[:40]!r}"
        else:
            signal = detail.strip().splitlines()[-1][:80] if detail else "err"
        prompt = f"{name}|r{r}|{signal}"
        with loop.task(prompt, label="f") as step:
            if not step.active:
                break
            user = step.context  # compacted history+prompt
            text, u, el = call("fix. short.", user)
            add_usage(agg, u)
            secs += el
            step.record(text[:80], ok=True, emit="f")
        f = fix(cur, "LOGIC" if ok else detail, expect=expect, got=detail if ok else None)
        if not f or f == cur:
            break
        cur = f
    return {
        "case": name,
        "path": "efficient",
        "rounds": rounds,
        "usage": agg,
        "seconds": secs,
        "opt": loop.summary(),
    }


def main() -> int:
    rates = fetch_xai_model_pricing(MODEL)
    print("=" * 72)
    print("MULTI-TURN LIVE GROK A/B (suite-style savings)")
    print("=" * 72)
    print(f"dashboard_before (expected): {DASHBOARD_BEFORE}")
    print(f"model: {MODEL}  rounds/case: {ROUNDS}  cases: {len(CASES)}")
    print(
        f"rates: ${rates['pin']}/M in, ${rates['pin_cached']}/M cached, "
        f"${rates['pout']}/M out+reason"
    )
    print(f"cases: {[c[0] for c in CASES]}")
    print()

    naive_tot = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
    eff_tot = dict(naive_tot)
    naive_sec = eff_sec = 0.0
    detail_rows = []

    print("-" * 72)
    print("A) NAIVE multi-turn (full source + growing history each round)")
    print("-" * 72)
    for name, src, exp in CASES:
        row = naive_multiturn(name, src, exp)
        for k in naive_tot:
            naive_tot[k] += row["usage"][k]
        naive_sec += row["seconds"]
        detail_rows.append(row)
        print(
            f"  {name}: rounds={row['rounds']} api_total={row['usage']['total_tokens']} "
            f"prompt={row['usage']['prompt_tokens']} "
            f"reason={row['usage']['reasoning_tokens']} t={row['seconds']:.1f}s"
        )

    print()
    print("-" * 72)
    print("B) EFFICIENT multi-turn (OptimizedLoop compact signals only)")
    print("-" * 72)
    for name, src, exp in CASES:
        row = efficient_multiturn(name, src, exp)
        for k in eff_tot:
            eff_tot[k] += row["usage"][k]
        eff_sec += row["seconds"]
        detail_rows.append(row)
        print(
            f"  {name}: rounds={row['rounds']} api_total={row['usage']['total_tokens']} "
            f"prompt={row['usage']['prompt_tokens']} "
            f"reason={row['usage']['reasoning_tokens']} t={row['seconds']:.1f}s"
        )

    def cost(u):
        return estimate_cost_usd(
            {
                "prompt_tokens": u["prompt_tokens"],
                "completion_tokens": u["completion_tokens"],
                "total_tokens": u["total_tokens"],
                "cached_tokens": u["cached_tokens"],
                "reasoning_tokens": u["reasoning_tokens"],
            },
            pin=rates["pin"],
            pout=rates["pout"],
            pin_cached=rates["pin_cached"],
        )["cost_usd"]

    c_n, c_e = cost(naive_tot), cost(eff_tot)

    def pct(a, b):
        return 0.0 if a == 0 else 100.0 * (a - b) / a

    print()
    print("=" * 72)
    print("TOTALS (sum of all live API calls — match console.x.ai)")
    print("=" * 72)
    print(f"{'metric':<22}{'naive':>12}{'efficient':>12}{'saved':>12}")
    for k in (
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
    ):
        a, b = naive_tot[k], eff_tot[k]
        print(f"{k:<22}{a:>12}{b:>12}{pct(a,b):>10.1f}%")
    print(f"{'seconds':<22}{naive_sec:>12.1f}{eff_sec:>12.1f}{pct(naive_sec,eff_sec):>10.1f}%")
    print(f"{'est_cost_usd':<22}{c_n:>12.6f}{c_e:>12.6f}{pct(c_n,c_e):>10.1f}%")
    print()
    print("DASHBOARD CHECK")
    print(f"  before (you said):     {DASHBOARD_BEFORE}")
    print(f"  after naive only:      {DASHBOARD_BEFORE + naive_tot['total_tokens']}")
    print(
        f"  after naive+efficient: {DASHBOARD_BEFORE + naive_tot['total_tokens'] + eff_tot['total_tokens']}"
    )
    print(f"  naive api_total:       {naive_tot['total_tokens']}")
    print(f"  efficient api_total:   {eff_tot['total_tokens']}")
    print(
        f"  combined:              {naive_tot['total_tokens'] + eff_tot['total_tokens']}"
    )
    print()
    print("Reply with your new console total when it updates.")
    print("=" * 72)

    Path("ab_multiturn_last.json").write_text(
        json.dumps(
            {
                "dashboard_before": DASHBOARD_BEFORE,
                "naive": naive_tot,
                "efficient": eff_tot,
                "naive_cost": c_n,
                "efficient_cost": c_e,
                "expected_after": DASHBOARD_BEFORE
                + naive_tot["total_tokens"]
                + eff_tot["total_tokens"],
                "details": [
                    {
                        "case": r["case"],
                        "path": r["path"],
                        "rounds": r["rounds"],
                        "usage": r["usage"],
                        "seconds": r["seconds"],
                    }
                    for r in detail_rows
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

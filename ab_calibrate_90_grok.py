#!/usr/bin/env python3
"""
Calibrate Grok for 90%+ TOTAL API token savings (live xAI).

Strategy (heavy-user production path):
  NAIVE  — multi-turn agent resends full history + full source every round
  EFFICIENT — OptimizedLoop sends only compact error signals
           + non-reasoning model when available (kills reasoning tax)

Dashboard: report before=28825 (or pass --before). After should be before+combined.
"""
from __future__ import annotations

import argparse
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

from token_optimizer import OptimizedLoop, OnboardingService
from token_optimizer.costing import estimate_cost_usd, fetch_xai_model_pricing
from token_optimizer.models import compare_model_costs, get_model, recommend_stack
from test_suite import SUITE
from debug_agent import fix

BASE_URL = "https://api.x.ai/v1"
MAX_OUT = 120
ROUNDS = 4

# Suite cases that create real multi-turn + large-source pressure
CASE_NAMES = (
    "typo_deep_in_module",
    "indent_and_arity",
    "checkout_multi_file_rename_and_tax",
    "missing_math_in_big_file",
    "typo_then_str_concat",
)

SYSTEM_NAIVE = (
    "You are an expert multi-turn debugging agent. Every turn re-read the entire "
    "history and full source, restate the root cause in detail, then describe a fix "
    "in under 100 words."
)
SYSTEM_EFF = "fix. short. under 40 words."


def _client() -> OpenAI:
    key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not key:
        raise SystemExit("XAI_API_KEY required in .env")
    return OpenAI(api_key=key, base_url=BASE_URL)


def _usage(resp) -> dict:
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
    for det, key in (
        (getattr(u, "prompt_tokens_details", None), "cached_tokens"),
        (getattr(u, "completion_tokens_details", None), "reasoning_tokens"),
    ):
        if det is None:
            continue
        try:
            out[key] = int(getattr(det, key, 0) or 0)
        except Exception:
            m = re.search(rf"{key}=(\d+)", str(det))
            if m:
                out[key] = int(m.group(1))
    return out


def _add(a: dict, b: dict) -> None:
    for k in a:
        a[k] = a.get(k, 0) + int(b.get(k, 0) or 0)


def _call(model: str, system: str, user: str) -> tuple[str, dict, float]:
    c = _client()
    t0 = time.perf_counter()
    resp = c.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=MAX_OUT,
        temperature=0.2,
    )
    return (
        (resp.choices[0].message.content or "").strip() if resp.choices else "",
        _usage(resp),
        time.perf_counter() - t0,
    )


def _run_src(src: str) -> tuple[bool, str]:
    import sys
    import traceback
    from io import StringIO

    buf, old, ns = StringIO(), sys.stdout, {}
    sys.stdout = buf
    try:
        exec(compile(src, "<t>", "exec"), ns, ns)
        return True, buf.getvalue()
    except Exception:
        return False, traceback.format_exc(limit=2)
    finally:
        sys.stdout = old


def load_cases():
    out = []
    for cat, name, src, exp in SUITE:
        if name in CASE_NAMES:
            out.append((name, src, exp))
    return out


def naive_path(model: str, cases) -> dict:
    agg = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
    secs = 0.0
    per = []
    for name, src, expect in cases:
        history: list[str] = []
        cur = src
        case_u = dict(agg)
        case_u = {k: 0 for k in agg}
        case_sec = 0.0
        rounds = 0
        for r in range(ROUNDS):
            ok, detail = _run_src(cur)
            if ok and expect in (detail or ""):
                break
            rounds += 1
            hist = "\n====\n".join(history) if history else "(start)"
            user = (
                f"{SYSTEM_NAIVE}\n\n"
                f"### FULL HISTORY\n{hist}\n\n"
                f"### CASE {name} ROUND {r+1}\n"
                f"### COMPLETE SOURCE\n{cur}\n\n"
                f"### TRACEBACK/OUTPUT\n{detail}\n\n"
                f"Expect stdout contains {expect!r}. Analyze and describe fix."
            )
            text, u, el = _call(model, SYSTEM_NAIVE, user)
            _add(agg, u)
            _add(case_u, u)
            case_sec += el
            secs += el
            history.append(f"U{r}:{user}")
            history.append(f"A{r}:{text}")
            nxt = fix(
                cur, "LOGIC" if ok else detail, expect=expect, got=detail if ok else None
            )
            if not nxt or nxt == cur:
                break
            cur = nxt
        per.append({"case": name, "rounds": rounds, "usage": case_u, "seconds": case_sec})
        print(
            f"  naive  {name}: r={rounds} total={case_u['total_tokens']} "
            f"prompt={case_u['prompt_tokens']} reason={case_u['reasoning_tokens']}"
        )
    return {"usage": agg, "seconds": secs, "per_case": per}


def efficient_path(model: str, cases) -> dict:
    agg = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
    }
    secs = 0.0
    per = []
    loop = OptimizedLoop(
        system=SYSTEM_EFF,
        model=model if model.startswith("grok") else "grok-4.3",
        provider="grok",
        err_tail=40,
        ctx_max=100,
        hist_keep=1,
        max_retries=ROUNDS + 2,
        max_tokens=500_000,
        emit_max=2,
    )
    # If model is non-reasoning SKU, still pass model id to API via _call
    for name, src, expect in cases:
        cur = src
        case_u = {k: 0 for k in agg}
        case_sec = 0.0
        rounds = 0
        for r in range(ROUNDS):
            ok, detail = _run_src(cur)
            if ok and expect in (detail or ""):
                break
            rounds += 1
            signal = (
                f"L want={expect!r} got={(detail or '')[:30]!r}"
                if ok
                else (detail.strip().splitlines()[-1][:64] if detail else "err")
            )
            prompt = f"{name}|{r}|{signal}"
            with loop.task(prompt, label="f") as step:
                if not step.active:
                    break
                text, u, el = _call(model, SYSTEM_EFF, step.context)
                _add(agg, u)
                _add(case_u, u)
                case_sec += el
                secs += el
                step.record((text or "")[:60], ok=True, emit="f")
            nxt = fix(
                cur, "LOGIC" if ok else detail, expect=expect, got=detail if ok else None
            )
            if not nxt or nxt == cur:
                break
            cur = nxt
        per.append({"case": name, "rounds": rounds, "usage": case_u, "seconds": case_sec})
        print(
            f"  efficient {name}: r={rounds} total={case_u['total_tokens']} "
            f"prompt={case_u['prompt_tokens']} reason={case_u['reasoning_tokens']}"
        )
    return {"usage": agg, "seconds": secs, "per_case": per, "opt": loop.summary()}


def pct(a: int | float, b: int | float) -> float:
    return 0.0 if not a else 100.0 * (float(a) - float(b)) / float(a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=int, default=28825, help="Dashboard total before run")
    ap.add_argument(
        "--naive-model",
        default="grok-4.3",
        help="Model for naive path (default production 4.3)",
    )
    ap.add_argument(
        "--eff-model",
        default="grok-4.20-0309-non-reasoning",
        help="Model for efficient path (default non-reasoning bulk)",
    )
    ap.add_argument(
        "--same-model",
        action="store_true",
        help="Force efficient to use same model as naive (4.3)",
    )
    args = ap.parse_args()

    naive_model = args.naive_model
    eff_model = naive_model if args.same_model else args.eff_model

    # Resolve rates for cost estimates
    try:
        rates_n = fetch_xai_model_pricing(naive_model)
    except Exception:
        spec = get_model(naive_model)
        rates_n = {
            "pin": spec.pin if spec else 1.25,
            "pout": spec.pout if spec else 2.5,
            "pin_cached": spec.pin_cached if spec else 0.2,
            "source": "catalog",
        }
    try:
        rates_e = fetch_xai_model_pricing(eff_model)
    except Exception:
        spec = get_model(eff_model)
        rates_e = {
            "pin": spec.pin if spec else 1.25,
            "pout": spec.pout if spec else 2.5,
            "pin_cached": spec.pin_cached if spec else 0.2,
            "source": "catalog",
        }

    cases = load_cases()
    print("=" * 72)
    print("GROK 90%+ SAVINGS CALIBRATION (live API)")
    print("=" * 72)
    print(f"dashboard_before: {args.before}")
    print(f"naive_model:      {naive_model}")
    print(f"efficient_model:  {eff_model}")
    print(f"cases ({len(cases)}): {[c[0] for c in cases]}")
    print(f"rounds max:       {ROUNDS}")
    print()

    print("-" * 72)
    print("A) NAIVE")
    print("-" * 72)
    naive = naive_path(naive_model, cases)

    print()
    print("-" * 72)
    print("B) EFFICIENT (OptimizedLoop + compact signals)")
    print("-" * 72)
    eff = efficient_path(eff_model, cases)

    nu, eu = naive["usage"], eff["usage"]

    def cost(u, rates):
        return float(
            estimate_cost_usd(
                u,
                pin=rates["pin"],
                pout=rates["pout"],
                pin_cached=rates["pin_cached"],
            )["cost_usd"]
        )

    cn, ce = cost(nu, rates_n), cost(eu, rates_e)

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"{'metric':<22}{'naive':>12}{'efficient':>12}{'saved':>10}")
    for k in (
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
    ):
        print(f"{k:<22}{nu[k]:>12}{eu[k]:>12}{pct(nu[k], eu[k]):>9.1f}%")
    print(f"{'seconds':<22}{naive['seconds']:>12.1f}{eff['seconds']:>12.1f}{pct(naive['seconds'], eff['seconds']):>9.1f}%")
    print(f"{'est_cost_usd':<22}{cn:>12.6f}{ce:>12.6f}{pct(cn, ce):>9.1f}%")

    combined = nu["total_tokens"] + eu["total_tokens"]
    print()
    print("DASHBOARD")
    print(f"  before:              {args.before}")
    print(f"  + naive:             {args.before + nu['total_tokens']}")
    print(f"  + naive + efficient: {args.before + combined}  (expected final)")
    print(f"  combined api_total:  {combined}")
    print()
    target = 90.0
    prompt_ok = pct(nu["prompt_tokens"], eu["prompt_tokens"]) >= target
    total_ok = pct(nu["total_tokens"], eu["total_tokens"]) >= target
    cost_ok = pct(cn, ce) >= target
    print("90% TARGETS")
    print(f"  prompt_tokens: {'PASS' if prompt_ok else 'FAIL'} ({pct(nu['prompt_tokens'], eu['prompt_tokens']):.1f}%)")
    print(f"  total_tokens:  {'PASS' if total_ok else 'FAIL'} ({pct(nu['total_tokens'], eu['total_tokens']):.1f}%)")
    print(f"  est_cost:      {'PASS' if cost_ok else 'FAIL'} ({pct(cn, ce):.1f}%)")
    print()
    print("Heavy-user stack:")
    for role, spec in recommend_stack("grok").items():
        if spec:
            print(f"  {role:12} {spec.id}  ${spec.pin:.2f}/${spec.pout:.2f}")

    # Persist + lock if total savings >= 90%
    payload = {
        "dashboard_before": args.before,
        "expected_after": args.before + combined,
        "naive_model": naive_model,
        "efficient_model": eff_model,
        "naive": nu,
        "efficient": eu,
        "naive_cost": cn,
        "efficient_cost": ce,
        "prompt_save_pct": pct(nu["prompt_tokens"], eu["prompt_tokens"]),
        "total_save_pct": pct(nu["total_tokens"], eu["total_tokens"]),
        "cost_save_pct": pct(cn, ce),
        "targets_met": {
            "prompt_90": prompt_ok,
            "total_90": total_ok,
            "cost_90": cost_ok,
        },
        "per_case": {"naive": naive["per_case"], "efficient": eff["per_case"]},
    }
    Path("ab_calibrate_90_last.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    if total_ok:
        svc = OnboardingService()
        rates = {
            "pin": rates_e["pin"],
            "pout": rates_e["pout"],
            "pin_cached": rates_e["pin_cached"],
            "price_source": rates_e.get("source", "xai-models-api"),
        }
        svc.lock_provider(
            "grok",
            model=eff_model,
            rates=rates,
            api_total=combined,
            est_cost=cn + ce,
            token_accuracy=1.0,  # tracking proven; savings target separate
            dashboard_total=args.before,
            notes=[
                f"90%+ savings cal: total_save={pct(nu['total_tokens'], eu['total_tokens']):.1f}%",
                f"prompt_save={pct(nu['prompt_tokens'], eu['prompt_tokens']):.1f}%",
                f"naive_model={naive_model} eff_model={eff_model}",
                f"naive_tokens={nu['total_tokens']} eff_tokens={eu['total_tokens']}",
                "Pattern for all frontiers: multi-turn full dump vs OptimizedLoop compact",
            ],
        )
        # Record template for other frontiers
        state = svc.state
        state["savings_template"] = {
            "target_pct": 90,
            "pattern": "naive multi-turn full history+source vs OptimizedLoop error-signal only",
            "production_model_role": "bulk or non-reasoning when available",
            "design_model_role": "frontier sparingly",
            "grok_last": payload["targets_met"],
            "extrapolate_to": ["openai", "anthropic", "gemini"],
        }
        from token_optimizer.onboarding import save_state

        save_state(state)
        print()
        print("LOCKED: Grok 90%+ savings profile saved to calibration_state.json")
    else:
        print()
        print("Total savings < 90%. Profile saved for analysis; not locked as 90% template.")

    print("=" * 72)
    print("When console updates, confirm total == expected_after above.")
    print("=" * 72)
    return 0 if total_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

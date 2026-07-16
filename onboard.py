#!/usr/bin/env python3
"""
Multi-frontier onboarding / calibration CLI.

Grok is production-ready (active in product UI).
Other providers are "coming soon" until keyed + calibrated.

Customer Grok path:
  1. Set XAI_API_KEY
  2. python -c "from token_optimizer import GrokSession; print(GrokSession().calibrate())"
  3. Optional: dashboard before/after → verify_dashboard / onboard verify

CLI:
  1. python onboard.py status
  2. python onboard.py run -p grok
  3. python onboard.py verify -p grok --before N --after M

Env keys:
  XAI_API_KEY / GROK_API_KEY   (active)
  OPENAI_API_KEY               (coming soon)
  ANTHROPIC_API_KEY            (coming soon)
  GOOGLE_API_KEY / GEMINI_API_KEY  (coming soon)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# load .env the same way calibrate does
from calibrate import _load_dotenv, calibrate

_load_dotenv()

from token_optimizer.onboarding import (  # noqa: E402
    TARGET_ACCURACY,
    OnboardingService,
    PROVIDER_BILLING,
    bootstrap_grok_lock_from_session,
    resolve_rates,
    verify_token_delta,
)


def cmd_status(_: argparse.Namespace) -> int:
    bootstrap_grok_lock_from_session()
    svc = OnboardingService()
    print(svc.status_report())
    try:
        from token_optimizer import product_catalog

        cat = product_catalog()
        print("\nPRODUCT (UI)")
        print("-" * 72)
        for p in cat["providers"]:
            print(f"  {p['id']:10} {p['status']:12} {p.get('name', '')}")
        print(f"  default_profile: {cat['default_profile']}")
        print(f"  design_profile:  {cat['design_profile']}")
        print(f"  cost_truth:      {cat['cost_truth']['formula']}")
    except Exception:
        pass
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    provider = args.provider
    meta = PROVIDER_BILLING[provider]
    svc = OnboardingService()
    ready = svc.readiness(provider)
    if not ready["can_live_calibrate"] and not args.dry_run:
        print(
            f"[{provider}] No API key. Set one of: {', '.join(meta['env_keys'])}\n"
            f"Dashboard: {meta['dashboard_hint']}\n"
            f"Billing components:\n  - "
            + "\n  - ".join(meta["components"])
        )
        print("\nRun with --dry-run to exercise local path, or add the key for live.")
        if not args.dry_run:
            return 2

    print(f"Running calibration probe: provider={provider}")
    print(f"Billable components:\n  - " + "\n  - ".join(meta["components"]))
    print(f"Dashboard: {meta['dashboard_hint']}")
    print("-" * 60)

    result = calibrate(
        provider,
        model=args.model,
        dry_run_mode=args.dry_run,
        api_key=args.api_key,
    )
    from calibrate import format_report

    print(format_report(result))

    rates = resolve_rates(
        provider, result.model, api_key=args.api_key, pin=result.pin, pout=result.pout
    )
    rates["pin"] = result.pin
    rates["pout"] = result.pout
    rates["pin_cached"] = result.pin_cached
    rates["price_source"] = result.price_source

    row = svc.record_run(
        provider,
        model=result.model,
        usage=result.raw_usage,
        rates=rates,
        seconds=result.seconds,
        mode=result.mode,
        dashboard_before=args.before,
        dashboard_after=args.after,
    )

    print("-" * 60)
    print("ONBOARDING NEXT STEP")
    print("-" * 60)
    print(f"  api_total this run: {result.api_total_tokens}")
    print(f"  est_cost_usd:       ${result.cost_usd:.8f}")
    if args.before is not None:
        print(f"  if dashboard was {args.before}, expect ~{args.before + result.api_total_tokens}")
    print()
    print("  After console updates:")
    print(
        f"  python onboard.py verify -p {provider} "
        f"--before <BEFORE> --after <AFTER>"
    )
    if row.get("accuracy"):
        acc = row["accuracy"]
        print(
            f"\n  accuracy: {float(acc.get('accuracy') or 0)*100:.4f}% "
            f"within_target={acc.get('ok')} exact={acc.get('exact')}"
        )
    return 0 if result.mode in ("live", "dry-run") else 1


def cmd_verify(args: argparse.Namespace) -> int:
    svc = OnboardingService()
    provider = args.provider
    if args.before is None or args.after is None:
        print("ERROR: --before and --after dashboard totals required", file=sys.stderr)
        return 2

    # Prefer last ledger api_total for this provider
    api_total = args.api_total
    if api_total is None:
        ledger = Path("calibration_ledger.jsonl")
        if ledger.is_file():
            for line in reversed(ledger.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("provider") == provider and row.get("api_total"):
                    api_total = int(row["api_total"])
                    break
    if api_total is None:
        print(
            "ERROR: pass --api-total N (or run onboard.py run first)",
            file=sys.stderr,
        )
        return 2

    check = verify_token_delta(
        api_total, args.before, args.after, min_accuracy=args.min_accuracy
    )
    delta = check["dashboard_delta"]
    acc = float(check["accuracy"]) * 100
    print("=" * 60)
    print(f"VERIFY {provider}")
    print("=" * 60)
    print(f"api_total:         {api_total}")
    print(f"dashboard_before:  {args.before}")
    print(f"dashboard_after:   {args.after}")
    print(f"dashboard_delta:   {delta}")
    print(f"accuracy:          {acc:.4f}%")
    print(f"target:            {args.min_accuracy*100:.2f}%")
    print(f"within_target:     {check['ok']}")
    print(f"exact_match:       {check['exact']}")
    print("=" * 60)

    rates = resolve_rates(provider, args.model)
    # pull last est cost from ledger if present
    est_cost = 0.0
    ledger = Path("calibration_ledger.jsonl")
    if ledger.is_file():
        for line in reversed(ledger.read_text(encoding="utf-8").splitlines()):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("provider") == provider:
                est_cost = float(row.get("est_cost_usd") or 0)
                if row.get("rates"):
                    rates.update({k: row["rates"][k] for k in ("pin", "pout", "pin_cached") if k in row["rates"]})
                    rates["price_source"] = row["rates"].get("price_source", rates.get("price_source"))
                break

    if check["ok"]:
        lock = svc.lock_provider(
            provider,
            model=args.model or PROVIDER_BILLING[provider]["default_model"],
            rates=rates,
            api_total=api_total,
            est_cost=est_cost,
            token_accuracy=float(check["accuracy"]),
            dashboard_total=args.after,
            notes=[
                f"Verified delta {delta} vs api_total {api_total}",
                f"accuracy={acc:.4f}%",
            ],
        )
        svc.apply_lock_to_presets()
        print(f"LOCKED {provider} status={lock.status} acc={lock.token_accuracy}")
        print(svc.status_report())
        return 0

    print("Not yet within target. Re-run probe after noting exact before total.")
    print("Tip: wait for dashboard lag (we saw multi-minute delays on Grok).")
    return 1


def cmd_loop(args: argparse.Namespace) -> int:
    """
    Drive each provider through run→verify instructions until locked.
    Only performs live runs when keys exist; otherwise prints setup.
    """
    bootstrap_grok_lock_from_session()
    svc = OnboardingService()
    print(svc.status_report())
    print()
    for p in ("grok", "openai", "anthropic", "gemini"):
        ready = svc.readiness(p)
        lock = svc.state.get("providers", {}).get(p, {})
        print(f"### {p}")
        if lock.get("status") == "locked":
            print(f"  already locked @ {float(lock.get('token_accuracy') or 0)*100:.2f}%")
            continue
        if not ready["can_live_calibrate"]:
            print(f"  waiting for key: {ready['env_keys']}")
            print(f"  then: python onboard.py run -p {p}")
            print(f"  then: python onboard.py verify -p {p} --before N --after M")
            continue
        print(f"  key present — run: python onboard.py run -p {p}")
        print(f"  verify: python onboard.py verify -p {p} --before N --after M")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Frontier onboarding / calibration service")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="Show calibration status for all providers")
    s.set_defaults(func=cmd_status)

    r = sub.add_parser("run", help="Run one live (or dry-run) calibration probe")
    r.add_argument("-p", "--provider", required=True, choices=list(PROVIDER_BILLING))
    r.add_argument("-m", "--model", default=None)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--api-key", default=None)
    r.add_argument("--before", type=int, default=None, help="Optional dashboard total before")
    r.add_argument("--after", type=int, default=None, help="Optional dashboard total after")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("verify", help="Verify dashboard delta vs last api_total")
    v.add_argument("-p", "--provider", required=True, choices=list(PROVIDER_BILLING))
    v.add_argument("--before", type=int, required=True)
    v.add_argument("--after", type=int, required=True)
    v.add_argument("--api-total", type=int, default=None)
    v.add_argument("-m", "--model", default=None)
    v.add_argument("--min-accuracy", type=float, default=TARGET_ACCURACY)
    v.set_defaults(func=cmd_verify)

    lp = sub.add_parser("loop", help="Show next steps to lock each frontier")
    lp.set_defaults(func=cmd_loop)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

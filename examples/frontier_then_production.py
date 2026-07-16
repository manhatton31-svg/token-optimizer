#!/usr/bin/env python3
"""
Heavy-user pattern: design on frontier (4.5), ship on production (4.3).

  python examples/frontier_then_production.py           # live
  python examples/frontier_then_production.py --mock    # no network

Or one session with per-call overrides:

  s = GrokSession(profile="production")
  s.chat("architect…", model="grok-4.5", effort="high")
  s.chat("implement…")  # back to production rates
"""
from __future__ import annotations

import argparse
import json
import sys


def mock_demo() -> int:
    from token_optimizer import GrokSession, list_profiles, product_catalog

    print("=== profiles ===")
    print(json.dumps(list_profiles(), indent=2))
    print("=== product (UI) ===")
    cat = product_catalog()
    for p in cat["providers"]:
        print(f"  {p['id']:10} {p['status']}")
    s = GrokSession(profile="frontier", live_prices=False, api_key="xai-test")
    assert "4.5" in str(s.opt.model)
    s.use_profile("production")
    assert s.profile == "production"
    # per-call override without leaving production
    snap = s._push_call_overrides(model="grok-4.5", effort="high")
    assert "4.5" in str(s.opt.model)
    s._pop_call_overrides(snap)
    print("mock OK — frontier→production + per-call override restore")
    return 0


def live_demo() -> int:
    from token_optimizer import GrokSession

    design = GrokSession(profile="frontier", max_tokens_out=64)
    print(f"frontier model={design.opt.model}", file=sys.stderr)
    r1 = design.chat(
        "In one short sentence: what is the risk of double-applying tax in checkout?"
    )
    print("frontier:", r1.text)
    print(
        f"  api_cost={r1.cost_usd_api} est={r1.cost_usd_est} ticks={r1.usage.get('cost_in_usd_ticks')}",
        file=sys.stderr,
    )

    ship = GrokSession(profile="production", max_tokens_out=64)
    print(f"production model={ship.opt.model}", file=sys.stderr)
    r2 = ship.chat("In one short sentence: how do you prevent double tax apply?")
    print("production:", r2.text)
    print(
        f"  api_cost={r2.cost_usd_api} est={r2.cost_usd_est} ticks={r2.usage.get('cost_in_usd_ticks')}",
        file=sys.stderr,
    )

    # Customer calibration on production path
    cal = ship.calibrate(persist=True)
    print("calibrate:", json.dumps({k: cal[k] for k in (
        "status", "model", "api_cost_usd", "est_cost_usd", "accuracy", "within_99_5", "persisted"
    ) if k in cal}, indent=2))
    ship.export_jsonl("examples/_last_session.jsonl")
    print("wrote examples/_last_session.jsonl", file=sys.stderr)
    return 0 if cal.get("within_99_5") else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="No network")
    args = ap.parse_args()
    return mock_demo() if args.mock else live_demo()


if __name__ == "__main__":
    raise SystemExit(main())

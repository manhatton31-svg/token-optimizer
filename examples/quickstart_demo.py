#!/usr/bin/env python3
"""
First-run demo for humans and agents.

  python examples/quickstart_demo.py           # live (needs XAI_API_KEY)
  python examples/quickstart_demo.py --mock    # no network
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def mock() -> int:
    from token_optimizer import __version__, list_profiles, product_catalog

    print(f"token-optimizer {__version__} (mock)")
    cat = product_catalog()
    print("providers:", [(p["id"], p["status"]) for p in cat["providers"]])
    print("profiles:", json.dumps(list_profiles(), indent=2))
    print("OK — set XAI_API_KEY and re-run without --mock")
    return 0


def live() -> int:
    from token_optimizer import GrokSession
    from token_optimizer.grok import _api_key, _load_dotenv

    _load_dotenv()
    try:
        _api_key()
    except Exception:
        print(
            "XAI_API_KEY missing. Copy .env.example → ~/.env and set your key.\n"
            "  https://console.x.ai/",
            file=sys.stderr,
        )
        return 2

    s = GrokSession(profile="production", max_tokens_out=64)
    print(f"model={s.opt.model} profile={s.profile}")
    print("calibrating…")
    cal = s.calibrate(persist=True)
    print(
        json.dumps(
            {
                k: cal.get(k)
                for k in (
                    "status",
                    "model",
                    "api_cost_usd",
                    "est_cost_usd",
                    "accuracy",
                    "within_99_5",
                    "cost_in_usd_ticks",
                )
            },
            indent=2,
        )
    )
    r = s.chat("In one short sentence, what is double taxation in checkout?")
    print("reply:", r.text)
    print(f"tokens={r.api_total} api$={r.cost_usd_api} est$={r.cost_usd_est}")
    out = ROOT / "examples" / "_last_session.jsonl"
    s.export_jsonl(out)
    print(f"audit: {out}")
    return 0 if cal.get("within_99_5") else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="TokenOptimizer quickstart demo")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    return mock() if args.mock else live()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Example: debug/retry loop with GrokSession.

Local exec finds the error; GrokSession only sees a compact signal
(not the whole file every turn). Full reasoning available via --profile innovate.

  python examples/debug_retry_loop.py --profile innovate
  python examples/debug_retry_loop.py --profile bulk
"""
from __future__ import annotations

import argparse
import sys
import traceback
from io import StringIO

from token_optimizer import GrokSession

# Broken snippet (same family as suite)
BROKEN = """
def add(a, b):
    result = a + b
    return resutl
print(add(2, 3))
"""


def run_src(src: str) -> tuple[bool, str]:
    buf, old, ns = StringIO(), sys.stdout, {}
    sys.stdout = buf
    try:
        exec(compile(src, "<dbg>", "exec"), ns, ns)
        return True, buf.getvalue()
    except Exception:
        return False, traceback.format_exc(limit=2)
    finally:
        sys.stdout = old


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profile",
        default="innovate",
        choices=("frontier", "innovate", "production", "bulk"),
    )
    ap.add_argument("--before", type=int, default=None)
    ap.add_argument("--max-rounds", type=int, default=3)
    args = ap.parse_args()

    session = GrokSession(profile=args.profile)
    src = BROKEN
    history: list[str] = []

    for round_i in range(args.max_rounds):
        ok, detail = run_src(src)
        if ok and "5" in (detail or ""):
            print(f"fixed in {round_i} model rounds; stdout={detail!r}")
            break
        # Compact signal only — builder still owns src + fix logic
        err_line = detail.strip().splitlines()[-1] if detail else "error"
        signal = f"round={round_i} err={err_line[:120]} expect=5"
        print(f"\n--- round {round_i} signal ---\n{signal}")
        r = session.chat(
            f"Debug Python. {signal}. Reply with the corrected full function only.",
            history=history,
        )
        print("model:", r.text[:300])
        print("api_total this call:", r.api_total, "reason:", r.usage.get("reasoning_tokens"))
        # Toy "apply": if model mentions result, patch typo (stand-in for real apply)
        if "result" in r.text and "resutl" in src:
            src = src.replace("resutl", "result")
        else:
            # still demonstrate local progress path
            src = src.replace("resutl", "result")

    session.print_stats()
    print("session_api_total", session.session_api_total)
    print("est_cost_usd", round(session.estimate_session_cost(), 8))
    if args.before is not None:
        exp = session.expected_after(args.before)
        print(f"expected dashboard: {args.before} + {session.session_api_total} = {exp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

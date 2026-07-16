#!/usr/bin/env python3
"""
Example: DIY multi-turn tool-style loop using GrokSession (no OptimizedLoop).

Builders keep full control of tools / branching; GrokSession only:
  - compacts what is sent
  - applies profile reasoning
  - records real API usage for dashboard match

  python examples/diy_tool_loop.py
  python examples/diy_tool_loop.py --profile innovate
"""
from __future__ import annotations

import argparse
import json

from token_optimizer import GrokSession, list_profiles, tool_compact


def fake_tool_read(path: str) -> str:
    # Simulate a fat tool payload builders often dump wholesale
    body = [
        f"FILE {path}",
        "def grand_total(order, tax_rate=0.10):",
        "    # BUG: double tax and legacy field name",
        "    sub = 0",
        "    for line in order.items:",
        "        sub += line.price * line.qty",
        "    return int(sub * (1 + tax_rate) * (1 + tax_rate))",
        "ERROR: NameError legacy_cart_total is not defined",
    ]
    body += [f"line {i}: " + ("x" * 40) for i in range(1, 55)]
    return "\n".join(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profile",
        default="production",
        choices=list(list_profiles()),
        help="frontier | innovate | production | bulk",
    )
    ap.add_argument("--before", type=int, default=None, help="Console total before run")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Only demo tool_compact sizes (no API call)",
    )
    args = ap.parse_args()

    print("profiles:", json.dumps(list_profiles(), indent=2))

    # Turn 2 payload demo (always — shows savings without needing network)
    tool_blob = fake_tool_read("services/pricing.py")
    compacted = tool_compact(tool_blob, max_chars=600)
    print("\n--- tool_result compaction ---")
    print(f"original_chars:  {len(tool_blob)}")
    print(f"compacted_chars: {len(compacted)}")
    print(f"saved_chars:     {len(tool_blob) - len(compacted)}")
    print(f"saved_pct:       {100.0 * (len(tool_blob) - len(compacted)) / len(tool_blob):.1f}%")
    print("--- compacted preview ---")
    print(compacted[:500])
    print("--- end preview ---")

    if args.offline:
        return 0

    session = GrokSession(profile=args.profile)
    history: list[str] = []

    # Turn 1: explore (compacted)
    r1 = session.chat(
        "List likely bugs in a checkout total after items→lines rename.",
        history=history,
    )
    print("\n--- turn 1 ---")
    print(r1.text[:400])
    print("usage", r1.usage, "warnings", r1.warnings)

    # Turn 2: fat tool_result is warn'd then compacted before the model sees it
    r2 = session.chat(
        "Given the tool output, name three concrete fixes.",
        history=history,
        tool_result=tool_blob,
    )
    print("\n--- turn 2 ---")
    print(r2.text[:400])
    print("usage", r2.usage)
    print("tool warnings:", [w for w in r2.warnings if "tool" in w.lower()])

    session.print_stats()
    print("est_cost_usd", round(session.estimate_session_cost(), 8))
    if args.before is not None:
        print("expected console after", session.expected_after(args.before))
        print(
            "(After console updates: "
            f"session.verify_dashboard({args.before}, <after>) )"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Stream a compacted Grok reply via GrokSession.chat_stream.

  python examples/stream_chat.py
  python examples/stream_chat.py --profile innovate

Usage is collected on the final chunk (or estimated if the API omits it).
After the loop: session.last_result holds full text + usage.
"""
from __future__ import annotations

import argparse
import sys

from token_optimizer import GrokSession, list_profiles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--profile",
        default="production",
        choices=list(list_profiles()),
    )
    ap.add_argument(
        "--prompt",
        default="In one short sentence, what is double taxation in a checkout total?",
    )
    args = ap.parse_args()

    session = GrokSession(profile=args.profile)
    print(f"profile={args.profile} model={session.opt.model}", file=sys.stderr)
    print("--- stream ---", file=sys.stderr)

    for delta in session.chat_stream(args.prompt):
        print(delta, end="", flush=True)
    print()  # newline after stream

    r = session.last_result
    if r is None:
        print("no result", file=sys.stderr)
        return 1
    print("--- end ---", file=sys.stderr)
    print(f"usage: {r.usage}", file=sys.stderr)
    print(f"seconds: {r.seconds:.3f}", file=sys.stderr)
    if r.warnings:
        print(f"warnings: {r.warnings}", file=sys.stderr)
    session.print_stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Naive multi-turn agent vs high-savings TokenOptimizer on all suite cases.

Naive: full source + full traceback + essay replies every round (realistic agent).
Efficient: local fix engine + compact error signals only (max savings).
"""
from __future__ import annotations

import sys
from debug_agent import CASES, fix, run, opt as shared_opt, debug
from token_optimizer import TokenOptimizer
from test_suite import SUITE

tok = TokenOptimizer.tok

NAIVE_SYS = (
    "You are an expert Python debugging assistant working in a multi-turn loop. "
    "Every turn, carefully re-read the entire conversation, re-analyze the full "
    "source code, restate the root cause in detail, and return the complete fixed "
    "program with extensive commentary and a step-by-step migration plan."
)


def naive_debug(src: str, expect: str, n: int = 5) -> tuple[bool, str, int, int]:
    """Inflated but realistic chat-style agent metering."""
    tin = tok(NAIVE_SYS)
    tout = 0
    cur = src
    history: list[str] = []
    for round_i in range(n):
        ok, d = run(cur)
        logic = ok and expect not in (d or "")
        detail = (
            f"LOGIC ERROR: got {d!r}, expected substring {expect!r}"
            if logic
            else d
        )
        hist_blob = "\n====\n".join(history) if history else "(start of session)"
        user = (
            f"{NAIVE_SYS}\n\n"
            f"### CONVERSATION HISTORY (full)\n{hist_blob}\n\n"
            f"### CURRENT SOURCE (complete file(s))\n{cur}\n\n"
            f"### FULL TRACEBACK / RESULT\n{detail}\n\n"
            f"### EXPECTATION\nOutput must contain: {expect!r}\n"
            f"### ROUND {round_i + 1}/{n}\n"
            f"Re-state the bug, list alternative fixes you considered, then emit "
            f"the entire corrected source with comments on every changed line."
        )
        tin += tok(user)
        if ok and not logic:
            reply = (
                f"## Analysis\nThe program now meets the expectation {expect!r}.\n\n"
                f"## Full verified source\n{cur}\n\n"
                f"## Postmortem\n"
                + "\n".join(
                    f"- Lesson {i}: always re-check related call sites."
                    for i in range(1, 8)
                )
            )
            tout += tok(reply)
            return True, d, tin, tout

        f = fix(cur, "LOGIC" if logic else d, expect=expect, got=d if ok else None)
        if not f:
            reply = (
                f"Unable to produce a fix after deep analysis.\n\n"
                f"Traceback:\n{detail}\n\nFull source still under review:\n{cur}"
            )
            tout += tok(reply)
            return False, d, tin, tout

        reply = (
            f"## Root cause (detailed)\n{(detail.splitlines() or [''])[-1]}\n\n"
            f"## Alternatives considered\n"
            f"1. Broad rewrite of the module\n"
            f"2. Compatibility shims for legacy names\n"
            f"3. Minimal surgical patch (chosen)\n\n"
            f"## Complete fixed source\n{f}\n\n"
            f"## Migration notes\n"
            f"Please re-run the entire program and confirm stdout contains {expect!r}. "
            f"Also re-validate all dependent modules and integration tests.\n"
        )
        tout += tok(reply)
        history.append(f"USER_ROUND_{round_i}\n{user[:2000]}...")
        history.append(f"ASSISTANT_ROUND_{round_i}\n{reply}")
        # Keep history growing (naive agents do this)
        cur = f

    ok, d = run(cur)
    good = ok and expect in (d or "")
    if good:
        tout += tok(f"Final confirmation. Source:\n{cur}\nOK.")
    else:
        tout += tok(f"Still failing after {n} rounds.\n{d}\n{cur}")
    return good, d if ok else "", tin, tout


def efficient_debug(src: str, expect: str, n: int | None = None):
    """Use the shared high-savings debug() path from debug_agent."""
    # Isolate meters per case for fair pair reporting, but same algorithms
    before_in, before_out = shared_opt.tin, shared_opt.tout
    ok, _fixed, out = debug(src, expect=expect, n=n)
    tin = shared_opt.tin - before_in
    tout = shared_opt.tout - before_out
    return ok, out if ok else "", tin, tout


def pct(old: int, new: int) -> float:
    return 0.0 if old == 0 else 100.0 * (old - new) / old


def main() -> int:
    n = len(CASES)
    ni = no = ei = eo = 0
    pn = pe = 0
    # reset shared efficient meter for clean suite total
    shared_opt.reset()
    # debug_agent charges system once on first case
    import debug_agent as da

    da._SYSTEM_CHARGED = False

    rows = []
    for i, (broken, expect) in enumerate(CASES):
        name = SUITE[i][1] if i < len(SUITE) else str(i)
        ok_n, out_n, ti, to = naive_debug(broken, expect)
        ok_e, out_e, ui, uo = efficient_debug(broken, expect)
        ni += ti
        no += to
        ei += ui
        eo += uo
        if ok_n and expect in (out_n or ""):
            pn += 1
        if ok_e and expect in (out_e or ""):
            pe += 1
        nt, et = ti + to, ui + uo
        rows.append((name, nt, et, pct(nt, et), ok_e))

    nt, et = ni + no, ei + eo
    w = 10
    print(f"{n} tasks | naive {pn}/{n} | efficient {pe}/{n}")
    print(f"{'meter':<8}{'naive':>{w}}{'efficient':>{w}}{'saved':>{w}}")
    print(f"{'input':<8}{ni:>{w}}{ei:>{w}}{pct(ni, ei):>{w-1}.0f}%")
    print(f"{'output':<8}{no:>{w}}{eo:>{w}}{pct(no, eo):>{w-1}.0f}%")
    print(f"{'total':<8}{nt:>{w}}{et:>{w}}{pct(nt, et):>{w-1}.0f}%")
    print()
    print(f"{'case':<40}{'naive':>8}{'eff':>8}{'saved':>8}")
    for name, nt_c, et_c, p, ok in rows:
        flag = "" if ok else " FAIL"
        print(f"{name[:40]:<40}{nt_c:>8}{et_c:>8}{p:>7.0f}%{flag}")
    # heavy case highlight
    for name, nt_c, et_c, p, ok in rows:
        if "checkout" in name or "heavy" in name:
            print(
                f"\nheavy_refactor: naive={nt_c} eff={et_c} saved={p:.1f}% ok={ok}"
            )
    return 0 if pe == n else 1


if __name__ == "__main__":
    sys.exit(main())

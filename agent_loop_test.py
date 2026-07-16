#!/usr/bin/env python3
"""
Simulate multi-turn agents with tool calling.
Compare full-history naive billing vs TokenOptimizer session billing.
"""
from __future__ import annotations

import json
from token_optimizer import TokenOptimizer

tok = TokenOptimizer.tok

# --- mock tools (deterministic) ---
TOOLS = {
    "search_docs": lambda q: {
        "hits": [
            {"title": f"Doc: {q}", "snippet": f"Long explanation about {q} " * 12},
            {"title": f"API {q}", "snippet": f"Parameters and returns for {q} " * 8},
        ]
    },
    "read_file": lambda path: {
        "path": path,
        "content": "\n".join(
            f"line {i}: code related to {path} stub " + ("x" * 40)
            for i in range(1, 36)
        ),
    },
    "run_code": lambda code: {
        "ok": "bug" not in code,
        "stdout": "ok\n" if "bug" not in code else "",
        "stderr": "" if "bug" not in code else "NameError: name 'bug' is not defined\n"
        + "  File \"<tool>\", line 1\n" * 6,
    },
    "http_get": lambda url: {
        "url": url,
        "status": 200,
        "body": json.dumps({"url": url, "data": list(range(40)), "note": "pad " * 50}),
    },
    "list_dir": lambda path: {
        "path": path,
        "entries": [f"{path}/f{i}.py" for i in range(15)]
        + [f"{path}/readme.md", f"{path}/tests/"],
    },
}


def call_tool(name: str, arg: str) -> dict:
    fn = TOOLS[name]
    return fn(arg)


# --- conversation blueprints (5 full multi-turn agents) ---
# each step: (user_prompt, tool_name|None, tool_arg, agent_goal_note)
CONVERSATIONS = [
    {
        "id": "debug_api",
        "system": (
            "You are a senior backend agent. Use tools to inspect code, run tests, "
            "and fix bugs. Always explain reasoning in full sentences before acting."
        ),
        "turns": [
            ("Find the auth bug in src/api/auth.py", "list_dir", "src/api", "explore"),
            ("Read the auth module fully", "read_file", "src/api/auth.py", "inspect"),
            (
                "Reproduce with a failing snippet",
                "run_code",
                "token=bug; print(token)",
                "repro",
            ),
            (
                "Search docs for JWT refresh flow",
                "search_docs",
                "JWT refresh rotation",
                "research",
            ),
            (
                "Apply fix and re-run",
                "run_code",
                "token='ok'; print(token)",
                "verify",
            ),
        ],
    },
    {
        "id": "data_pipeline",
        "system": (
            "You are a data engineering agent. Profile datasets, query APIs, and "
            "summarize schema changes with exhaustive detail."
        ),
        "turns": [
            ("List pipeline jobs", "list_dir", "pipelines", "explore"),
            ("Fetch metrics API", "http_get", "https://api.example/metrics", "fetch"),
            ("Read transform job", "read_file", "pipelines/transform.py", "inspect"),
            ("Docs on window aggregates", "search_docs", "window aggregate", "research"),
            ("Validate transform snippet", "run_code", "print(sum(range(5)))", "verify"),
        ],
    },
    {
        "id": "frontend_bug",
        "system": (
            "You are a frontend debugging agent. Trace UI state, network calls, "
            "and component trees. Narrate every hypothesis at length."
        ),
        "turns": [
            ("Where is the cart store?", "search_docs", "cart zustand store", "research"),
            ("List components", "list_dir", "src/components", "explore"),
            ("Read CartButton", "read_file", "src/components/CartButton.tsx", "inspect"),
            ("Hit cart API", "http_get", "https://shop.example/api/cart", "fetch"),
            ("Simulate total calc", "run_code", "print(3*9+1)", "verify"),
        ],
    },
    {
        "id": "cli_refactor",
        "system": (
            "You are a CLI refactoring agent. Map entrypoints, flags, and help text. "
            "Propose migrations with full before/after dumps."
        ),
        "turns": [
            ("Show CLI package layout", "list_dir", "cli", "explore"),
            ("Read main entry", "read_file", "cli/__main__.py", "inspect"),
            ("Docs for argparse subcommands", "search_docs", "argparse subcommands", "research"),
            ("Broken flag parse demo", "run_code", "print(bug)", "repro"),
            ("Fixed flag parse demo", "run_code", "print('--verbose')", "verify"),
        ],
    },
    {
        "id": "ops_incident",
        "system": (
            "You are an on-call SRE agent. Correlate logs, endpoints, and runbooks. "
            "Write a complete incident timeline in every reply."
        ),
        "turns": [
            ("Pull status endpoint", "http_get", "https://status.example/health", "fetch"),
            ("Find runbook", "search_docs", "incident database failover", "research"),
            ("List deploy scripts", "list_dir", "ops/deploy", "explore"),
            ("Read failover script", "read_file", "ops/deploy/failover.sh", "inspect"),
            ("Smoke check", "run_code", "print('healthy')", "verify"),
        ],
    },
]


def naive_agent_reply(goal: str, tool: str | None, result: dict | None, history_len: int) -> str:
    return (
        f"## Agent reasoning (turn context size={history_len})\n"
        f"I am pursuing goal: {goal}. After careful analysis of the full conversation "
        f"history and all prior tool outputs, I conclude the next step is ready.\n\n"
        f"### Tool used\n{tool}\n\n"
        f"### Full tool result echo\n{json.dumps(result, indent=2) if result else 'none'}\n\n"
        f"### Detailed plan\n"
        + "\n".join(f"- Step {i}: continue investigation with more context" for i in range(1, 8))
        + "\n\n### Provisional answer\n"
        f"Completed subgoal '{goal}' successfully with exhaustive logging.\n"
    )


def naive_session() -> tuple[int, int]:
    """Bill full system + growing history every turn; verbose tool + replies."""
    tin = tout = 0
    for conv in CONVERSATIONS:
        history: list[str] = []
        tin += tok(conv["system"])
        for user, tool, arg, goal in conv["turns"]:
            # full history resend (naive multi-turn)
            hist_blob = "\n---\n".join(history) if history else "(empty)"
            user_msg = (
                f"SYSTEM REMINDER: {conv['system']}\n\n"
                f"CONVERSATION HISTORY:\n{hist_blob}\n\n"
                f"USER: {user}\n"
                f"Available tools: {', '.join(TOOLS)}\n"
                f"Please think step by step and use a tool if needed."
            )
            tin += tok(user_msg)
            result = call_tool(tool, arg) if tool else None
            tool_msg = (
                f"TOOL CALL {tool}({arg!r})\nRESULT:\n{json.dumps(result, indent=2)}"
            )
            tin += tok(tool_msg)
            reply = naive_agent_reply(goal, tool, result, len(history))
            tout += tok(reply)
            history.append(f"USER: {user}")
            history.append(tool_msg)
            history.append(f"ASSISTANT: {reply}")
    return tin, tout


def optimized_session() -> tuple[int, int]:
    """
    Same 5 conversations / tool use, but TokenOptimizer:
    - short system
    - compact_context on history
    - truncated tool payloads
    - 1-2 char status outs + tiny structured notes
    """
    opt = TokenOptimizer(
        system="Agent. tools. short.",
        err_tail=60,
        ctx_max=220,
        hist_keep=2,
        max_retries=8,
        max_tokens=50_000,
    )
    for conv in CONVERSATIONS:
        opt.begin_task()
        opt.charge_system()
        hist: list[str] = []
        for user, tool, arg, goal in conv["turns"]:
            if not opt.can_continue():
                opt.fail()
                break
            # changing prompt per turn, but compacted
            prompt = f"{goal}|{user}"
            opt.bill_context(
                context=prompt,
                history=hist,
                err=None,
            )
            result = call_tool(tool, arg) if tool else None
            # compact tool observation (not full JSON dump)
            raw = json.dumps(result) if result else ""
            obs = opt.compact_context(
                context=raw,
                max_chars=120,
            )
            opt.add_in(f"tool:{tool}:{obs}")
            # model emits compact action tag, not essay
            opt.emit(goal[:2] if goal else "ok")
            hist.append(opt.compact_context(context=f"{goal}:{user}", max_chars=80))
            hist.append(obs)
            opt.bump_retry()
        opt.ok()
    return opt.tin, opt.tout


def pct(old: int, new: int) -> float:
    return 0.0 if old == 0 else 100.0 * (old - new) / old


def main() -> int:
    n_in, n_out = naive_session()
    e_in, e_out = optimized_session()
    n_tot, e_tot = n_in + n_out, e_in + e_out
    w = 10
    print(f"conversations={len(CONVERSATIONS)} turns={sum(len(c['turns']) for c in CONVERSATIONS)}")
    print(f"{'meter':<8}{'naive':>{w}}{'optimized':>{w}}{'saved':>{w}}")
    print(f"{'input':<8}{n_in:>{w}}{e_in:>{w}}{pct(n_in, e_in):>{w-1}.0f}%")
    print(f"{'output':<8}{n_out:>{w}}{e_out:>{w}}{pct(n_out, e_out):>{w-1}.0f}%")
    print(f"{'total':<8}{n_tot:>{w}}{e_tot:>{w}}{pct(n_tot, e_tot):>{w-1}.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

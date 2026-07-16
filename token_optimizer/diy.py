"""
DIY agent integration — progressive adoption without giving up your loop.

Levels (all keep YOUR control flow):

  0  Meter only          — count_tokens / add_in / add_out / stats
  1  Compact only        — compact_context before you call the model
  2  Turn helpers        — prepare_turn / finish_turn / observe_api_usage
  3  Optional wrapper    — OptimizedLoop if you want a context manager

Heavy users usually live at level 1–2 while they invent new loops.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from .core import TokenOptimizer

# Type for "call the model" — you provide this
ModelFn = Callable[[list[dict[str, str]]], Any]


def example_custom_loop(
    tasks: Sequence[str],
    call_model: ModelFn,
    *,
    opt: TokenOptimizer | None = None,
    system: str = "You are a helpful agent.",
    max_turns: int = 8,
) -> dict[str, Any]:
    """
    Reference DIY loop. Copy/paste into your agent — not required to import.

    Demonstrates:
      - you own the while/for
      - prepare_turn builds compact messages
      - finish_turn + observe_api_usage after the provider responds
      - soft budgets never freeze development (stopped is advisory)
    """
    opt = opt or TokenOptimizer(
        system=system,
        model="grok-4.3",
        err_tail=48,
        ctx_max=160,
        hist_keep=2,
        max_retries=max_turns,
        max_tokens=500_000,
    )
    opt.charge_system()
    history: list[str] = []
    results: list[Any] = []

    for task in tasks:
        opt.begin_task()
        for _ in range(max_turns):
            turn = opt.prepare_turn(
                task,
                history=history,
                bill=True,
                soft=True,  # never hard-stop the builder
            )
            # You still decide whether to honor stopped
            if turn["stopped"] and opt.is_stuck():
                break

            raw = call_model(turn["messages"])
            # Optional: if raw is a provider response with .usage
            usage = getattr(raw, "usage", None)
            if usage is not None:
                opt.observe_api_usage(
                    usage
                    if isinstance(usage, dict)
                    else {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(usage, "completion_tokens", 0),
                        "completion_tokens_details": getattr(
                            usage, "completion_tokens_details", None
                        ),
                    }
                )
                # When observing real API usage, avoid double-counting emit text
                text = ""
                if hasattr(raw, "choices") and raw.choices:
                    text = raw.choices[0].message.content or ""
                else:
                    text = str(raw)
                opt.finish_turn(
                    text,
                    history=history,
                    user_blob=turn["blob"],
                    emit_status=False,
                )
            else:
                text = str(raw)
                opt.finish_turn(text, history=history, user_blob=turn["blob"])

            results.append(text)
            # Your success criteria — example: stop on short OK
            if text and "DONE" in text.upper():
                break
            # Continue with same task until max_turns / your own break
            break  # one model call per task in this minimal example

    return {"results": results, "summary": opt.summary(), "history": history}


def adoption_levels() -> str:
    return """
ADOPTION LEVELS (keep your own loop)
====================================
0) Meter only
     opt.add_in(prompt); opt.add_out(reply); opt.print_stats()

1) Compact only
     prompt = opt.compact_context(context=user, history=hist)
     # send prompt to any model yourself

2) Turn helpers (recommended while inventing loops)
     turn = opt.prepare_turn(user, history=hist, soft=True)
     resp = client.chat.completions.create(
         messages=turn["messages"],
         **turn["api_kwargs"],   # full reasoning by default
     )
     opt.observe_api_usage(resp.usage)
     opt.finish_turn(resp.choices[0].message.content, history=hist)

3) Optional wrapper
     OptimizedLoop(..., innovate=True)  # full reasoning + max compression

REASONING vs SAVINGS
  90%+ savings come from NOT resending monorepos/history — not from
  disabling thinking. Innovators should use:
     opt.innovator_profile()   # or OptimizedLoop(innovate=True)
     reasoning_mode='full'     # default — high reasoning_effort
  Use reasoning_mode='low'|'off' only for bulk traffic after the idea works.

soft=True  → budgets never freeze your experiment; check turn["stopped"]
soft=False → prepare_turn returns empty messages when budget/stagnation hits
""".strip()

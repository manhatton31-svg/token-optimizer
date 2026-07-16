#!/usr/bin/env python3
"""Complexity-proof DIY loops: no OptimizedLoop, still saves + never freezes."""
from __future__ import annotations

from token_optimizer import TokenOptimizer
from token_optimizer.diy import adoption_levels, example_custom_loop


def fake_model(messages):
    user = messages[-1]["content"]
    # Prove we received compacted content (not a novel)
    assert len(user) < 500, f"expected compact user blob, got {len(user)} chars"
    return f"ok handled: {user[:40]} DONE"


def test_prepare_turn_soft_never_blocks():
    opt = TokenOptimizer(
        system="x",
        max_retries=2,
        max_tokens=50,
        stagnate_after=3,
        err_tail=40,
        ctx_max=80,
    )
    opt.charge_system()
    hist = []
    # Blow retry budget
    for i in range(5):
        turn = opt.prepare_turn(
            "debug this huge " + ("code " * 200),
            history=hist,
            soft=True,
        )
        assert "messages" in turn and turn["user"] is not None
        # Always get something to send (soft)
        assert isinstance(turn["messages"], list)
        opt.finish_turn("Error: still failing", ok=False, history=hist)
    # Loop continued without exception
    assert opt.total_tokens >= 0


def test_diy_example_loop():
    out = example_custom_loop(
        ["fix tax bug", "rename items to lines"],
        fake_model,
        system="fix",
        max_turns=2,
    )
    assert len(out["results"]) == 2
    assert out["summary"]["input_tokens"] > 0


def test_compact_only_level():
    opt = TokenOptimizer(system="s", ctx_max=100, hist_keep=1)
    big = "FULL SOURCE\n" + ("def foo():\n    pass\n" * 100)
    hist = [big, big, big]
    small = opt.compact_context(context="fix now", history=hist, max_chars=100)
    assert len(small) <= 120


def test_observe_api_usage_additive():
    opt = TokenOptimizer(system="s")
    opt.observe_api_usage(
        {"prompt_tokens": 100, "completion_tokens": 20, "reasoning_tokens": 50}
    )
    assert opt.tin == 100
    assert opt.tout == 70


if __name__ == "__main__":
    test_prepare_turn_soft_never_blocks()
    test_diy_example_loop()
    test_compact_only_level()
    test_observe_api_usage_additive()
    print("diy tests OK")
    print()
    print(adoption_levels())

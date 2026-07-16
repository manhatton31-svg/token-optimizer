#!/usr/bin/env python3
"""
A/B live test: naive Grok call vs OptimizedLoop-wrapped call.

Both hit the real xAI API. Prints API usage so you can match console.x.ai.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

# load .env
for candidate in (Path.cwd() / ".env", Path.home() / ".env"):
    if candidate.is_file():
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip("'").strip('"')
            if k and not os.environ.get(k):
                os.environ[k] = v

from openai import OpenAI

from token_optimizer import OptimizedLoop
from token_optimizer.costing import estimate_cost_usd, fetch_xai_model_pricing

MODEL = "grok-4.3"
BASE_URL = "https://api.x.ai/v1"
MAX_OUT = 256

SYSTEM = "You are a careful coding agent. Prefer short, actionable answers."

# Shared task (agent-style)
TASK = """# Task
Find three bugs in this checkout snippet and give the correct total for
LineItem(10,1)+LineItem(20,2) at 10% tax. Under 150 words.

```python
def grand_total(order, tax_rate=0.10):
    sub = 0
    for line in order.items:
        sub += line.price * line.qty
    return int(sub * (1 + tax_rate) * (1 + tax_rate))

def checkout(order):
    return {"total": legacy_cart_total(order)}
```
"""

# Naive path also resends bulky history (what agents do without the wrapper)
NAIVE_HISTORY = """
### Prior conversation (full dump — no compaction)
USER: explore repo
ASSISTANT: """ + ("I listed many files and tools. " * 40) + """
TOOL list_dir result: """ + json.dumps({"entries": [f"src/f{i}.py" for i in range(40)]}) + """
USER: read pricing
ASSISTANT: """ + ("Here is a long explanation of pricing modules. " * 30) + """
TOOL read_file: """ + ("line stub xxxxxxxxxx\n" * 50) + """
"""


def _client() -> OpenAI:
    key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if not key:
        raise SystemExit("Set XAI_API_KEY in .env")
    return OpenAI(api_key=key, base_url=BASE_URL)


def _usage_dict(resp) -> dict:
    u = getattr(resp, "usage", None)
    out = {
        "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(u, "total_tokens", 0) or 0),
    }
    for attr, key in (
        ("prompt_tokens_details", "cached_tokens"),
        ("completion_tokens_details", "reasoning_tokens"),
    ):
        det = getattr(u, attr, None) if u else None
        if det is None:
            continue
        try:
            if key == "cached_tokens":
                out["cached_tokens"] = int(getattr(det, "cached_tokens", 0) or 0)
            else:
                out["reasoning_tokens"] = int(getattr(det, "reasoning_tokens", 0) or 0)
        except Exception:
            m = re.search(rf"{key}=(\d+)", str(det))
            if m:
                out[key] = int(m.group(1))
    return out


def call_grok(system: str, user: str) -> tuple[str, dict, float]:
    client = _client()
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=MAX_OUT,
        temperature=0.2,
    )
    elapsed = time.perf_counter() - t0
    text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
    return text, _usage_dict(resp), elapsed


def summarize(label: str, usage: dict, seconds: float, rates: dict) -> dict:
    est = estimate_cost_usd(
        usage,
        pin=rates["pin"],
        pout=rates["pout"],
        pin_cached=rates["pin_cached"],
    )
    row = {
        "label": label,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "cached_tokens": usage.get("cached_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "reasoning_tokens": usage.get("reasoning_tokens", 0),
        "api_total": usage.get("total_tokens", 0),
        "seconds": round(seconds, 3),
        "est_cost_usd": round(float(est["cost_usd"]), 8),
        "breakdown": est.get("breakdown"),
    }
    return row


def main() -> int:
    rates = fetch_xai_model_pricing(MODEL)
    print("=" * 72)
    print("A/B TEST: naive Grok vs OptimizedLoop (live API)")
    print("=" * 72)
    print(f"model: {MODEL}")
    print(
        f"rates: pin=${rates['pin']}/M cached=${rates['pin_cached']}/M "
        f"out=${rates['pout']}/M ({rates['source']})"
    )
    print()

    # ----- A: NORMAL (no wrapper) -----
    print("-" * 72)
    print("A) NORMAL — full history dump, no TokenOptimizer")
    print("-" * 72)
    naive_user = NAIVE_HISTORY + "\n\n" + TASK
    text_a, usage_a, sec_a = call_grok(SYSTEM, naive_user)
    row_a = summarize("normal", usage_a, sec_a, rates)
    print(f"prompt chars: {len(SYSTEM) + len(naive_user)}")
    print(json.dumps(row_a, indent=2))
    print(f"preview: {text_a[:200]}…")
    print()

    # ----- B: WRAPPER (OptimizedLoop) -----
    print("-" * 72)
    print("B) WRAPPER — OptimizedLoop compact context + metering")
    print("-" * 72)
    loop = OptimizedLoop(
        system=SYSTEM,
        provider="grok",
        pin=rates["pin"],
        pout=rates["pout"],
        max_retries=5,
        max_tokens=100_000,
    )
    # Simulate prior bulky history going through the wrapper (compacted)
    for chunk in (
        "explore repo " + ("files " * 40),
        "tool list_dir " + json.dumps([f"f{i}" for i in range(40)]),
        "read pricing " + ("line stub\n" * 50),
    ):
        with loop.task(chunk, label="hist") as step:
            if step.active:
                # don't call model for history seed — just record compacted obs
                step.record("ok summarized", ok=True, emit="h")

    with loop.task(TASK, label="fix") as step:
        if not step.active:
            print("step inactive:", step.reason)
            return 1
        # What we actually send to the model = compacted context
        compact_user = step.context
        print(f"compact prompt chars: {len(SYSTEM) + len(compact_user)}")
        text_b, usage_b, sec_b = call_grok(SYSTEM, compact_user)
        step.record(text_b, ok=True)

    row_b = summarize("wrapper", usage_b, sec_b, rates)
    # Include optimizer internal meters (local estimate of what we billed)
    row_b["optimizer_stats"] = loop.summary()
    print(json.dumps(row_b, indent=2))
    print(f"preview: {text_b[:200]}…")
    print()

    # ----- Compare -----
    print("=" * 72)
    print("COMPARE (API-reported — match these on console.x.ai)")
    print("=" * 72)
    print(f"{'metric':<22}{'normal':>12}{'wrapper':>12}{'saved':>12}")
    for key in (
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "api_total",
    ):
        a, b = row_a[key], row_b[key]
        saved = a - b
        pct = (100.0 * saved / a) if a else 0.0
        print(f"{key:<22}{a:>12}{b:>12}{saved:>8} ({pct:.0f}%)")
    print(
        f"{'seconds':<22}{row_a['seconds']:>12}{row_b['seconds']:>12}"
        f"{row_a['seconds'] - row_b['seconds']:>12.3f}"
    )
    print(
        f"{'est_cost_usd':<22}{row_a['est_cost_usd']:>12.8f}"
        f"{row_b['est_cost_usd']:>12.8f}"
        f"{row_a['est_cost_usd'] - row_b['est_cost_usd']:>12.8f}"
    )
    combined = row_a["api_total"] + row_b["api_total"]
    print()
    print("DASHBOARD EXPECTATION (if starting total = S):")
    print(f"  after A (normal):  S + {row_a['api_total']}")
    print(f"  after B (wrapper): S + {row_a['api_total']} + {row_b['api_total']} = S + {combined}")
    print(f"  combined api_total both runs: {combined}")
    print()
    print("Tell us your console total before and after A, then after B.")
    print("=" * 72)

    Path("ab_test_last.json").write_text(
        json.dumps({"normal": row_a, "wrapper": row_b, "rates": {
            "pin": rates["pin"], "pout": rates["pout"], "pin_cached": rates["pin_cached"],
        }}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

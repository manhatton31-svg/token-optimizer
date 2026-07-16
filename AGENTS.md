# Agent install card (Cursor / Grok Build / coding agents)

## Goal

Use TokenOptimizer so multi-turn Grok agent loops save **≥90% tokens** vs resending full files/history, and meter **$** to xAI `cost_in_usd_ticks`.

## Install

```bash
pip install "grok-token-optimizer[grok]"
# or from source: pip install -e ".[grok]"
# requires XAI_API_KEY in ~/.env or environment
```

## Minimal wrap

```python
from token_optimizer import GrokSession, tool_compact

session = GrokSession(profile="production")  # or innovate | bulk | frontier
# once per environment:
# session.calibrate()

hist = []
while not done:
    tool_blob = tool_compact(raw_tool_output)  # required for fat tools
    r = session.chat(user_task, history=hist, tool_result=tool_blob)
    # use r.text; meters: r.api_total, r.cost_usd_api
```

## Profile picker

- Hard design / architecture → `frontier` or `chat(..., model="grok-4.5", effort="high")`
- Invent + reason → `innovate`
- Default volume → `production`
- Cheap after idea works → `bulk`

## Do / don’t

- **Do** compact tool results and history via session / `prepare_turn`
- **Do** calibrate once so api $ ≈ est $ (≥99.5%)
- **Don’t** resend entire repos each turn
- **Don’t** claim 90% token savings for Imagine/Voice (different billing)

## Verify

```python
cal = session.calibrate()
assert cal["within_99_5"]
session.export_jsonl("audit.jsonl")
```

See `SAVINGS.md` and `README.md`.

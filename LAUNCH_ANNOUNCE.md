# Launch announce (X / Discord / Slack)

~150 words — paste as-is or trim for character limits.

---

**Naive Grok agents burn tokens.** They resend full history, tool JSON, and whole files every turn. That doesn’t make the model smarter — it just hits your xAI bill.

**TokenOptimizer** is a drop-in path for multi-turn Grok agents that targets **~90%+ total token savings** vs that naive loop (locked live: ~93% bulk, ~91% innovate with full high reasoning). Dollars meter to your real API bill via `cost_in_usd_ticks`. Specs: https://github.com/manhatton31-svg/token-optimizer/blob/main/SAVINGS.md

**Grok is active now.** OpenAI / Anthropic / Gemini are marked coming soon.

```bash
pip install "grok-token-optimizer[grok]"
# or: pip install "git+https://github.com/manhatton31-svg/token-optimizer.git#egg=grok-token-optimizer[grok]"
```

You need an **`XAI_API_KEY`** (console.x.ai → `~/.env`). Then:

```python
from token_optimizer import GrokSession
s = GrokSession(profile="production")
print(s.calibrate()["within_99_5"])
```

Or UI: `python -m ui.server --open` (localhost only).

**Looking for 3 design partners** building real Grok/Build agent loops. Reply with interest — we’ll help you calibrate and capture your before/after numbers.

Repo: https://github.com/manhatton31-svg/token-optimizer

---

## Short X variant (~280 chars)

Naive Grok agents resend full dumps every turn. TokenOptimizer targets 90%+ token cut + meters $ to real xAI ticks. Grok live; others soon.

pip install "grok-token-optimizer[grok]"
(needs XAI_API_KEY)

Seeking 3 design partners → github.com/manhatton31-svg/token-optimizer

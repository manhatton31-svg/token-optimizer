# Target personas — first command each

Use these for outreach, onboarding copy, and “who is this for?” slides.

| # | Persona | Pain | First command | Then |
|---|---------|------|---------------|------|
| 1 | **Agent builder** (custom tool loop) | History + tool JSON explode token count | `pip install "git+https://github.com/manhatton31-svg/token-optimizer.git#egg=token-optimizer[grok]"` | Wrap loop with `GrokSession` + `tool_compact`; `calibrate()` once |
| 2 | **Grok Build / coding-agent user** | Resends whole files every repair turn | `python examples/quickstart_demo.py` | `GrokSession(profile="bulk", model="grok-build-0.1")` or `innovate` for hard bugs |
| 3 | **Indie SaaS on xAI** | Unpredictable Grok bill; no $ truth | `python -c "from token_optimizer import GrokSession; print(GrokSession().calibrate())"` | Ship `production` profile; watch `cost_usd_api` vs est ≥99.5% |
| 4 | **Heavy innovator / researcher** | Wants full reasoning *and* savings | `GrokSession(profile="innovate")` then `calibrate()` | Design on `frontier` (4.5); volume on `production`/`bulk` |
| 5 | **Product / ops lead (try before eng)** | Needs a demo without writing an agent | `python -m ui.server --open` | Click **Calibrate my Grok key** → Chat → Export JSONL |

---

## One-liners by persona

### 1. Agent builder
```bash
pip install "git+https://github.com/manhatton31-svg/token-optimizer.git#egg=token-optimizer[grok]"
```

### 2. Grok Build / coding agent
```bash
python examples/quickstart_demo.py
```
*(clone repo first if not installed editable)*

### 3. Indie SaaS
```bash
python -c "from token_optimizer import GrokSession; print(GrokSession().calibrate())"
```

### 4. Heavy innovator
```bash
python -c "from token_optimizer import GrokSession; s=GrokSession(profile='innovate'); print(s.calibrate()); print(s.chat('Reply: ready') .text)"
```

### 5. Product / ops (UI)
```bash
python -m ui.server --open
```

---

## Shared prerequisites

- Python 3.10+  
- `XAI_API_KEY` in `~/.env` or environment ([console.x.ai](https://console.x.ai/))  
- Never paste keys into Discord/email/GitHub  

## Success signal to collect

From every persona: **`within_99_5`** from calibrate + a short note on their loop type.

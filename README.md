# TokenOptimizer

**Cut multi-turn Grok agent tokens ~90%+ and meter cost to your real xAI bill.**

Naive agents resend full history, tool dumps, and whole files every turn. TokenOptimizer compacts context, applies Grok profiles (frontier → bulk), and locks dollars to provider `cost_in_usd_ticks` (≥ 99.5%).

| Status | Provider |
|--------|----------|
| **Active** | xAI Grok |
| Coming soon | OpenAI · Anthropic · Gemini |

Locked savings (see [SAVINGS.md](SAVINGS.md)): **~93.4%** tokens on bulk · **~91.3%** tokens on innovate with full high reasoning.

---

## 5-minute quickstart

### 1. Key

```bash
# https://console.x.ai/ — copy to ~/.env
cp .env.example ~/.env   # then edit XAI_API_KEY=...
```

### 2. Install

```bash
pip install -e ".[grok]"
# or: pip install -e ".[all]"
```

### 3. Calibrate + chat

```python
from token_optimizer import GrokSession

s = GrokSession(profile="production")  # default volume on grok-4.3
cal = s.calibrate()                    # live probe on YOUR key
print(cal["api_cost_usd"], cal["est_cost_usd"], cal["within_99_5"])

r = s.chat("In one sentence: what is double tax in checkout?")
print(r.text, r.api_total, r.cost_usd_api)
```

### 4. UI (optional)

```bash
token-optimizer-ui --open
# → http://127.0.0.1:8787
# Calibrate my Grok key → Chat → Export JSONL
```

**Security:** UI binds to `127.0.0.1` by default. Do not expose it to the public internet. API keys are never returned by the API.

---

## Profiles

| Profile | Model | Effort | Use |
|---------|--------|--------|-----|
| `frontier` | grok-4.5 | high | design / hardest bugs |
| `innovate` | grok-4.3 | high | invent + full reason, still ≥90% tokens |
| `production` | grok-4.3 | balanced | **default volume** |
| `bulk` | non-reasoning | off | max savings after the idea works |

```python
# Design turn without leaving production session
s.chat("Architect the fix", model="grok-4.5", effort="high")
s.chat("Implement step 1")  # back to production rates
```

**Cost truth:** `usd = cost_in_usd_ticks / 1e10` · target match ≥ 99.5%.

---

## For coding agents (Cursor / Grok Build / DIY)

```python
from token_optimizer import GrokSession, tool_compact

session = GrokSession(profile="innovate")  # or production / bulk
hist = []

# Your loop — we only compact + meter
turn_user = "Fix NameError in tax.py"
tool_out = tool_compact(open("huge_trace.txt").read())  # never ship full dumps
r = session.chat(turn_user, history=hist, tool_result=tool_out)
# r.usage, r.cost_usd_api, r.cost_usd_est
session.export_jsonl("session_audit.jsonl")
```

DIY without `GrokSession`: `TokenOptimizer.prepare_turn` → your `client.chat.completions.create` → `observe_api_usage`.

```bash
python -m token_optimizer   # catalog + version
token-optimizer-onboard status
```

---

## Examples

| Script | Purpose |
|--------|---------|
| `examples/diy_tool_loop.py` | Tool loop + profiles |
| `examples/debug_retry_loop.py` | Debug retries |
| `examples/frontier_then_production.py` | Design → ship |
| `examples/stream_chat.py` | Streaming |
| `ab_full_xai_matrix.py` | Text + Build + Imagine + TTS |

---

## Modalities

| Modality | Optimizer | Notes |
|----------|-----------|--------|
| Text / stream / Grok Build | **Full** | 90%+ token playbook |
| Imagine image | Partial | flat $/image; compact prompts |
| Voice TTS | Partial | compact scripts ($/char) |

---

## Develop / test

```bash
pip install -e ".[dev]"
python test_grok_session.py
python test_ui_api.py
python test_diy_loop.py
```

Live (spends real $): `python ab_full_xai_matrix.py`

---

## Docs

- [SAVINGS.md](SAVINGS.md) — claims, profiles, locked numbers  
- [CHANGELOG.md](CHANGELOG.md) — releases  
- [LAUNCH.md](LAUNCH.md) — remaining launch checklist / prompt stream  

## License

MIT — see [LICENSE](LICENSE).

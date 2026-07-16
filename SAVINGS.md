# Grok savings specifications

Repo: https://github.com/manhatton31-svg/token-optimizer

**Baseline (“naive”):** multi-turn agent that resends full history, full tool JSON, and full source files every turn.

**Efficient:** `GrokSession` / `TokenOptimizer.prepare_turn` compact context + short emits + profile model/effort.

**Product target:** ≥ **90%** total token reduction vs naive on Grok text/agent loops.

**Dollar accuracy:** provider `cost_in_usd_ticks / 1e10` vs our sheet estimate ≥ **99.5%** after `session.calibrate()`.

---

## Locked live measurements (xAI)

| Path | Profile | Model | Prompt save | **Total token save** | Cost save | Source |
|------|---------|--------|------------:|---------------------:|----------:|--------|
| Bulk / volume after idea works | `bulk` | `grok-4.20-0309-non-reasoning` | ~93.6% | **~93.4%** | **~96.1%** | `ab_calibrate_90_last.json` |
| Full reasoning invent/debug | `innovate` | `grok-4.3` + high effort | ~97.3% | **~91.3%** | ~87.6%* | `ab_innovate_last.json` |
| Default production | `production` | `grok-4.3` balanced | target ≥90% | target ≥90% | tracks ticks | same compact stack |
| Frontier design | `frontier` | `grok-4.5` high | compact lever same | use sparingly | higher $/tok | design then ship on 4.3/bulk |
| Grok Build agents | bulk or `model=grok-build-0.1` | Build SKU | same as text agents | target ≥90% vs fat dumps | ticks | full optimizer |

\*Cost save can trail token save when reasoning tokens bill as output — **token** target still clears 90%+.

### Method notes

- **Innovate 91.3%:** compact context + **one deep consult per case** (not unbounded multi-round high-reason loops).
- **Bulk 93.4%:** non-reasoning model after the idea works — removes reasoning tax.
- **Prompt-side** is usually the largest win (93–97%): history and tools, not “shorter answers only.”

---

## Profiles (what to use when)

| Profile | Role | Savings posture |
|---------|------|-----------------|
| `frontier` | Design / hardest bugs on 4.5 | Compact hard; spend on quality |
| `innovate` | Invent + high reason on 4.3 | ≥90% tokens with full reasoning |
| `production` | Day-to-day volume (default) | ≥90% via compact; balanced effort |
| `bulk` | After the idea works | Max tokens + $ |

Per-call overrides: `session.chat(..., model="grok-4.5", effort="high")` then back to production rates.

---

## Modalities (not the same 90% claim)

| Modality | Optimizer | Savings lever | Cost truth |
|----------|-----------|---------------|------------|
| Text / stream / Grok Build | **Full** | compact + profiles | `cost_in_usd_ticks / 1e10` |
| Imagine image | **Partial** | short briefs; cheaper image SKU | flat `image_price` / image |
| Voice TTS | **Partial** | compact spoken script (chars) | ~$/1M characters |
| Voice realtime / video | **Partial** | short instructions / duration | per-minute / per-second |

Do **not** market “90% tokens” for Imagine or Voice.

---

## Customer onboarding

```python
from token_optimizer import GrokSession

s = GrokSession(profile="production")
cal = s.calibrate()          # live probe on THEIR key
assert cal["within_99_5"]
```

`calibration_state.json` is **local / per environment** — not a shared multi-tenant DB.

---

## Re-verify

```bash
python ab_calibrate_90_grok.py   # bulk path
python ab_innovate_90_grok.py    # innovate path
python ab_full_xai_matrix.py     # modality matrix
```

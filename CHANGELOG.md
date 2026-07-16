# Changelog

## 0.2.0 — 2026-07-16 (beta, Grok launch)

### Ready for tryouts
- **GrokSession** profiles: `frontier` | `innovate` | `production` | `bulk`
- Per-call `model=` / `effort=` on `chat` / `chat_stream`
- Provider `$` via `cost_in_usd_ticks / 1e10`; dual meters api vs est
- `session.calibrate()` — customer live lock ≥ 99.5%
- `export_jsonl()` session audit
- `product_catalog()` — Grok **active**, OpenAI/Anthropic/Gemini **coming soon**
- Local customer UI: `python -m ui.server` or `token-optimizer-ui`
- Full xAI matrix harness: text, Build, Imagine, TTS
- Locked savings: bulk ~93.4% tokens, innovate ~91.3% tokens (see `SAVINGS.md`)

### Package
- MIT LICENSE, `.gitignore`, `.env.example`
- Optional dependency: `openai`
- Console scripts: `token-optimizer-ui`, `token-optimizer-onboard`

## 0.1.0 — earlier

- TokenOptimizer / OptimizedLoop core
- DIY prepare_turn path
- Multi-provider onboarding stubs

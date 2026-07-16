# Beta invite email

**Subject:** Try TokenOptimizer on your Grok agents (~90%+ token target)

---

Hi {{name}},

I’m inviting a small set of people to try **TokenOptimizer** — a Python library that compacts multi-turn Grok agent context (history / tools / errors) and meters cost to your **real** xAI usage (`cost_in_usd_ticks`), not a guess.

**Why it exists:** naive agents resend full files and transcripts every turn. We target **≥90% total token savings** vs that baseline (details: [SAVINGS.md](https://github.com/manhatton31-svg/token-optimizer/blob/main/SAVINGS.md)). **Grok is active**; OpenAI/Anthropic/Gemini are coming soon.

### 10-minute try

1. Get an API key: https://console.x.ai/ → put `XAI_API_KEY=...` in `~/.env`

2. Install:

```bash
pip install "git+https://github.com/manhatton31-svg/token-optimizer.git#egg=token-optimizer[grok]"
```

3. Calibrate + one chat:

```bash
python -c "from token_optimizer import GrokSession; s=GrokSession(profile='production'); print(s.calibrate())"
```

Or:

```bash
git clone https://github.com/manhatton31-svg/token-optimizer.git
cd token-optimizer
pip install -e ".[grok]"
python examples/quickstart_demo.py
```

Optional UI (local only): `python -m ui.server --open`

### What I’d love back (reply to this email)

Please send:

1. **`within_99_5`** from `calibrate()` (true/false) and `api_cost_usd` / `est_cost_usd` if handy  
2. **One agent loop result** — e.g. profile used (`production` / `innovate` / `bulk` / `frontier`), rough naive vs efficient token totals, or a short note on what you wrapped  

**Do not send API keys.**

Repo / issues: https://github.com/manhatton31-svg/token-optimizer  

Thanks — excited to see your numbers.

{{your_name}}

---

## Follow-up (if no reply in 5 days)

**Subject:** Re: TokenOptimizer beta — 2-min calibrate?

Quick nudge: if you have an `XAI_API_KEY`, this is the only command that matters:

```bash
python -c "from token_optimizer import GrokSession; print(GrokSession().calibrate()['within_99_5'])"
```

Reply with `True`/`False` and whether you’re on a debug loop, Grok Build, or something else. Happy to help if install fails.

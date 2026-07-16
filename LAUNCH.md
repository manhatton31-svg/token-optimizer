# Launch checklist & prompt stream

Use these **in order**. Each block is a copy-paste prompt for the next coding session (or do them yourself). Stop when the exit criteria for that step pass.

---

## Already done

- [x] `.gitignore`, `LICENSE` (MIT), `.env.example`
- [x] `SAVINGS.md`, `CHANGELOG.md`, launch `README.md`, `AGENTS.md`, `SECURITY.md`
- [x] Issue templates (bug + savings)
- [x] `examples/quickstart_demo.py` (+ `--mock`)
- [x] `pyproject.toml` v0.2.0 — `[grok]` extra, console scripts, package-data for UI
- [x] `python -m token_optimizer`, `python -m ui`
- [x] Grok product core + UI + matrix (prior work)

---

## Prompt 1 — Git foundation

```
Initialize git for TokenOptimizer launch packaging.

Requirements:
1. git init if needed; ensure .gitignore is respected.
2. Stage only public-safe files (no .env, no calibration_ledger, no matrix_artifacts audio).
3. Create initial commit with message: "chore: v0.2.0 beta packaging for Grok launch".
4. Do NOT push or create a remote unless I provide the GitHub URL.
5. Print git status -sb and the commit hash.
```

**Exit:** clean `git status`, one root commit, no secrets staged.

---

## Prompt 2 — Fix package URLs + install smoke

```
Prepare the package for a real public repo.

Requirements:
1. Replace YOUR_ORG in pyproject.toml / README / SAVINGS links with: manhatton31-svg/token-optimizer.
2. pip install -e ".[all]" and verify: python -m token_optimizer ; python -c "import ui.server, onboard".
3. Run: python test_grok_session.py ; python test_ui_api.py ; python test_diy_loop.py ; python examples/quickstart_demo.py --mock
4. Confirm no API keys in any test output.
```

**Exit:** editable install works; unit tests green.

---

## Prompt 3 — GitHub remote + first public push

```
Connect the local repo to GitHub and publish the beta.

Requirements:
1. Use remote: git@github.com:manhatton31-svg/token-optimizer.git (or HTTPS).
2. Create repo if needed via gh (public), default branch main.
3. Push main. Do not force-push.
4. Add topics: grok, xai, agents, tokens, llm.
5. Paste the public clone URL and: pip install "git+https://github.com/manhatton31-svg/token-optimizer.git#egg=token-optimizer[grok]"
```

**Exit:** stranger can clone + install from GitHub.

---

## Prompt 4 — CI (unit only)

```
Add GitHub Actions CI for TokenOptimizer.

Requirements:
1. .github/workflows/ci.yml — on push/PR: Python 3.11, pip install -e ".[dev]", run test_grok_session.py test_ui_api.py test_diy_loop.py examples/quickstart_demo.py --mock.
2. No live API calls in CI (no XAI_API_KEY required).
3. Badge snippet for README.
```

**Exit:** green CI on the default branch.

---

## Prompt 5 — Soft launch copy + invite list ✅

- [x] `LAUNCH_ANNOUNCE.md` — X/Discord post + short variant  
- [x] `EMAIL_BETA.md` — invite + follow-up  
- [x] `PERSONAS.md` — 5 personas with first commands  

**Exit:** you can paste announce + email today.

---

## Prompt 6 — PyPI

- [x] Name `token-optimizer` taken → use **`grok-token-optimizer`**
- [x] Build sdist+wheel; twine check PASSED
- [x] README / AGENTS / announce prefer `pip install "grok-token-optimizer[grok]"`
- [ ] Upload (needs PyPI API token in env — not present yet)
- [ ] Verify: `pip install grok-token-optimizer==0.2.0 && python -m token_optimizer`

**Upload when ready** (never paste token into chat):

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-..."   # create at https://pypi.org/manage/account/token/
# then say: upload to pypi now
```

---

## Launch gate (you’re ready when)

| Gate | Check |
|------|--------|
| Install | `pip install -e ".[grok]"` or git+URL works on a second machine |
| Calibrate | New key → `calibrate()` → `within_99_5` |
| Agent | AGENTS.md loop runs one real task |
| UI | `token-optimizer-ui` calibrate + chat |
| Trust | LICENSE, no secrets in git, localhost warning |
| Claims | SAVINGS.md linked from announce |
| Feedback | Issue templates live |

**Not required for beta:** multi-provider, hosted SaaS, PyPI, video matrix.

---

## Current next prompt

→ **Prompt 6 — Optional PyPI** (after 3 external installs), **or** send announce + 3 beta emails now.

**Soft-launch gate (no PyPI required):**
- [x] Public GitHub repo + CI green  
- [x] Announce + email + personas written  
- [ ] You post `LAUNCH_ANNOUNCE.md` and send `EMAIL_BETA.md` to 3 people  
- [ ] Collect 3× `within_99_5` replies

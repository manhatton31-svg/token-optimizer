# Security

## Reporting

If you find a vulnerability (key leakage, path traversal in exports, etc.), open a **private** security advisory on GitHub or email the maintainers. Do not file a public issue with secrets.

## API keys

- Store keys in `~/.env` or the process environment (`XAI_API_KEY`).
- Never commit `.env`.
- The local UI **rejects** client-supplied `api_key` fields and never returns keys in JSON.
- Do not paste keys into GitHub issues.

## Local UI

- Default bind: `127.0.0.1` only.
- **Do not** expose `token-optimizer-ui` / port 8787 to the public internet without auth and TLS.
- Session export writes local JSONL under `ui/_exports/` (gitignored).

## Supply chain

- Prefer pinned installs for production.
- Optional dependency: `openai` (official SDK) for xAI-compatible API access.

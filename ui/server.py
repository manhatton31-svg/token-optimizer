#!/usr/bin/env python3
"""
Customer UI server (stdlib only).

  python ui/server.py
  python ui/server.py --port 8787

Open http://127.0.0.1:8787

API under /api/* — see ui/api.py. Never logs or returns API keys.
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.api import dispatch  # noqa: E402


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        # Avoid logging bodies / query strings that might contain secrets
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self._cors()
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/"):
            status, body = dispatch("GET", path)
            self._json(status, body)
            return
        if path in ("/", ""):
            self.path = "/index.html"
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if not path.startswith("/api/"):
            self._json(404, {"ok": False, "error": "Not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json(400, {"ok": False, "error": "Invalid JSON body"})
            return
        if not isinstance(body, dict):
            self._json(400, {"ok": False, "error": "JSON body must be an object"})
            return
        # Never accept client-supplied API keys into process env via this UI
        body.pop("api_key", None)
        body.pop("xai_api_key", None)
        status, out = dispatch("POST", path, body)
        self._json(status, out)


def main() -> int:
    ap = argparse.ArgumentParser(description="TokenOptimizer customer UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--open", action="store_true", help="Open browser")
    args = ap.parse_args()

    if not (STATIC / "index.html").is_file():
        print(f"Missing UI static file: {STATIC / 'index.html'}", file=sys.stderr)
        return 1

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"TokenOptimizer UI → {url}")
    print("  Grok: active | OpenAI / Anthropic / Gemini: coming soon")
    print("  Ctrl+C to stop")
    if args.open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

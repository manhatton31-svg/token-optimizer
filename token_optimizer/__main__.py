"""python -m token_optimizer → print product catalog + version."""
from __future__ import annotations

import json

from token_optimizer import __version__, product_catalog


def main() -> int:
    cat = product_catalog()
    print(f"token-optimizer {__version__}")
    print(json.dumps(
        {
            "ready": cat.get("ready"),
            "default_provider": cat.get("default_provider"),
            "default_profile": cat.get("default_profile"),
            "providers": [
                {"id": p["id"], "status": p["status"]} for p in cat.get("providers", [])
            ],
            "profiles": list((cat.get("profiles") or {}).keys()),
        },
        indent=2,
    ))
    print("UI:  token-optimizer-ui   or   python -m ui.server --open")
    print("Docs: SAVINGS.md  |  README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

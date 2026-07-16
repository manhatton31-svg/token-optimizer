#!/usr/bin/env python3
"""
Robust TokenOptimizer / debug_agent test suite.
Each entry: (category, name, broken_source, expect_substring)
"""


def _heavy_checkout_refactor() -> str:
    """
    Multi-file checkout refactor (simulated monorepo) that heavy users hit:

    - Cross-module API rename (legacy_total → grand_total)
    - Attribute rename after model refactor (items → lines)
    - Subtle double-tax logic bug (plausible "tax inclusive" mistake)
    - Misleading comments that push an agent toward the *wrong* helper
      (subtotal_items) and a long retry loop of "almost right" fixes

    Expected healthy total: (10*1 + 20*2) * 1.10 = 55 → int 55
    """
    # pad with realistic decoy modules so context is large (naive agents re-send all)
    decoy = []
    for mod in ("auth_service", "inventory_sync", "promo_engine", "ledger_export"):
        for i in range(1, 6):
            decoy.append(
                f"# === module: services/{mod}.py ===\n"
                f"def {mod}_step_{i}(payload, cfg=None):\n"
                f"    cfg = cfg or {{'retries': 3, 'mod': '{mod}'}}\n"
                f"    acc = 0\n"
                f"    for n in range(1, 8):\n"
                f"        acc += (payload.get('n', n) if isinstance(payload, dict) else n) * i\n"
                f"    if acc < 0:\n"
                f"        raise ValueError('negative accumulator')\n"
                f"    return {{'mod': '{mod}', 'step': {i}, 'acc': acc, 'cfg': cfg}}\n"
            )
    core = r'''
# === module: models/order.py ===
class LineItem:
    """One SKU line after catalog v3 migration."""
    def __init__(self, sku, price, qty):
        self.sku = sku
        self.price = price
        self.qty = qty

    def ext_price(self):
        return self.price * self.qty


class Order:
    """
    Cart/order aggregate.
    REFACTOR NOTE (PR #4418): field `items` renamed to `lines`.
    Several call sites still use the old name — do not "fix" by re-adding items.
    """
    def __init__(self, lines):
        self.lines = list(lines)
        self.currency = "USD"
        self.meta = {"channel": "web", "version": 3}


# === module: services/pricing.py ===
def subtotal_items(order):
    """
    TRAP / plausible wrong target after rename:
    PR description incorrectly said legacy_total -> subtotal_items.
    This helper IGNORES qty (unit price only) — looks fine on qty=1 tests.
    """
    return sum(line.price for line in order.lines)


def total_with_tax_broken_draft(order, tax_rate=0.10):
    """Abandoned draft — do not wire checkout here."""
    sub = subtotal_items(order)
    return int(sub * (1 + tax_rate))


def grand_total(order, tax_rate=0.10):
    """
    Canonical post-refactor total: sum(price*qty) then apply tax once.
    BUG: tax factor applied twice (common "tax on tax" slip in reviews).
    """
    sub = 0
    # BUG: still uses pre-refactor attribute name `items`
    for line in order.items:
        sub += line.price * line.qty
    # BUG: double application of (1 + tax_rate)
    return int(sub * (1 + tax_rate) * (1 + tax_rate))


# === module: services/checkout.py ===
def apply_promos(order, codes=None):
    # no-op stub kept for API compatibility
    return order


def legacy_total(order):
    """Removed in PR #4418 — body deleted; name retained in call sites."""
    raise RuntimeError("legacy_total removed; use grand_total")


def checkout(order, promo_codes=None):
    """
    Checkout entrypoint.
    Incorrect review comment (still in tree):
      "Wire checkout to subtotal_items — pricing owns tax elsewhere."
    Correct wiring is grand_total (tax-inclusive single shot).
    """
    order = apply_promos(order, promo_codes)
    # BUG: call site never updated after rename (NameError if legacy deleted;
    # here we call a missing free function name to force agent resolution)
    amount = legacy_cart_total(order)
    return {"status": "ok", "total": amount, "currency": order.currency}


# === module: api/checkout_handler.py ===
def handle_checkout():
    order = Order(
        [
            LineItem("sku-a", 10, 1),
            LineItem("sku-b", 20, 2),
        ]
    )
    result = checkout(order)
    # Operators grep this exact prefix in logs
    print("ORDER total=%s" % result["total"])
    return result


if __name__ == "__main__" or True:
    handle_checkout()
'''
    return "\n".join(decoy) + "\n" + core


def _pad_module(core: str, seed: str = "svc") -> str:
    """Wrap a buggy core in a 50+ line multi-function module shell."""
    helpers = []
    for i in range(1, 9):
        helpers.append(
            f"def {seed}_helper_{i}(x, y=0):\n"
            f"    \"\"\"Utility {i} for {seed} pipeline.\"\"\"\n"
            f"    if x is None:\n"
            f"        return y\n"
            f"    acc = 0\n"
            f"    for n in range(max(1, int(y) + 1)):\n"
            f"        acc += (x + n) % (n + 3)\n"
            f"    return acc\n"
        )
    helpers.append(
        f"def {seed}_validate(rows):\n"
        f"    if not rows:\n"
        f"        return False\n"
        f"    return all(isinstance(r, (dict, list, tuple, int, str)) for r in rows)\n"
    )
    helpers.append(
        f"def {seed}_config():\n"
        f"    return {{'retries': 3, 'timeout': 30, 'seed': '{seed}'}}\n"
    )
    return "\n".join(helpers) + "\n\n" + core


# (category, name, source, expect)
SUITE = [
    # --- large multi-function (50+ lines) ---
    (
        "large_module",
        "typo_deep_in_module",
        _pad_module(
            "def compute_total(a, b):\n"
            "    result = a + b\n"
            "    return resutl\n"
            "\n"
            "def run_pipeline():\n"
            "    cfg = svc_config()\n"
            "    xs = [svc_helper_1(i, 1) for i in range(3)]\n"
            "    if not svc_validate(xs):\n"
            "        return -1\n"
            "    return compute_total(2, 3)\n"
            "\n"
            "print(run_pipeline())\n",
            "svc",
        ),
        "5",
    ),
    (
        "large_module",
        "missing_math_in_big_file",
        _pad_module(
            "def geometry_area(r):\n"
            "    return math.pi * r * r\n"
            "\n"
            "def report():\n"
            "    a = geometry_area(1)\n"
            "    return int(a)  # pi~3\n"
            "\n"
            "print(report())\n",
            "geo",
        ),
        "3",
    ),
    # --- multiple bugs in one file (needs multi-round) ---
    (
        "multi_bug",
        "typo_then_str_concat",
        "def format_score(name, score):\n"
        "    label = 'user='\n"
        "    # bug1: undefined resutl used below via wrong return path\n"
        "    resutl = label + name\n"
        "    return resutl + ':' + score\n"
        "print(format_score('ada', 10))\n",
        "user=ada:10",
    ),
    (
        "multi_bug",
        "indent_and_arity",
        "def greet(name):\n"
        "print('hi ' + name)\n"
        "greet('a', 'extra')\n",
        "hi a",
    ),
    # --- web development ---
    (
        "web",
        "json_response_missing_import",
        "def api_handler(payload):\n"
        "    data = json.loads(payload)\n"
        "    return data.get('id', 0)\n"
        "print(api_handler('{\"id\": 42}'))\n",
        "42",
    ),
    (
        "web",
        "query_param_type_error",
        "def page(limit):\n"
        "    # web handlers often get strings from query params\n"
        "    return 'showing=' + limit\n"
        "print(page(25))\n",
        "showing=25",
    ),
    (
        "web",
        "route_dict_keyerror",
        "def get_user(req):\n"
        "    # typo key: 'user_id' vs 'userid'\n"
        "    return req['userid']\n"
        "print(get_user({'user_id': 7}))\n",
        "7",
    ),
    # --- data processing / pandas ---
    (
        "data_pandas",
        "pandas_missing_import",
        "df = pd.DataFrame({'n': [1, 2, 3]})\n"
        "print(int(df['n'].sum()))\n",
        "6",
    ),
    (
        "data_pandas",
        "pandas_wrong_column",
        "import pandas as pd\n"
        "df = pd.DataFrame({'name': ['a', 'b'], 'score': [1, 2]})\n"
        "print(int(df['scores'].sum()))\n",
        "3",
    ),
    (
        "data_process",
        "batch_off_by_one",
        "def batch_sum(nums):\n"
        "    total = 0\n"
        "    for i in range(1, len(nums)):  # skips first\n"
        "        total += nums[i]\n"
        "    return total\n"
        "print(batch_sum([10, 20, 30]))\n",
        "60",
    ),
    (
        "data_process",
        "csv_row_div_zero",
        "def safe_rate(hits, total):\n"
        "    return hits / total\n"
        "print(int(safe_rate(10, 0)))\n",
        "0",
    ),
    # --- CLI tools ---
    (
        "cli",
        "argv_index_error",
        "import sys\n"
        "def main(argv):\n"
        "    # forgot default when flag missing\n"
        "    return argv[1]\n"
        "print(main(['tool']))\n",
        "default",
    ),
    (
        "cli",
        "flag_parse_typo",
        "def parse_flags(args):\n"
        "    out = {'verbose': False}\n"
        "    for a in args:\n"
        "        if a == '--verbose':\n"
        "            out['verbos'] = True  # wrong key written\n"
        "    return out['verbose']\n"
        "print(parse_flags(['--verbose']))\n",
        "True",
    ),
    # --- state / scope ---
    (
        "scope",
        "local_leak",
        "def calc():\n"
        "    total = 10 + 5\n"
        "print(total)\n",
        "15",
    ),
    (
        "scope",
        "closure_wrong_default",
        "def make_adders():\n"
        "    fns = []\n"
        "    for i in range(3):\n"
        "        fns.append(lambda x: x + i)  # classic late bind -> always +2\n"
        "    return fns\n"
        "print(make_adders()[0](1))\n",
        "1",  # want i=0 capture -> 1+0=1; fix with default arg
    ),
    # --- imports / dependencies ---
    (
        "imports",
        "from_import_typo",
        "from math import squareroot\n"
        "print(int(squareroot(16)))\n",
        "4",
    ),
    (
        "imports",
        "aliased_module_missing",
        "print(int(json.dumps({'a': 1}).count('a')))\n",
        "1",
    ),
    # --- mixed realistic ---
    (
        "hallucinated",
        "str_method_chain",
        "msg = 'hello'\n"
        "print(msg.uppper())\n",
        "HELLO",
    ),
    (
        "logic",
        "is_even_inverted",
        "def is_even(n):\n"
        "    return n % 2 == 1\n"
        "print(is_even(4))\n",
        "True",
    ),
    (
        "web",
        "status_concat_and_code",
        "def http_msg(code):\n"
        "    return 'status=' + code\n"
        "print(http_msg(200))\n",
        "status=200",
    ),
    # --- heavy real-world refactor (multi-module, traps, multi-bug) ---
    (
        "heavy_refactor",
        "checkout_multi_file_rename_and_tax",
        _heavy_checkout_refactor(),
        # (10*1 + 20*2) * 1.10 = 55
        "ORDER total=55",
    ),
]

# Compatible with debug_agent / benchmark: (source, expect)
CASES = [(src, exp) for _, _, src, exp in SUITE]


def coverage_summary() -> str:
    from collections import Counter

    c = Counter(cat for cat, _, _, _ in SUITE)
    lines = [f"cases={len(SUITE)}"]
    for cat, n in sorted(c.items()):
        lines.append(f"  {cat}: {n}")
    # line counts
    big = sum(1 for _, _, s, _ in SUITE if s.count("\n") >= 50)
    lines.append(f"  large_50plus_lines: {big}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(coverage_summary())
    for i, (cat, name, src, exp) in enumerate(SUITE, 1):
        print(f"{i:02d} [{cat}] {name} lines={src.count(chr(10))+1} expect={exp!r}")

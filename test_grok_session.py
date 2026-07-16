#!/usr/bin/env python3
"""Unit tests for GrokSession (no network except optional live)."""
from __future__ import annotations

import warnings

from token_optimizer import product_catalog, tool_compact
from token_optimizer.grok import (
    PROFILES,
    GrokSession,
    list_profiles,
    warn_fat_prompt,
)
from token_optimizer.core import TokenOptimizer


def test_profiles_exist():
    assert set(list_profiles()) == {"frontier", "innovate", "production", "bulk"}
    assert PROFILES["frontier"]["model"] == "grok-4.5"
    assert PROFILES["frontier"]["reasoning_mode"] in ("innovate", "full", "high")
    assert PROFILES["innovate"]["reasoning_mode"] in ("innovate", "full", "high")
    assert "non-reasoning" in PROFILES["bulk"]["model"]
    assert list_profiles()["frontier"]["role"] == "design"
    assert list_profiles()["production"]["role"] == "volume"


def test_product_catalog_ui():
    cat = product_catalog()
    assert cat["ready"] is True
    assert cat["default_provider"] == "grok"
    by_id = {p["id"]: p for p in cat["providers"]}
    assert by_id["grok"]["status"] == "active"
    assert by_id["openai"]["status"] == "coming_soon"
    assert by_id["anthropic"]["status"] == "coming_soon"
    assert by_id["gemini"]["status"] == "coming_soon"
    assert "frontier" in cat["profiles"]
    assert cat["cost_truth"]["ticks_per_usd"] == 10_000_000_000


def test_warn_fat_prompt():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        msgs = warn_fat_prompt("def f():\n" + ("    x=1\n" * 100), warn_chars=100)
        assert msgs
        assert w


def test_session_init_no_network_prices():
    # live_prices=False avoids models.list
    s = GrokSession(profile="production", live_prices=False, api_key="xai-test-key")
    assert s.profile == "production"
    assert s.opt.model
    assert s.session_api_total == 0
    s.use_profile("bulk")
    assert s.profile == "bulk"
    s.use_profile("innovate")
    assert s.opt.reasoning_mode in ("innovate", "full")
    s.use_profile("frontier")
    assert s.profile == "frontier"
    assert "4.5" in str(s.opt.model)


def test_verify_dashboard_math():
    s = GrokSession(profile="production", live_prices=False, api_key="xai-test")
    s.session_api_total = 1007
    check = s.verify_dashboard(5780, 6787)
    assert check["exact"] is True
    assert check["ok"] is True
    assert s.expected_after(5780) == 6787


def test_ticks_to_usd_scale():
    from token_optimizer.costing import (
        XAI_USD_TICKS_PER_DOLLAR,
        compare_cost_accuracy,
        ticks_to_usd,
    )

    # Empirical: 6546500 ticks ≈ $0.00065465
    usd = ticks_to_usd(6_546_500)
    assert usd is not None
    assert abs(usd - 0.00065465) < 1e-12
    assert XAI_USD_TICKS_PER_DOLLAR == 10_000_000_000
    cmp = compare_cost_accuracy(0.00065465, 0.00065465)
    assert cmp["within_99_5"] is True
    assert cmp["accuracy"] == 1.0


def test_compare_cost_accuracy_on_session():
    s = GrokSession(profile="production", live_prices=False, api_key="xai-test")
    s.session_cost_ticks = 6_546_500
    s.session_cost_usd_api = 0.00065465
    s.session_cost_usd_est = 0.00065465
    c = s.compare_cost_accuracy()
    assert c["within_99_5"] is True
    assert c["api_cost"] == 0.00065465


def test_per_call_overrides_restore():
    s = GrokSession(profile="production", live_prices=False, api_key="xai-test")
    base_model = s.opt.model
    base_mode = s.opt.reasoning_mode
    snap = s._push_call_overrides(model="grok-4.5", effort="high")
    assert "4.5" in str(s.opt.model)
    assert s.opt.reasoning_mode == "high"
    s._pop_call_overrides(snap)
    assert s.opt.model == base_model
    assert s.opt.reasoning_mode == base_mode
    # invalid effort
    try:
        s._push_call_overrides(effort="turbo")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_export_jsonl(tmp_path=None):
    import tempfile
    from pathlib import Path

    s = GrokSession(profile="production", live_prices=False, api_key="xai-test")
    s._events.append(
        {
            "ts": "2026-07-15T00:00:00Z",
            "profile": "production",
            "model": "grok-4.3",
            "seconds": 0.1,
            "usage": {"total_tokens": 10},
            "cost_in_usd_ticks": 1000,
            "cost_usd_api": 0.0000001,
            "cost_usd_est": 0.0000001,
            "rates": {"pin": 1.25, "pout": 2.5, "pin_cached": 0.2},
            "stopped": False,
            "reason": "",
        }
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "session.jsonl"
        n = s.export_jsonl(path)
        assert n == 1
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2  # summary + call
        assert "session_summary" in lines[0]
        assert "call" in lines[1]
        assert "xai-" not in path.read_text()  # no keys


class _FakeDelta:
    def __init__(self, content=None):
        self.content = content


class _FakeChoice:
    def __init__(self, content=None):
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content=None, usage=None):
        self.choices = [_FakeChoice(content)] if content is not None else []
        self.usage = usage


class _FakeUsage:
    def __init__(self, prompt=10, completion=3, total=13, reasoning=0, cached=0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total

        class _P:
            cached_tokens = cached

        class _C:
            reasoning_tokens = reasoning

        self.prompt_tokens_details = _P()
        self.completion_tokens_details = _C()


class _FakeStreamClient:
    """Mimics OpenAI client.chat.completions.create(stream=True)."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.last_kwargs = None

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        assert kwargs.get("stream") is True

        def gen():
            for c in self._chunks:
                yield c

        return gen()


def test_chat_stream_mocked():
    s = GrokSession(profile="production", live_prices=False, api_key="xai-test")
    usage = _FakeUsage(prompt=12, completion=4, total=16, reasoning=0, cached=2)
    chunks = [
        _FakeChunk("Hel"),
        _FakeChunk("lo"),
        _FakeChunk(None, usage=usage),  # final usage-only chunk
    ]
    fake = _FakeStreamClient(chunks)
    s._client = fake  # inject mock client

    parts = list(s.chat_stream("Say hello briefly"))
    assert parts == ["Hel", "lo"]
    assert s.last_result is not None
    assert s.last_result.text == "Hello"
    assert s.last_result.usage.get("prompt_tokens") == 12
    assert s.last_result.usage.get("completion_tokens") == 4
    assert s.session_api_total == 16
    assert s.session_calls == 1
    # stream_options preferred when SDK accepts kwargs
    assert fake.last_kwargs.get("stream") is True


def test_chat_stream_missing_usage_fallback():
    s = GrokSession(profile="bulk", live_prices=False, api_key="xai-test")
    chunks = [_FakeChunk("ok"), _FakeChunk("!")]
    s._client = _FakeStreamClient(chunks)

    parts = list(s.chat_stream("ping"))
    assert "".join(parts) == "ok!"
    assert s.last_result is not None
    assert s.last_result.text == "ok!"
    # Fallback estimate still records something
    assert s.last_result.usage.get("total_tokens", 0) > 0
    assert any("usage missing" in w for w in s.last_result.warnings)


def _fat_tool_dump(n_lines: int = 60) -> str:
    lines = [
        "FILE services/pricing.py",
        "def grand_total(order, tax_rate=0.10):",
        "    for line in order.items:",
        "        sub += line.price",
        "    return int(sub * (1 + tax_rate) * (1 + tax_rate))",
        "ERROR: NameError: legacy_cart_total is not defined",
        "path: C:\\repo\\services\\pricing.py",
        '{"status": "fail", "total": null}',
    ]
    lines += [f"line {i}: " + ("x" * 40) for i in range(1, n_lines)]
    return "\n".join(lines)


def test_tool_compact_fat_file_dump():
    raw = _fat_tool_dump(60)
    assert raw.count("\n") >= 50
    out = tool_compact(raw, max_chars=600)
    assert len(out) < len(raw)
    assert len(out) <= 650  # small slack for prefix
    # Keeps high-signal bits
    assert "ERROR" in out or "error" in out.lower() or "legacy" in out.lower()
    # Never raises on junk
    assert tool_compact(None) == ""
    assert tool_compact(123) != ""  # type: ignore[arg-type]
    assert tool_compact("\x00\x00") == "" or isinstance(tool_compact("\x00"), str)


def test_prepare_turn_compacts_tool_result():
    opt = TokenOptimizer(system="s", ctx_max=100)
    raw = _fat_tool_dump(55)
    turn = opt.prepare_turn("fix checkout", tool_result=raw, bill=False, bump=False)
    assert turn.get("tool_result_compact") is not None
    assert len(turn["tool_result_compact"]) < len(raw)
    assert len(turn["user"]) < len(raw)


def test_grok_session_warns_on_original_tool_then_compacts():
    s = GrokSession(profile="production", live_prices=False, api_key="xai-test")
    raw = _fat_tool_dump(55)
    hist, turn, early = s._prepare_chat_turn(
        "name three fixes",
        tool_result=raw,
        bill_prepare=False,
    )
    assert early is None
    # Original fat warn + compact notice
    joined = " ".join(s.last_warnings).lower()
    assert "tool" in joined
    # Model-facing tool content is compact (embedded in turn blob / shorter than raw)
    assert len(turn["user"]) < len(raw)
    assert turn.get("tool_result_compact") is None or len(
        turn.get("tool_result_compact") or ""
    ) < len(raw)


if __name__ == "__main__":
    test_profiles_exist()
    test_product_catalog_ui()
    test_warn_fat_prompt()
    test_session_init_no_network_prices()
    test_verify_dashboard_math()
    test_ticks_to_usd_scale()
    test_compare_cost_accuracy_on_session()
    test_per_call_overrides_restore()
    test_export_jsonl()
    test_chat_stream_mocked()
    test_chat_stream_missing_usage_fallback()
    test_tool_compact_fat_file_dump()
    test_prepare_turn_compacts_tool_result()
    test_grok_session_warns_on_original_tool_then_compacts()
    print("test_grok_session OK")

#!/usr/bin/env python3
"""
Full xAI API matrix test for TokenOptimizer / GrokSession.

Covers:
  A) Text chat  — production, innovate, frontier, bulk (+ stream)
  B) Grok Build — grok-build-0.1 coding path via GrokSession
  C) Imagine    — images/generations (flat per-image $)
  D) Voice TTS  — /v1/tts (per-character $)
  E) Benefit map — which modalities get compact/cost metering

Never prints API keys. Writes ab_full_xai_matrix_last.json

  python ab_full_xai_matrix.py
  python ab_full_xai_matrix.py --skip-imagine --skip-tts
  python ab_full_xai_matrix.py --skip-video   # video skipped by default (cost/latency)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from token_optimizer import GrokSession, product_catalog  # noqa: E402
from token_optimizer.costing import (  # noqa: E402
    XAI_USD_TICKS_PER_DOLLAR,
    compare_cost_accuracy,
    ticks_to_usd,
)
from token_optimizer.grok import BASE_URL, _api_key, _load_dotenv  # noqa: E402

OUT_PATH = ROOT / "ab_full_xai_matrix_last.json"
ARTIFACTS = ROOT / "matrix_artifacts"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe(obj: Any) -> Any:
    """JSON-friendly + strip anything key-like."""
    if isinstance(obj, dict):
        return {
            k: _safe(v)
            for k, v in obj.items()
            if "api_key" not in k.lower() and k.lower() not in ("authorization",)
        }
    if isinstance(obj, (list, tuple)):
        return [_safe(x) for x in obj]
    if isinstance(obj, float):
        return round(obj, 12)
    return obj


def _row(
    name: str,
    *,
    ok: bool,
    modality: str,
    **kwargs: Any,
) -> dict[str, Any]:
    return {"name": name, "modality": modality, "ok": ok, **kwargs}


def test_text_profile(profile: str, prompt: str = "Reply with exactly: pong") -> dict[str, Any]:
    t0 = time.perf_counter()
    s = GrokSession(profile=profile, max_tokens_out=48, live_prices=True)
    r = s.chat(prompt)
    elapsed = time.perf_counter() - t0
    cmp = s.compare_cost_accuracy()
    return _row(
        f"text/{profile}",
        ok=bool(r.text) and (cmp.get("within_99_5") in (True, False, None)),
        modality="text",
        model=r.model,
        profile=profile,
        text=(r.text or "")[:80],
        api_total=r.api_total,
        cost_in_usd_ticks=r.usage.get("cost_in_usd_ticks"),
        api_cost_usd=r.cost_usd_api,
        est_cost_usd=r.cost_usd_est,
        cost_accuracy=cmp.get("accuracy"),
        within_99_5=cmp.get("within_99_5"),
        seconds=round(elapsed, 3),
        benefit="full: compact + usage + ticks $ + profiles",
    )


def test_text_stream() -> dict[str, Any]:
    t0 = time.perf_counter()
    s = GrokSession(profile="production", max_tokens_out=48, live_prices=True)
    parts: list[str] = []
    for d in s.chat_stream("Reply with exactly: stream-ok"):
        parts.append(d)
    r = s.last_result
    elapsed = time.perf_counter() - t0
    assert r is not None
    cmp = s.compare_cost_accuracy()
    return _row(
        "text/stream",
        ok=bool("".join(parts)) and r.api_total > 0,
        modality="text",
        model=r.model,
        text=(r.text or "")[:80],
        deltas=len(parts),
        api_total=r.api_total,
        cost_in_usd_ticks=r.usage.get("cost_in_usd_ticks"),
        api_cost_usd=r.cost_usd_api,
        est_cost_usd=r.cost_usd_est,
        within_99_5=cmp.get("within_99_5"),
        seconds=round(elapsed, 3),
        benefit="full: stream deltas + final usage/ticks",
    )


def test_per_call_override() -> dict[str, Any]:
    s = GrokSession(profile="production", max_tokens_out=32, live_prices=True)
    base = s.opt.model
    r = s.chat("Reply with exactly: ok", model="grok-4.5", effort="high")
    restored = s.opt.model
    return _row(
        "text/per_call_frontier_override",
        ok=bool(r.model) and "4.5" in str(r.model) and restored == base,
        modality="text",
        model=r.model,
        restored_model=restored,
        text=(r.text or "")[:40],
        api_total=r.api_total,
        cost_in_usd_ticks=r.usage.get("cost_in_usd_ticks"),
        api_cost_usd=r.cost_usd_api,
        est_cost_usd=r.cost_usd_est,
        within_99_5=s.compare_cost_accuracy().get("within_99_5"),
        benefit="full: mix 4.5 design turn inside production session",
    )


def test_grok_build() -> dict[str, Any]:
    """Grok Build is a chat/completions coding model — full optimizer applies."""
    t0 = time.perf_counter()
    s = GrokSession(
        profile="bulk",
        model="grok-build-0.1",
        max_tokens_out=128,
        live_prices=True,
    )
    # Fat tool-style dump → compact should still apply via prepare_turn
    fat = (
        "FILE services/tax.py\n"
        + ("def grand_total(order):\n    # duplicate tax bug\n" * 40)
        + "ERROR: NameError: legacy_cart_total is not defined\n"
        "Fix the NameError in one short sentence."
    )
    r = s.chat(fat)
    elapsed = time.perf_counter() - t0
    cmp = s.compare_cost_accuracy()
    return _row(
        "build/grok-build-0.1",
        ok=bool(r.text) and r.api_total > 0,
        modality="code/agent",
        model=r.model,
        text=(r.text or "")[:120],
        api_total=r.api_total,
        cost_in_usd_ticks=r.usage.get("cost_in_usd_ticks"),
        api_cost_usd=r.cost_usd_api,
        est_cost_usd=r.cost_usd_est,
        within_99_5=cmp.get("within_99_5"),
        cost_accuracy=cmp.get("accuracy"),
        seconds=round(elapsed, 3),
        rates={"pin": s.opt.pin, "pout": s.opt.pout, "pin_cached": s.opt.pin_cached},
        benefit="full: same as text — compact fat source dumps, ticks $, live sheet",
    )


def test_imagine_image(n: int = 1) -> dict[str, Any]:
    """Flat per-image pricing — prompt compact helps a little; $ is not token ticks."""
    _load_dotenv()
    key = _api_key()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model = "grok-imagine-image"  # cheaper sheet than -quality
    prompt = "Minimal flat icon of a green checkmark on dark background, simple"
    body = {
        "model": model,
        "prompt": prompt,
        "n": n,
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{BASE_URL.rstrip('/')}/images/generations",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            data = json.loads(raw.decode("utf-8"))
            status = resp.status
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:500]
        return _row(
            "imagine/image",
            ok=False,
            modality="image",
            model=model,
            error=f"HTTP {e.code}: {err}",
            benefit="partial: compact prompts before generate; $ is flat/image not tokens",
        )
    elapsed = time.perf_counter() - t0

    # Catalog price: image_price field / 1e10 → USD (same tick scale as text cost)
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=BASE_URL)
    meta = client.models.retrieve(model)
    md = meta.model_dump() if hasattr(meta, "model_dump") else {}
    image_price_ticks = md.get("image_price")
    sheet_usd = ticks_to_usd(image_price_ticks) if image_price_ticks else None
    if sheet_usd is not None:
        sheet_usd = sheet_usd * n

    urls = []
    for i, item in enumerate(data.get("data") or []):
        u = item.get("url")
        b64 = item.get("b64_json")
        if u:
            urls.append(u[:120] + ("…" if len(u) > 120 else ""))
        if b64:
            p = ARTIFACTS / f"imagine_{i}.b64.txt"
            p.write_text(b64[:200] + "…", encoding="utf-8")  # stub only
            urls.append(f"b64:{p.name}")

    usage = data.get("usage") or {}
    return _row(
        "imagine/image",
        ok=status == 200 and bool(data.get("data")),
        modality="image",
        model=model,
        n=n,
        seconds=round(elapsed, 3),
        image_price_ticks=image_price_ticks,
        sheet_usd_per_batch=sheet_usd,
        ticks_per_usd=XAI_USD_TICKS_PER_DOLLAR,
        response_usage=usage if usage else None,
        urls_or_refs=urls[:3],
        response_keys=sorted(data.keys()),
        benefit=(
            "partial: compact long design briefs before /images/generations; "
            "bill is flat per image (image_price ticks/1e10), not chat tokens; "
            "no cost_in_usd_ticks on typical image response"
        ),
    )


def test_voice_tts() -> dict[str, Any]:
    """TTS is character-priced — compact the script before speak."""
    _load_dotenv()
    key = _api_key()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    # Naive long script vs compact script (benefit demo)
    long_script = (
        "Hello. This is a demonstration of the Grok text to speech API. "
        "We are verifying that TokenOptimizer can still help by shortening "
        "the text that gets sent to the voice endpoint. " * 3
    )
    compact_script = "Hello. Grok TTS matrix probe."
    # Use compact for the live call (save $); report naive char count
    text = compact_script
    body = {
        "text": text,
        "voice_id": "eve",
        "language": "en",
    }
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{BASE_URL.rstrip('/')}/tts",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            audio = resp.read()
            status = resp.status
            ct = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:500]
        return _row(
            "voice/tts",
            ok=False,
            modality="audio",
            error=f"HTTP {e.code}: {err}",
            benefit="partial: compact TTS script chars; $ is per-character not tokens",
        )
    elapsed = time.perf_counter() - t0
    out = ARTIFACTS / "tts_probe.mp3"
    out.write_bytes(audio)

    # Docs: ~$15 / 1M chars
    tts_rate_per_m = 15.0
    est_compact = len(text) / 1e6 * tts_rate_per_m
    est_naive = len(long_script) / 1e6 * tts_rate_per_m
    save_pct = (
        (1 - est_compact / est_naive) * 100 if est_naive else 0.0
    )

    return _row(
        "voice/tts",
        ok=status == 200 and len(audio) > 100,
        modality="audio",
        voice_id="eve",
        language="en",
        content_type=ct,
        audio_bytes=len(audio),
        artifact=str(out.relative_to(ROOT)),
        chars_sent=len(text),
        chars_naive_demo=len(long_script),
        est_usd_sent=round(est_compact, 8),
        est_usd_if_naive=round(est_naive, 8),
        compact_char_save_pct=round(save_pct, 1),
        seconds=round(elapsed, 3),
        benefit=(
            "partial: shorten/compact spoken scripts before /tts "
            f"(demo ~{save_pct:.0f}% fewer chars → ~{save_pct:.0f}% lower TTS $); "
            "realtime voice agent still needs short instructions + tool compact"
        ),
    )


def benefit_matrix() -> list[dict[str, Any]]:
    return [
        {
            "modality": "Text chat (4.3 / 4.5 / bulk)",
            "token_optimizer": "full",
            "what_helps": "compact_context, tool_compact, profiles, usage observe, ticks $",
            "cost_truth": "cost_in_usd_ticks / 1e10",
        },
        {
            "modality": "Grok Build (grok-build-0.1)",
            "token_optimizer": "full",
            "what_helps": "same as text — agent loops dump huge files; compact is the 90%+ lever",
            "cost_truth": "cost_in_usd_ticks / 1e10 + live sheet rates",
        },
        {
            "modality": "Stream chat",
            "token_optimizer": "full",
            "what_helps": "same compact before stream; usage on final chunk",
            "cost_truth": "cost_in_usd_ticks when provided",
        },
        {
            "modality": "Imagine image",
            "token_optimizer": "partial",
            "what_helps": "compact long briefs; batch n carefully; pick imagine-image vs quality",
            "cost_truth": "flat image_price ticks/1e10 per image (not chat tokens)",
        },
        {
            "modality": "Imagine video",
            "token_optimizer": "partial",
            "what_helps": "short prompts, shorter duration, lower resolution",
            "cost_truth": "per-second video pricing (not token ticks)",
        },
        {
            "modality": "Voice TTS",
            "token_optimizer": "partial",
            "what_helps": "compact_context / emit short scripts before /tts",
            "cost_truth": "~$15/1M characters (not token ticks)",
        },
        {
            "modality": "Voice realtime agent",
            "token_optimizer": "partial",
            "what_helps": "short session instructions, tool_compact on tool results, avoid monologue system prompts",
            "cost_truth": "per-minute realtime + separate tool/LLM charges",
        },
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-imagine", action="store_true")
    ap.add_argument("--skip-tts", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-text", action="store_true")
    args = ap.parse_args()

    _load_dotenv()
    try:
        _api_key()
    except Exception as e:
        print("API key missing:", e)
        return 2

    results: list[dict[str, Any]] = []
    print("=" * 72)
    print("FULL xAI MATRIX TEST")
    print("=" * 72)

    # --- Text ---
    if not args.skip_text:
        for profile in ("production", "innovate", "frontier", "bulk"):
            print(f"\n>> text/{profile} …")
            try:
                row = test_text_profile(profile)
            except Exception as e:
                row = _row(f"text/{profile}", ok=False, modality="text", error=str(e)[:300])
            results.append(row)
            _print_row(row)

        print("\n>> text/stream …")
        try:
            row = test_text_stream()
        except Exception as e:
            row = _row("text/stream", ok=False, modality="text", error=str(e)[:300])
        results.append(row)
        _print_row(row)

        print("\n>> text/per_call_frontier_override …")
        try:
            row = test_per_call_override()
        except Exception as e:
            row = _row(
                "text/per_call_frontier_override",
                ok=False,
                modality="text",
                error=str(e)[:300],
            )
        results.append(row)
        _print_row(row)

    # --- Build ---
    if not args.skip_build:
        print("\n>> build/grok-build-0.1 …")
        try:
            row = test_grok_build()
        except Exception as e:
            row = _row("build/grok-build-0.1", ok=False, modality="code/agent", error=str(e)[:300])
        results.append(row)
        _print_row(row)

    # --- Imagine ---
    if not args.skip_imagine:
        print("\n>> imagine/image …")
        try:
            row = test_imagine_image()
        except Exception as e:
            row = _row("imagine/image", ok=False, modality="image", error=str(e)[:300])
        results.append(row)
        _print_row(row)

    # --- Voice ---
    if not args.skip_tts:
        print("\n>> voice/tts …")
        try:
            row = test_voice_tts()
        except Exception as e:
            row = _row("voice/tts", ok=False, modality="audio", error=str(e)[:300])
        results.append(row)
        _print_row(row)

    benefits = benefit_matrix()
    ok_n = sum(1 for r in results if r.get("ok"))
    fail_n = len(results) - ok_n

    # Dollar accuracy on rows that have both api+est
    tick_rows = [
        r
        for r in results
        if r.get("api_cost_usd") is not None and r.get("est_cost_usd") is not None
    ]
    tick_ok = sum(1 for r in tick_rows if r.get("within_99_5"))

    report = {
        "ts": _now(),
        "summary": {
            "total": len(results),
            "ok": ok_n,
            "failed": fail_n,
            "tick_metered_calls": len(tick_rows),
            "tick_within_99_5": tick_ok,
            "ticks_per_usd": XAI_USD_TICKS_PER_DOLLAR,
        },
        "results": _safe(results),
        "benefit_matrix": benefits,
        "product": {
            "providers": [
                {"id": p["id"], "status": p["status"]}
                for p in product_catalog()["providers"]
            ]
        },
    }
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  tests: {ok_n}/{len(results)} ok  ({fail_n} failed)")
    print(f"  tick-metered: {tick_ok}/{len(tick_rows)} within 99.5%")
    print(f"  wrote: {OUT_PATH.name}")
    print("\nBENEFIT MATRIX")
    for b in benefits:
        print(f"  [{b['token_optimizer']:7}] {b['modality']}")
        print(f"            {b['what_helps']}")
        print(f"            cost: {b['cost_truth']}")
    print("=" * 72)
    return 0 if fail_n == 0 else 1


def _print_row(row: dict[str, Any]) -> None:
    flag = "OK " if row.get("ok") else "FAIL"
    bits = [flag, row.get("name", "?")]
    if row.get("model"):
        bits.append(f"model={row['model']}")
    if row.get("api_total") is not None:
        bits.append(f"tok={row['api_total']}")
    if row.get("api_cost_usd") is not None:
        bits.append(f"api=${row['api_cost_usd']}")
    if row.get("est_cost_usd") is not None:
        bits.append(f"est=${row['est_cost_usd']}")
    if row.get("within_99_5") is not None:
        bits.append(f"99.5%={row['within_99_5']}")
    if row.get("sheet_usd_per_batch") is not None:
        bits.append(f"sheet_img=${row['sheet_usd_per_batch']}")
    if row.get("audio_bytes") is not None:
        bits.append(f"audio={row['audio_bytes']}B")
    if row.get("compact_char_save_pct") is not None:
        bits.append(f"char_save={row['compact_char_save_pct']}%")
    if row.get("error"):
        bits.append(f"err={row['error'][:120]}")
    print("  " + " | ".join(str(b) for b in bits))


if __name__ == "__main__":
    raise SystemExit(main())

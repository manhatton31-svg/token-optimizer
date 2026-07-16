"""
Frontier model tiers for heavy token users.

Pattern many teams use:
  1) Plan / design / hard debug on a TOP frontier model (e.g. grok-4.5)
  2) Distill workflows into compact loops on CHEAPER models (e.g. grok-4.3,
     fast variants) with TokenOptimizer / OptimizedLoop for volume.

Prices: USD per 1M tokens. For Grok, prefer live fetch via fetch_xai_model_pricing().
Fallback numbers below match xAI models API field/10000 as of last calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    tier: str  # frontier | standard | fast | image | other
    label: str
    pin: float  # USD / 1M input
    pout: float  # USD / 1M output (+ reasoning when billed as output)
    pin_cached: float  # USD / 1M cached input
    role: str  # design | production | bulk | multimodal
    notes: str = ""

    @property
    def cost_index(self) -> float:
        """Rough relative cost vs grok-4.3 blended (in+out)."""
        base = 1.25 + 2.50
        return (self.pin + self.pout) / base if base else 1.0


# Live-verified / sheet defaults (Grok). Update via fetch_xai_model_pricing.
GROK_MODELS: dict[str, ModelSpec] = {
    "grok-4.5": ModelSpec(
        id="grok-4.5",
        provider="grok",
        tier="frontier",
        label="Grok 4.5 (flagship)",
        pin=2.00,
        pout=6.00,
        pin_cached=0.50,
        role="design",
        notes="Top quality; use for architecture, hard bugs, distillation targets.",
    ),
    "grok-4.3": ModelSpec(
        id="grok-4.3",
        provider="grok",
        tier="standard",
        label="Grok 4.3 (calibrated)",
        pin=1.25,
        pout=2.50,
        pin_cached=0.20,
        role="production",
        notes="Session-calibrated @ 100% token match. Default production path.",
    ),
    "grok-4.20-0309-reasoning": ModelSpec(
        id="grok-4.20-0309-reasoning",
        provider="grok",
        tier="standard",
        label="Grok 4.20 reasoning",
        pin=1.25,
        pout=2.50,
        pin_cached=0.20,
        role="production",
        notes="Explicit reasoning SKU; same ballpark as 4.3.",
    ),
    "grok-4.20-0309-non-reasoning": ModelSpec(
        id="grok-4.20-0309-non-reasoning",
        provider="grok",
        tier="fast",
        label="Grok 4.20 non-reasoning",
        pin=1.25,
        pout=2.50,
        pin_cached=0.20,
        role="bulk",
        notes="Same rates but no reasoning tokens → much lower total on agents.",
    ),
    "grok-4.20-multi-agent-0309": ModelSpec(
        id="grok-4.20-multi-agent-0309",
        provider="grok",
        tier="standard",
        label="Grok 4.20 multi-agent",
        pin=1.25,
        pout=2.50,
        pin_cached=0.20,
        role="production",
        notes="Multi-agent SKU; still compact context aggressively.",
    ),
    "grok-build-0.1": ModelSpec(
        id="grok-build-0.1",
        provider="grok",
        tier="fast",
        label="Grok Build 0.1",
        pin=1.00,
        pout=2.00,
        pin_cached=0.20,
        role="bulk",
        notes="Lower sheet rates; good for high-volume optimized loops.",
    ),
}

# Cross-frontier examples heavy users often mix (presets; calibrate when keyed)
FRONTIER_CATALOG: dict[str, ModelSpec] = {
    **GROK_MODELS,
    "gpt-4o": ModelSpec(
        "gpt-4o", "openai", "frontier", "GPT-4o", 2.50, 10.00, 1.25, "design"
    ),
    "gpt-4o-mini": ModelSpec(
        "gpt-4o-mini", "openai", "fast", "GPT-4o-mini", 0.15, 0.60, 0.075, "bulk"
    ),
    "claude-sonnet-4-20250514": ModelSpec(
        "claude-sonnet-4-20250514",
        "anthropic",
        "frontier",
        "Claude Sonnet",
        3.00,
        15.00,
        0.30,
        "design",
    ),
    "claude-haiku-4-20250414": ModelSpec(
        "claude-haiku-4-20250414",
        "anthropic",
        "fast",
        "Claude Haiku",
        0.80,
        4.00,
        0.08,
        "bulk",
    ),
    "gemini-2.0-flash": ModelSpec(
        "gemini-2.0-flash", "gemini", "fast", "Gemini Flash", 0.10, 0.40, 0.025, "bulk"
    ),
    "gemini-1.5-pro": ModelSpec(
        "gemini-1.5-pro", "gemini", "frontier", "Gemini Pro", 1.25, 5.00, 0.31, "design"
    ),
}


def get_model(model_id: str) -> ModelSpec | None:
    return FRONTIER_CATALOG.get(model_id) or GROK_MODELS.get(model_id)


def models_for_provider(provider: str) -> list[ModelSpec]:
    return [m for m in FRONTIER_CATALOG.values() if m.provider == provider]


def recommend_stack(provider: str = "grok") -> dict[str, ModelSpec]:
    """
    Suggested heavy-user stack: design on frontier, ship on standard/fast.
    """
    pool = models_for_provider(provider)
    design = next((m for m in pool if m.role == "design"), None)
    prod = next((m for m in pool if m.role == "production"), None)
    bulk = next((m for m in pool if m.role == "bulk"), None)
    if provider == "grok":
        design = get_model("grok-4.5")
        prod = get_model("grok-4.3")
        bulk = get_model("grok-4.20-0309-non-reasoning") or get_model("grok-build-0.1")
    return {
        "design": design,
        "production": prod,
        "bulk": bulk,
    }


def estimate_pair_cost(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
    model: ModelSpec,
) -> float:
    uncached = max(0, prompt_tokens - cached_tokens)
    return (
        uncached / 1e6 * model.pin
        + cached_tokens / 1e6 * model.pin_cached
        + (completion_tokens + reasoning_tokens) / 1e6 * model.pout
    )


def compare_model_costs(
    usage: dict[str, int],
    models: Iterable[str | ModelSpec],
) -> list[dict[str, Any]]:
    """What the same usage would cost on each model (for tier planning)."""
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    reasoning = int(usage.get("reasoning_tokens") or 0)
    cached = int(usage.get("cached_tokens") or 0)
    rows = []
    for m in models:
        spec = get_model(m) if isinstance(m, str) else m
        if not spec:
            continue
        cost = estimate_pair_cost(
            prompt,
            completion,
            reasoning_tokens=reasoning,
            cached_tokens=cached,
            model=spec,
        )
        rows.append(
            {
                "model": spec.id,
                "tier": spec.tier,
                "role": spec.role,
                "cost_usd": cost,
                "pin": spec.pin,
                "pout": spec.pout,
                "relative_to_4.3": spec.cost_index,
            }
        )
    rows.sort(key=lambda r: r["cost_usd"])
    return rows


def print_heavy_user_guide() -> str:
    stack = recommend_stack("grok")
    lines = [
        "HEAVY TOKEN USER — GROK TIERING",
        "=" * 60,
        "Pattern: design on frontier → compact with TokenOptimizer → run volume cheaper",
        "",
        f"  DESIGN     {stack['design'].id if stack['design'] else '?':28} "
        f"${stack['design'].pin:.2f}/${stack['design'].pout:.2f} per 1M"
        if stack.get("design")
        else "",
        f"  PRODUCTION {stack['production'].id if stack['production'] else '?':28} "
        f"${stack['production'].pin:.2f}/${stack['production'].pout:.2f} per 1M  [calibrated]"
        if stack.get("production")
        else "",
        f"  BULK       {stack['bulk'].id if stack['bulk'] else '?':28} "
        f"${stack['bulk'].pin:.2f}/${stack['bulk'].pout:.2f} per 1M"
        if stack.get("bulk")
        else "",
        "",
        "Cost index (in+out) vs grok-4.3:",
    ]
    for mid in ("grok-4.3", "grok-4.5", "grok-build-0.1", "grok-4.20-0309-non-reasoning"):
        m = get_model(mid)
        if m:
            lines.append(
                f"  {m.id:36} ×{m.cost_index:.2f}  "
                f"(${m.pin:.2f} in / ${m.pout:.2f} out)"
            )
    lines += [
        "",
        "Example: same 19,319-token multi-turn job",
        "  (approx scale from last live A/B: heavy prompts + reasoning)",
        "",
    ]
    # scale last verified combined usage
    sample = {
        "prompt_tokens": 14058 + 1366,
        "completion_tokens": 345 + 164,
        "reasoning_tokens": 1259 + 2127,
        "cached_tokens": 1152 + 832,
    }
    for row in compare_model_costs(
        sample, ["grok-4.3", "grok-4.5", "grok-build-0.1"]
    ):
        lines.append(
            f"  {row['model']:36} ~${row['cost_usd']:.4f}  tier={row['tier']}"
        )
    lines += [
        "",
        "OptimizedLoop on production/bulk is where 90%+ context savings land.",
        "Use 4.5 sparingly for hard steps; don't put full monorepos on 4.5 every turn.",
        "=" * 60,
    ]
    return "\n".join(lines)

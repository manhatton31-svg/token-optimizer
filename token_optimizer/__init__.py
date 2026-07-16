"""
token-optimizer
---------------
Model-agnostic token budgeting and context compaction for agent loops.
"""

from .core import LoopStep, OptimizedLoop, TokenOptimizer, tool_compact
from .tokenizers import (
    COST_PRESETS,
    approx_tokens,
    list_tokenizers,
    register_tokenizer,
    resolve_provider,
    resolve_tokenizer,
)
from .costing import (
    XAI_USD_TICKS_PER_DOLLAR,
    compare_cost_accuracy,
    cost_within_tolerance,
    estimate_cost_usd,
    fetch_xai_model_pricing,
    normalize_usage,
    parse_cost_in_usd_ticks,
    ticks_to_usd,
)
from .onboarding import (
    OnboardingService,
    TARGET_ACCURACY,
    bootstrap_grok_lock_from_session,
    verify_token_delta,
)
from .models import (
    FRONTIER_CATALOG,
    GROK_MODELS,
    ModelSpec,
    compare_model_costs,
    get_model,
    print_heavy_user_guide,
    recommend_stack,
)
from .diy import adoption_levels, example_custom_loop
from .grok import (
    ChatResult,
    GrokSession,
    list_profiles,
    product_catalog,
    warn_fat_prompt,
)

__all__ = [
    "TokenOptimizer",
    "OptimizedLoop",
    "LoopStep",
    "tool_compact",
    "register_tokenizer",
    "resolve_tokenizer",
    "resolve_provider",
    "approx_tokens",
    "list_tokenizers",
    "COST_PRESETS",
    "estimate_cost_usd",
    "fetch_xai_model_pricing",
    "normalize_usage",
    "cost_within_tolerance",
    "compare_cost_accuracy",
    "ticks_to_usd",
    "parse_cost_in_usd_ticks",
    "XAI_USD_TICKS_PER_DOLLAR",
    "OnboardingService",
    "TARGET_ACCURACY",
    "bootstrap_grok_lock_from_session",
    "verify_token_delta",
    "ModelSpec",
    "GROK_MODELS",
    "FRONTIER_CATALOG",
    "get_model",
    "recommend_stack",
    "compare_model_costs",
    "print_heavy_user_guide",
    "adoption_levels",
    "example_custom_loop",
    "GrokSession",
    "ChatResult",
    "list_profiles",
    "product_catalog",
    "warn_fat_prompt",
]
__version__ = "0.2.0"

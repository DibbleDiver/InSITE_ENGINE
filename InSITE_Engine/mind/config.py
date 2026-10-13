"""Typed view of cfg["mind"] with defaults for every key in config.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_MODEL = "claude-fable-5-1"
NULLABLE_KEYS = {"budget_usd_total", "fallbacks"}

# Conservative default prices (USD per million tokens) used for any model that
# is missing from cfg mind.pricing_usd_per_mtok: Claude Fable 5.1 list prices.
FABLE_PRICES = {"input": 10.00, "output": 50.00, "cache_write": 12.50, "cache_read": 0.25}


@dataclass
class MindConfig:
    model: str = DEFAULT_MODEL
    effort: str = "high"
    critic_model: str = DEFAULT_MODEL
    critic_effort: str = "medium"
    max_tokens: int = 64000
    max_steps: int = 40
    budget_usd_per_cycle: float = 5.0
    budget_usd_total: float | None = None
    thinking_display: str = "summarized"
    fallbacks: str | None = "default"
    eager_input_streaming: bool = True
    memory_dir: str = "InSITE_Engine/output/mind"
    loop_interval_seconds: float = 900.0
    max_idle_cycles: int = 2
    pricing_usd_per_mtok: dict = field(default_factory=lambda: {DEFAULT_MODEL: dict(FABLE_PRICES)})

    @classmethod
    def from_cfg(cls, cfg: dict | None) -> "MindConfig":
        mind = dict((cfg or {}).get("mind") or {})
        kwargs = {}
        for name in cls.__dataclass_fields__:
            if name not in mind:
                continue
            value = mind[name]
            if value is None and name not in NULLABLE_KEYS:
                continue  # null for a non-nullable key means "use the default"
            kwargs[name] = value
        config = cls(**kwargs)
        config.validate()
        return config

    def validate(self) -> None:
        if self.effort not in EFFORT_LEVELS:
            raise ValueError(f"mind.effort must be one of {EFFORT_LEVELS}, got {self.effort!r}")
        if self.critic_effort not in EFFORT_LEVELS:
            raise ValueError(f"mind.critic_effort must be one of {EFFORT_LEVELS}, got {self.critic_effort!r}")
        if self.thinking_display not in ("summarized", "omitted"):
            raise ValueError("mind.thinking_display must be 'summarized' or 'omitted'")
        if self.fallbacks not in (None, "default"):
            raise ValueError("mind.fallbacks must be 'default' or null")
        if self.max_steps < 1:
            raise ValueError("mind.max_steps must be at least 1")
        if self.budget_usd_per_cycle <= 0:
            raise ValueError("mind.budget_usd_per_cycle must be positive")

    def prices_for(self, model: str | None) -> dict:
        """Per-million-token prices for `model`; unknown models get the Fable prices."""
        entry = (self.pricing_usd_per_mtok or {}).get(model or "")
        if not entry:
            return dict(FABLE_PRICES)
        return {k: float(entry.get(k, FABLE_PRICES[k])) for k in FABLE_PRICES}

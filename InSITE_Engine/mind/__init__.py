"""The Mind: the autonomous, self-directed reasoning layer of the InSITE Engine.

    from InSITE_Engine.mind import Mind, MindUnavailable
    mind = Mind(cfg)                      # real Claude client from cfg["mind"]; MindUnavailable without credentials
    outcome = mind.think()                # one cognitive cycle -> CycleOutcome
    outcome = mind.ask("why did pressure drop on day 3?")
    mind.live(interval_seconds=900, max_cycles=None, budget_usd_total=None)
"""
from .config import MindConfig
from .llm import AnthropicLLM, MindUnavailable
from .mind import CycleOutcome, Mind

__all__ = ["Mind", "MindUnavailable", "CycleOutcome", "MindConfig", "AnthropicLLM"]

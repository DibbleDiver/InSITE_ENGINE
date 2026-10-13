"""Per-signal state timelines (normal / anomalous / flatlined / offline)."""
from .states import KIND_TO_STATE, PRECEDENCE, derive_states

__all__ = ["derive_states", "KIND_TO_STATE", "PRECEDENCE"]

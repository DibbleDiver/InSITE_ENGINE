"""Event detection (spikes, step changes, flatlines, gaps)."""
from .events import detect_events, event_id, fmt_duration, severity

__all__ = ["detect_events", "event_id", "fmt_duration", "severity"]

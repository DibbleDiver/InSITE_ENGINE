"""Turn SDK pydantic objects, test fakes and plain dicts into JSON-ready data.

The journal stores the full conversation of every cycle. Content blocks arrive
either as the SDK's pydantic models (`model_dump`), as the dataclass fakes in
`mind.testing`, or as the plain dicts the harness builds itself (tool results,
text notes). Everything funnels through `to_jsonable`.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Recursively convert `value` into something json.dumps accepts."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):  # pydantic (SDK blocks, usage, stop_details)
        try:
            return to_jsonable(value.model_dump(mode="json", by_alias=True))
        except Exception:  # pragma: no cover - defensive: fall through to generic handling
            pass
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (_dt.datetime, _dt.date)):
        return value.isoformat()
    if hasattr(value, "isoformat"):  # pandas Timestamp
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover
            return str(value)
    if hasattr(value, "item"):  # numpy scalars
        try:
            return to_jsonable(value.item())
        except Exception:  # pragma: no cover
            return str(value)
    if hasattr(value, "__dict__"):
        return {k: to_jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def block_to_dict(block: Any) -> dict:
    """A content block (SDK model, fake or dict) as a plain dict with a 'type' key."""
    data = to_jsonable(block)
    if not isinstance(data, dict):
        return {"type": "unknown", "value": data}
    if "type" not in data:
        data["type"] = getattr(block, "type", "unknown")
    return data


def message_to_dict(message: dict) -> dict:
    content = message.get("content")
    if isinstance(content, str):
        return {"role": message.get("role"), "content": content}
    return {"role": message.get("role"), "content": [block_to_dict(b) for b in (content or [])]}


def messages_to_dicts(messages: list[dict]) -> list[dict]:
    return [message_to_dict(m) for m in messages]


def block_type(block: Any) -> str | None:
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def dumps(value: Any, **kwargs) -> str:
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(to_jsonable(value), **kwargs)

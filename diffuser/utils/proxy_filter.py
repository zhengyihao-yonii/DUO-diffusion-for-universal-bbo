"""PROXY_FILTER / proxy_filter：是否训练 proxy 并在评估中用其打分筛选 queries（0=关闭，仅扩散采样后 eval）。"""

from __future__ import annotations

import os
from typing import Any, Mapping


def _coerce_bool(v: Any, *, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("0", "false", "no", "off"):
        return False
    if s in ("1", "true", "yes", "on"):
        return True
    try:
        return int(v) != 0
    except (TypeError, ValueError):
        return default


def resolve_proxy_filter_for_eval(deps: Mapping[str, Any]) -> bool:
    """
    评估路径：real_task_zero_shot_eval 时返回 False（不加载/不训 proxy，仅扩散采样后 eval）。
    否则 deps['proxy_filter'] 优先，其次环境变量 PROXY_FILTER（默认 1=开启筛选）。
    """
    if deps.get("real_task_zero_shot_eval"):
        return False
    v = deps.get("proxy_filter")
    if v is None:
        v = os.environ.get("PROXY_FILTER", "1")
    return _coerce_bool(v, default=True)


def proxy_filter_enabled_for_train(deps: Mapping[str, Any]) -> bool:
    """训练路径：无 zero-shot 特例；默认开启 proxy。"""
    v = deps.get("proxy_filter")
    if v is None:
        v = os.environ.get("PROXY_FILTER", "1")
    return _coerce_bool(v, default=True)

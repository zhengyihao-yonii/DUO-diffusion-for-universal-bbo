"""Force project wandb login before init (override shared-server ~/.netrc / env)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_DEFAULT_ENTITY = "1585515136-"
_DEFAULT_BASE_URL = "https://api.wandb.ai/"


def duo_root() -> Path:
    """DUO repository root."""
    return Path(__file__).resolve().parents[2]


def _parse_wandb_local() -> dict[str, str]:
    """Parse ``config/wandb_local.sh`` (project-owned credentials)."""
    local = duo_root() / "config" / "wandb_local.sh"
    out: dict[str, str] = {}
    if not local.is_file():
        return out
    text = local.read_text(encoding="utf-8")
    for name in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_BASE_URL"):
        match = re.search(rf"{name}=['\"]?([^'\"\n]+)", text)
        if match:
            out[name] = match.group(1).strip()
    return out


def _use_shell_env_wandb() -> bool:
    """Opt-in: allow pre-exported WANDB_* to override project config."""
    return os.environ.get("DUO_WANDB_USE_ENV", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def load_wandb_api_key() -> str:
    """Project ``wandb_local.sh`` wins over shell env (shared-account safe)."""
    local = _parse_wandb_local()
    if local.get("WANDB_API_KEY") and not _use_shell_env_wandb():
        return local["WANDB_API_KEY"]
    key = os.environ.get("WANDB_API_KEY", "").strip()
    if len(key) >= 40:
        return key
    return local.get("WANDB_API_KEY", "")


def get_wandb_entity() -> str:
    """W&B entity; project config wins over shell env."""
    local = _parse_wandb_local()
    if local.get("WANDB_ENTITY") and not _use_shell_env_wandb():
        return local["WANDB_ENTITY"]
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    if entity:
        return entity
    return local.get("WANDB_ENTITY", _DEFAULT_ENTITY)


def get_wandb_base_url() -> str:
    """W&B API URL; project config wins over ``.bashrc`` exports."""
    local = _parse_wandb_local()
    if local.get("WANDB_BASE_URL") and not _use_shell_env_wandb():
        return local["WANDB_BASE_URL"]
    url = os.environ.get("WANDB_BASE_URL", "").strip()
    if url:
        return url
    return local.get("WANDB_BASE_URL", _DEFAULT_BASE_URL)


def ensure_wandb_login() -> bool:
    """Set WANDB_* from project config, then login with project key."""
    key = load_wandb_api_key()
    if len(key) < 40:
        print("[wandb] WANDB_API_KEY missing; set config/wandb_local.sh", flush=True)
        return False
    # 中文注释: 强制覆盖共享 shell / .bashrc / 他人 export 的 WANDB_*（在 import wandb 之前）
    os.environ["WANDB_API_KEY"] = key
    os.environ["WANDB_BASE_URL"] = get_wandb_base_url()
    os.environ["WANDB_ENTITY"] = get_wandb_entity()
    return True


def init_wandb_run(
    project: str,
    entity: str | None = None,
    **kwargs: Any,
) -> Any:
    """``ensure_wandb_login`` + ``wandb.login(key=...)`` + ``wandb.init(entity=...)``."""
    if os.environ.get("WANDB_DISABLED", "").strip().lower() in ("1", "true", "yes"):
        raise RuntimeError("WANDB_DISABLED=1")

    if not ensure_wandb_login():
        raise RuntimeError("WANDB_API_KEY not configured in config/wandb_local.sh")

    import wandb

    try:
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    except Exception as exc:
        print(f"[wandb] login warn (init uses WANDB_API_KEY): {exc}", flush=True)

    ent = (entity if entity is not None else os.environ["WANDB_ENTITY"]).strip()
    init_kw = dict(kwargs)
    init_kw["project"] = str(project).strip()
    if ent:
        init_kw["entity"] = ent
    return wandb.init(**init_kw)

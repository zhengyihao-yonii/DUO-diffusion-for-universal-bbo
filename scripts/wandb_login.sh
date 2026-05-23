#!/usr/bin/env bash
# English: Export project WANDB_* from config/wandb_local.sh (no ~/.netrc write).
# 中文注释: 共享 Linux 账号下勿用 ``wandb login`` 写 netrc，避免覆盖他人凭证。

_wandb_login_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_wandb_duo_root="$(cd "${_wandb_login_script_dir}/.." && pwd)"
_wandb_local="${_wandb_duo_root}/config/wandb_local.sh"

if [[ ! -f "${_wandb_local}" ]]; then
  echo "[warn] wandb: missing ${_wandb_local}; cp config/wandb_local.sh.example" >&2
  return 0 2>/dev/null || exit 0
fi

# shellcheck source=/dev/null
source "${_wandb_local}"

if [[ -z "${WANDB_API_KEY:-}" || ${#WANDB_API_KEY} -lt 40 ]]; then
  echo "[warn] wandb: invalid WANDB_API_KEY in ${_wandb_local}" >&2
  return 0 2>/dev/null || exit 0
fi

export WANDB_API_KEY WANDB_BASE_URL WANDB_ENTITY
echo "[wandb] project credentials loaded (entity=${WANDB_ENTITY:-?}, base=${WANDB_BASE_URL:-?})" >&2

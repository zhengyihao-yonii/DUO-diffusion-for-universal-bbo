#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# English doc: Copy v3.5 FS step-500 mt_text tail ckpts into v4 bundle (skip re-finetune).
# 中文注释: 从 quality_bundle_3_5/fs_step_sweep/*/step_500 复用 ckpt_mt_text_fs_tail*.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="${V35_SWEEP_ROOT:-${DUO_ROOT}/results/quality_bundle_3_5/fs_step_sweep}"
DST_ROOT="${V4_BUNDLE_ROOT:-${DUO_ROOT}/results/quality_bundle_4}"

for shift in sim_low sim_mid sim_high; do
  src="${SRC_ROOT}/${shift}_3/step_500/fs_checkpoints"
  dst="${DST_ROOT}/shift_${shift}_3/fs_checkpoints"
  if [[ ! -d "${src}" ]]; then
    echo "[error] missing ${src}" >&2
    exit 1
  fi
  mkdir -p "${dst}"
  n=0
  for f in "${src}"/ckpt_mt_text_fs_tail*.pt; do
    [[ -f "${f}" ]] || continue
    cp -f "${f}" "${dst}/"
    n=$((n + 1))
  done
  echo "[seed] ${shift}: copied ${n} mt_text fs ckpts -> ${dst}"
done

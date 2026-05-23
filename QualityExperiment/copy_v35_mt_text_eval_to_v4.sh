#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# English doc: Copy v3.5 step-500 mt_text eval NPZ into v4 artifacts (exact table match).
# 中文注释: 评估 npz 也复用，避免采样随机性导致与 v3.5 表不一致。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="${V35_SWEEP_ROOT:-${DUO_ROOT}/results/quality_bundle_3_5/fs_step_sweep}"
DST_ROOT="${V4_BUNDLE_ROOT:-${DUO_ROOT}/results/quality_bundle_4}"
SEED="${SEED:-0}"

for shift in sim_low sim_mid sim_high; do
  src_eval="${SRC_ROOT}/${shift}_3/step_500/eval"
  src_zs="${SRC_ROOT}/${shift}_3/eval_zs"
  dst_art="${DST_ROOT}/shift_${shift}_3/artifacts_exp1_${shift}_3_seed${SEED}"
  mkdir -p "${dst_art}"
  n=0
  for f in "${src_eval}"/quality_trace_mt_text_*.npz; do
    [[ -f "${f}" ]] || continue
    cp -f "${f}" "${dst_art}/"
    n=$((n + 1))
  done
  if [[ -d "${src_zs}" ]]; then
    for f in "${src_zs}"/quality_trace_mt_text_*.npz; do
      [[ -f "${f}" ]] || continue
      cp -f "${f}" "${dst_art}/"
      n=$((n + 1))
    done
  fi
  echo "[copy] ${shift}: ${n} mt_text npz (fs+zs) -> ${dst_art}"
done

#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# English doc: v3.6 FS step sweep — same 500..2500 milestones as v3.5; LR=1e-5; output bundle_3_6.
# 中文注释: 复用 v3 主训 ckpt；每 500 步微调+评估；结果在 quality_bundle_3_6/fs_step_sweep/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/wandb_login.sh
source "${DUO_ROOT}/scripts/wandb_login.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHON="${PYTHON:-${HOME}/anaconda3/envs/gtg/bin/python}"
export QUAL_CPU_THREADS="${QUAL_CPU_THREADS:-8}"
export OMP_NUM_THREADS="${QUAL_CPU_THREADS}"
export MKL_NUM_THREADS="${QUAL_CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${QUAL_CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${QUAL_CPU_THREADS}"
export TOKENIZERS_PARALLELISM=false

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-duo-quality-suite}"
export QUAL_TRAIN_CONFIG="${QUAL_TRAIN_CONFIG:-config.quality_exp36_train}"

SHIFTS_ARGS=()
if [[ -n "${QUAL36_SHIFTS:-}" ]]; then
  SHIFTS_ARGS=(--shifts "${QUAL36_SHIFTS}")
fi
LR_ARGS=()
if [[ -n "${QUAL_FINETUNE_LR:-}" ]]; then
  LR_ARGS=(--finetune_lr "${QUAL_FINETUNE_LR}")
fi
MS_ARGS=()
if [[ -n "${QUAL36_STEP_MILESTONES:-}" ]]; then
  MS_ARGS=(--milestones "${QUAL36_STEP_MILESTONES}")
fi

exec "${PYTHON}" "${SCRIPT_DIR}/quality_exp35_fs_step_sweep.py" \
  --duo_root "${DUO_ROOT}" \
  --python "${PYTHON}" \
  --sweep_root "${QUAL36_SWEEP_ROOT:-${DUO_ROOT}/results/quality_bundle_3_6/fs_step_sweep}" \
  --train_root "${QUAL_TRAIN_ROOT:-${DUO_ROOT}/results/quality_training_3}" \
  --config_module "${QUAL_TRAIN_CONFIG}" \
  --run_label "${QUAL36_RUN_LABEL:-exp36}" \
  "${LR_ARGS[@]}" \
  "${MS_ARGS[@]}" \
  --device cuda \
  "${SHIFTS_ARGS[@]}" \
  "$@"

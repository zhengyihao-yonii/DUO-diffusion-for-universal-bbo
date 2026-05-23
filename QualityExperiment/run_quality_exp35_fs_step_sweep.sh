#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# English doc: FS finetune step sweep (same LR for tail10/20/50) + rank table for mt_text.
# 中文注释: 复用 v3 主训 ckpt；默认步数网格 500..2500；结果在 quality_bundle_3_5/fs_step_sweep/

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

# Override: QUAL35_STEP_MILESTONES=500,1000,1500  QUAL_FINETUNE_LR=5e-5  QUAL35_SHIFTS=sim_mid,sim_high
SHIFTS_ARGS=()
if [[ -n "${QUAL35_SHIFTS:-}" ]]; then
  SHIFTS_ARGS=(--shifts "${QUAL35_SHIFTS}")
fi
LR_ARGS=(--finetune_lr "${QUAL_FINETUNE_LR:-5e-5}")
MS_ARGS=()
if [[ -n "${QUAL35_STEP_MILESTONES:-}" ]]; then
  MS_ARGS=(--milestones "${QUAL35_STEP_MILESTONES}")
fi

# Override: QUAL35_STEP_MILESTONES=500,1000,1500  QUAL_FINETUNE_LR=5e-5
exec "${PYTHON}" "${SCRIPT_DIR}/quality_exp35_fs_step_sweep.py" \
  --duo_root "${DUO_ROOT}" \
  --python "${PYTHON}" \
  --sweep_root "${QUAL35_SWEEP_ROOT:-${DUO_ROOT}/results/quality_bundle_3_5/fs_step_sweep}" \
  --train_root "${QUAL_TRAIN_ROOT:-${DUO_ROOT}/results/quality_training_3}" \
  --config_module "${QUAL_TRAIN_CONFIG:-config.quality_exp35_train}" \
  --run_label "${QUAL35_RUN_LABEL:-exp35}" \
  "${LR_ARGS[@]}" \
  "${MS_ARGS[@]}" \
  --device cuda \
  "${SHIFTS_ARGS[@]}" \
  "$@"

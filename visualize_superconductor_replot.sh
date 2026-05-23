#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# English doc: Replot superconductor seed-0 sample_viz curves from wandb (dfgo project).
# 中文注释: 在 t∈[1,0] 上做 W&B TWEMA（sqrt(p)，密度归一化 changeInX），再绘图。
#
# 用法:
#   bash visualize_superconductor_replot.sh
#
# 环境变量:
#   PYTHON          默认 ~/anaconda3/envs/gtg/bin/python
#   WANDB_PROJECT   默认 1585515136-/dfgo
#   OUT_DIR         默认 $PROJECT/results/figures
#   SMOOTHING         W&B TWEMA 参数，默认 0.99
#   SMOOTH_X          TWEMA 内部 x 轴：step（默认）或 t

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"

PYTHON="${PYTHON:-/home/xk/anaconda3/envs/gtg/bin/python}"
# shellcheck source=scripts/wandb_login.sh
source "${PROJECT}/scripts/wandb_login.sh"

export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai/}"
WANDB_VIZ_PROJECT="${WANDB_VIZ_PROJECT:-1585515136-/dfgo}"

OUT_DIR="${OUT_DIR:-${PROJECT}/results/figures}"
SMOOTHING="${SMOOTHING:-0.99}"
SMOOTH_X="${SMOOTH_X:-step}"
EPOCHS="${EPOCHS:-250,500,750,1000,1250,1500}"

echo "[info] wandb_project=${WANDB_VIZ_PROJECT} epochs=${EPOCHS} smoothing=${SMOOTHING} smooth_x=${SMOOTH_X} (TWEMA)"
echo "[info] out=${OUT_DIR}/superconductor_seed0"

"${PYTHON}" "${PROJECT}/scripts/plot_superconductor_sample_viz_from_wandb.py" \
  --project "${WANDB_VIZ_PROJECT}" \
  --epochs "${EPOCHS}" \
  --out_dir "${OUT_DIR}" \
  --smoothing "${SMOOTHING}" \
  --smooth-x "${SMOOTH_X}" \
  --task superconductor

echo "[done] figures under ${OUT_DIR}/superconductor_seed0/"

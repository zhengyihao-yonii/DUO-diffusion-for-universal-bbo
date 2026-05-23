#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# English doc: Quality **v4** — v3 main ckpts; FS 500 steps @ 5e-5; reuse v3.5 mt_text; merged LaTeX table.
# 中文注释: 复用 v3 主训与 v3.5 step500 mt_text；其余三模型重新微调+全量评估。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/wandb_login.sh
source "${DUO_ROOT}/scripts/wandb_login.sh"

export QUAL_CPU_THREADS="${QUAL_CPU_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${QUAL_CPU_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${QUAL_CPU_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${QUAL_CPU_THREADS}}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-${QUAL_CPU_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${QUAL_CPU_THREADS}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export QUAL_EXP1_SUFFIX="${QUAL_EXP1_SUFFIX:-_3}"
export TRAIN_D_X="${TRAIN_D_X:-5,6,7,9,10}"
export TEST_D_X="${TEST_D_X:-8}"
export LATENT_DIM="${LATENT_DIM:-8}"
export D_PAD="${D_PAD:-16}"
export EXP1_BRANIN_DOMAIN="${EXP1_BRANIN_DOMAIN:-standard}"
export EXP1_FAMILY_MODE="${EXP1_FAMILY_MODE:-scene_aware}"
export EXP1_SIMILARITY_BLEND="${EXP1_SIMILARITY_BLEND:-0.65}"
export QUAL_TEST_SHIFTS="${QUAL_TEST_SHIFTS:-sim_low,sim_mid,sim_high}"
export QUAL_TRAIN_DATA_SHIFT="${QUAL_TRAIN_DATA_SHIFT:-sim_low}"

export QUAL_BUNDLE_ROOT="${QUAL_BUNDLE_ROOT:-${SCRIPT_DIR}/../results/quality_bundle_4}"
export QUAL_TRAIN_ROOT="${QUAL_TRAIN_ROOT:-${SCRIPT_DIR}/../results/quality_training_3}"
export QUAL_TRAIN_CONFIG="${QUAL_TRAIN_CONFIG:-config.quality_exp1_train}"
export QUAL_REF_TRAIN_DOMAIN="${QUAL_REF_TRAIN_DOMAIN:-${QUAL_TRAIN_ROOT}/dtrain_universal_seed0/eval_train_domain}"
export QUAL_MERGED_OUT="${QUAL_MERGED_OUT:-${QUAL_BUNDLE_ROOT}/analysis_table/quality_best_by_task_merged.tex}"

export SKIP_MAIN_TRAIN="${SKIP_MAIN_TRAIN:-1}"
export FORCE_DATA="${FORCE_DATA:-0}"
export FORCE_EMBED="${FORCE_EMBED:-0}"
export FORCE_TRAIN="${FORCE_TRAIN:-0}"
export FORCE_FINETUNE="${FORCE_FINETUNE:-1}"
# 中文注释: 已 seed 的 mt_text 不要被 --force_finetune 覆盖
export FINETUNE_PRESERVE_VARIANTS="${FINETUNE_PRESERVE_VARIANTS:-mt_text}"
export FORCE_SUITE="${FORCE_SUITE:-1}"
export FORCE_TRAIN_DOMAIN_EVAL="${FORCE_TRAIN_DOMAIN_EVAL:-0}"
# 中文注释: 阶梯图单独 wandb group；suite 只写 npz
export NO_LANDSCAPE_FIGURE="${NO_LANDSCAPE_FIGURE:-1}"

mkdir -p "${QUAL_BUNDLE_ROOT}/analysis_table" "${QUAL_BUNDLE_ROOT}/pipeline_logs"

echo "[step] seed v4 mt_text fs ckpts from v3.5 step_500"
bash "${SCRIPT_DIR}/seed_v4_fs_mt_text_from_v35.sh"

LOG="${QUAL_BUNDLE_ROOT}/pipeline_logs/run_v4_$(date +%Y%m%d_%H%M%S).log"
echo "[info] logging to ${LOG}"
"${SCRIPT_DIR}/run_quality_exp1_pipeline.sh" "$@" 2>&1 | tee -a "${LOG}"

echo "[step] v4 wandb ladder plots (one run per task, 4 figures each)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHON="${PYTHON:-${HOME}/anaconda3/envs/gtg/bin/python}"
"${PYTHON}" \
  -m QualityExperiment.run_quality_v4_wandb_ladder \
  --bundle_root "${QUAL_BUNDLE_ROOT}" \
  --train_root "${QUAL_TRAIN_ROOT}" \
  --wandb_project "${WANDB_PROJECT:-duo-quality-suite}" \
  --wandb_group "${QUAL_V4_LADDER_GROUP:-quality_v4_ladder_traces}" \
  --device cuda \
  2>&1 | tee -a "${LOG}"

echo "[done] v4 bundle=${QUAL_BUNDLE_ROOT} merged_tex=${QUAL_MERGED_OUT}"

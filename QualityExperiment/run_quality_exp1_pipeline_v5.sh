#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# English doc: Quality **v5** — v4 FS budget + simple proxy (train on PKL points; eval proxy-best decode).
# 中文注释: 复用 v3 主训 ckpt；FS 500@5e-5；训练/评估均启用 proxy 筛选；输出 quality_bundle_5 合并表。

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

export QUAL_BUNDLE_ROOT="${QUAL_BUNDLE_ROOT:-${SCRIPT_DIR}/../results/quality_bundle_5}"
export QUAL_TRAIN_ROOT="${QUAL_TRAIN_ROOT:-${SCRIPT_DIR}/../results/quality_training_3}"
export QUAL_TRAIN_CONFIG="${QUAL_TRAIN_CONFIG:-config.quality_exp5_train}"
export QUAL_REF_TRAIN_DOMAIN="${QUAL_REF_TRAIN_DOMAIN:-${QUAL_TRAIN_ROOT}/dtrain_universal_seed0/eval_train_domain}"
export QUAL_MERGED_OUT="${QUAL_MERGED_OUT:-${QUAL_BUNDLE_ROOT}/analysis_table/quality_best_by_task_merged.tex}"

export SKIP_MAIN_TRAIN="${SKIP_MAIN_TRAIN:-1}"
export FORCE_DATA="${FORCE_DATA:-0}"
export FORCE_EMBED="${FORCE_EMBED:-0}"
export FORCE_TRAIN="${FORCE_TRAIN:-0}"
# 中文注释: 不重跑扩散微调/主训；复用 v4 FS ckpt；仅补 proxy + 重评估
export FORCE_FINETUNE="${FORCE_FINETUNE:-0}"
export FORCE_PROXY="${FORCE_PROXY:-0}"
export FINETUNE_PRESERVE_VARIANTS="${FINETUNE_PRESERVE_VARIANTS:-}"
export FORCE_SUITE="${FORCE_SUITE:-1}"
export FORCE_TRAIN_DOMAIN_EVAL="${FORCE_TRAIN_DOMAIN_EVAL:-1}"
export QUAL_FS_BUNDLE_ROOT="${QUAL_FS_BUNDLE_ROOT:-${SCRIPT_DIR}/../results/quality_bundle_4}"
export SKIP_PROXY_DIFFUSION="${SKIP_PROXY_DIFFUSION:-1}"
export NO_LANDSCAPE_FIGURE="${NO_LANDSCAPE_FIGURE:-1}"

mkdir -p "${QUAL_BUNDLE_ROOT}/analysis_table" "${QUAL_BUNDLE_ROOT}/pipeline_logs"

LOG="${QUAL_BUNDLE_ROOT}/pipeline_logs/run_v5_$(date +%Y%m%d_%H%M%S).log"
echo "[info] v5 bundle=${QUAL_BUNDLE_ROOT} config=${QUAL_TRAIN_CONFIG} log=${LOG}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHON="${PYTHON:-${HOME}/anaconda3/envs/gtg/bin/python}"

"${SCRIPT_DIR}/run_quality_exp1_pipeline.sh" "$@" 2>&1 | tee -a "${LOG}"

echo "[done] v5 merged_tex=${QUAL_MERGED_OUT}"

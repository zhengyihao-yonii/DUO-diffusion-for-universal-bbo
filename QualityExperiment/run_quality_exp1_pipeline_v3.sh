#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# English doc: Quality exp1 **v3** — standard Branin coords/ranges, per-method landscape PNGs,
# scene-aware metadata + MiniLM-correlated affine A.
# 中文注释: 后缀 _3；branin_domain=standard + family_mode=scene_aware。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/wandb_login.sh
source "${DUO_ROOT}/scripts/wandb_login.sh"

# 中文注释: 单卡任务限制 CPU 线程，避免 64 核上过度抢占（可按机器改 QUAL_CPU_THREADS）
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
export QUAL_BUNDLE_ROOT="${QUAL_BUNDLE_ROOT:-${SCRIPT_DIR}/../results/quality_bundle_3}"
export QUAL_TRAIN_ROOT="${QUAL_TRAIN_ROOT:-${SCRIPT_DIR}/../results/quality_training_3}"

# v3 data generation flags (passed through to run_exp1.py)
export EXP1_BRANIN_DOMAIN="${EXP1_BRANIN_DOMAIN:-standard}"
export EXP1_FAMILY_MODE="${EXP1_FAMILY_MODE:-scene_aware}"
export EXP1_SIMILARITY_BLEND="${EXP1_SIMILARITY_BLEND:-0.65}"
export QUAL_TEST_SHIFTS="${QUAL_TEST_SHIFTS:-sim_low,sim_mid,sim_high}"
export QUAL_TRAIN_DATA_SHIFT="${QUAL_TRAIN_DATA_SHIFT:-sim_low}"
export EXP1_OBS_NOISE_BASE_STD="${EXP1_OBS_NOISE_BASE_STD:-0.03}"
export EXP1_OBS_NOISE_HIGH_STD="${EXP1_OBS_NOISE_HIGH_STD:-0.40}"
export EXP1_OBS_NOISE_HIGH_DIMS="${EXP1_OBS_NOISE_HIGH_DIMS:-1}"

exec "${SCRIPT_DIR}/run_quality_exp1_pipeline.sh" "$@"

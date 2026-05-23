#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# English doc: exp1 Quality with **one universal D_train quartet** (checkpoints under
# ``QUAL_TRAIN_ROOT/dtrain_universal_seed*``), **per-gap** ``run_exp1`` data + D_test FS finetune
# (``gap_*/fs_checkpoints``) + shift evaluation + merged LaTeX. Default **WANDB_MODE=online** for
# training loss. ``QUAL_TRAIN_DATA_GAP`` picks which ``exp1_gap*`` PKL directory pins D_train+VAE
# for main train (D_train geometry is gap-invariant in ``task_family.py``; only D_test drifts).
# 中文注释: 主训四模型与 D_train 评估共用 **通用目录** ``dtrain_universal_seed*``，不挂在某个测试 gap 目录下。
# ``QUAL_GAPS`` 中每个值只做：该 gap 的 ``run_exp1`` 数据、该 gap D_test few-shot 微调产物、shift 评估。
# ``QUAL_TRAIN_DATA_GAP``：用于主训/嵌入/D_train 评估的 ``exp1_gap*``（默认 0；若不想用 gap0 可设 0.25 等）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DUO_ROOT}"

_quality_on_err() {
  local _ec=$?
  echo "[error] pipeline aborted (exit ${_ec})" >&2
  if [[ -n "${UNIV_LOG:-}" && -f "${UNIV_LOG}/train_exp1_checkpoints.log" ]]; then
    echo "[error] last 40 lines: ${UNIV_LOG}/train_exp1_checkpoints.log" >&2
    tail -n 40 "${UNIV_LOG}/train_exp1_checkpoints.log" >&2 || true
  fi
  if [[ -n "${UNIV_LOG:-}" && -f "${UNIV_LOG}/build_task_text_embeds.log" ]]; then
    tail -n 25 "${UNIV_LOG}/build_task_text_embeds.log" >&2 || true
  fi
  if [[ -n "${UNIV_LOG:-}" && -f "${UNIV_LOG}/run_quality_suite_train_domain.log" ]]; then
    tail -n 40 "${UNIV_LOG}/run_quality_suite_train_domain.log" >&2 || true
  fi
}
trap _quality_on_err ERR

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON="${PYTHON:-${HOME}/anaconda3/envs/gtg/bin/python}"
if ! command -v "${PYTHON}" >/dev/null 2>&1 && [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_LOG_TRAIN="${WANDB_LOG_TRAIN:-1}"

# shellcheck source=scripts/wandb_login.sh
source "${DUO_ROOT}/scripts/wandb_login.sh"

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "[error] WANDB_MODE=online: set WANDB_API_KEY or config/wandb_local.sh" >&2
  exit 1
fi

export SEED="${SEED:-0}"
export QUAL_TEST_SHIFTS="${QUAL_TEST_SHIFTS:-sim_low,sim_mid,sim_high}"
export QUAL_TRAIN_DATA_SHIFT="${QUAL_TRAIN_DATA_SHIFT:-sim_low}"
# Legacy gap list (mapped to test_shifts if QUAL_TEST_SHIFTS unset)
export GAPS="${QUAL_GAPS:-}"
# 中文注释: 实验线 v2 — D_train 四维 2,3,4,5（共 4 任务）；D_test d_x=2；VAE latent_dim=4；路径后缀 _2。
export QUAL_EXP1_SUFFIX="${QUAL_EXP1_SUFFIX:-_2}"
export TRAIN_D_X="${TRAIN_D_X:-2,3,4,5}"
export TEST_D_X="${TEST_D_X:-2}"
export LATENT_DIM="${LATENT_DIM:-4}"
export D_PAD="${D_PAD:-32}"
export TRAIN_POOL_BEST_FRAC="${TRAIN_POOL_BEST_FRAC:-0.9}"
# v3: standard Branin coords + scene-aware metadata/A (run_quality_exp1_pipeline_v3.sh)
export EXP1_BRANIN_DOMAIN="${EXP1_BRANIN_DOMAIN:-legacy}"
export EXP1_FAMILY_MODE="${EXP1_FAMILY_MODE:-random_orth}"
export EXP1_SIMILARITY_BLEND="${EXP1_SIMILARITY_BLEND:-0.65}"

export TRAIN_STEP_TARGET="${TRAIN_STEP_TARGET:-0}"
export FINETUNE_ADD_STEPS="${FINETUNE_ADD_STEPS:-0}"
export FORCE_DATA="${FORCE_DATA:-0}"
export FORCE_EMBED="${FORCE_EMBED:-0}"
export FORCE_TRAIN="${FORCE_TRAIN:-0}"
export FORCE_FINETUNE="${FORCE_FINETUNE:-0}"
export FORCE_SUITE="${FORCE_SUITE:-0}"
export FORCE_TRAIN_DOMAIN_EVAL="${FORCE_TRAIN_DOMAIN_EVAL:-0}"
export SKIP_MAIN_TRAIN="${SKIP_MAIN_TRAIN:-0}"
export QUAL_TRAIN_CONFIG="${QUAL_TRAIN_CONFIG:-config.quality_exp1_train}"
export QUAL_FINETUNE_LR="${QUAL_FINETUNE_LR:-}"
export QUAL_FINETUNE_STEPS="${QUAL_FINETUNE_STEPS:-}"

export DEVICE="${DEVICE:-cuda}"
export HORIZON="${HORIZON:-64}"
export BUNDLE_ROOT="${QUAL_BUNDLE_ROOT:-${DUO_ROOT}/results/quality_bundle_2}"
export QUAL_TRAIN_ROOT="${QUAL_TRAIN_ROOT:-${DUO_ROOT}/results/quality_training_2}"
export OUT_ROOT_EXP1="${OUT_ROOT_EXP1:-results/comparison1/exp1}"
export UNISO_ROOT="${UNISO_ROOT:-${DUO_ROOT}/../UniSO}"
export WANDB_PROJECT="${WANDB_PROJECT:-duo-quality-suite}"
export NO_LANDSCAPE_FIGURE="${NO_LANDSCAPE_FIGURE:-0}"

_TEXT_SNAP="${DUO_ROOT}/../models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
if [[ -d "${_TEXT_SNAP}" ]]; then
  export TEXT_ENCODER="${TEXT_ENCODER:-$(cd "${DUO_ROOT}" && realpath "${_TEXT_SNAP}")}"
else
  export TEXT_ENCODER="${TEXT_ENCODER:-sentence-transformers/all-MiniLM-L6-v2}"
fi

NTASK="$(echo "${TRAIN_D_X}" | awk -F',' '{print NF}')"

_pkl_tag_for_shift() {
  local _sh="$1"
  QUAL_EXP1_SUFFIX="${QUAL_EXP1_SUFFIX}" "${PYTHON}" -c "
import os
sh='${_sh}'.strip()
s=os.environ.get('QUAL_EXP1_SUFFIX', '')
print(f'exp1_{sh}{s}')
"
}
_CFG_ERR="$(mktemp)"
_QUAL_PY_OUT="$(
  PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=3 \
  QUAL_TRAIN_CONFIG="${QUAL_TRAIN_CONFIG}" \
  "${PYTHON}" -c "
import importlib
import os
import sys
mod = importlib.import_module(os.environ['QUAL_TRAIN_CONFIG'])
n_tr = int(mod.N_TRAIN_STEPS)
n_ft = int(os.environ.get('QUAL_FINETUNE_STEPS') or mod.N_FINETUNE_STEPS)
ctx_tr = int(mod.CONTEXT_LENGTH_TRAIN)
ctx_fs = int(mod.CONTEXT_LENGTH_FEWSHOT)
ft_lr = float(os.environ.get('QUAL_FINETUNE_LR') or getattr(mod, 'FINETUNE_LR', mod.LR))
sys.stdout.write('{} {} {} {} {}'.format(n_tr, n_ft, ctx_tr, ctx_fs, ft_lr))
" 2>"${_CFG_ERR}"
)" || {
  echo "[error] failed to load ${QUAL_TRAIN_CONFIG}" >&2
  cat "${_CFG_ERR}" >&2 || true
  rm -f "${_CFG_ERR}"
  exit 1
}
rm -f "${_CFG_ERR}"
read -r QUAL_N_TRAIN QUAL_N_FINETUNE QUAL_CTX_TRAIN QUAL_CTX_FS QUAL_FINETUNE_LR_RESOLVED <<< "${_QUAL_PY_OUT}" || exit 1
if [[ -z "${QUAL_FINETUNE_LR}" ]]; then
  QUAL_FINETUNE_LR="${QUAL_FINETUNE_LR_RESOLVED}"
fi
if [[ -n "${QUAL_FINETUNE_STEPS}" ]]; then
  QUAL_N_FINETUNE="${QUAL_FINETUNE_STEPS}"
fi

mkdir -p "${BUNDLE_ROOT}/analysis_table"

CANON_TAG="$(_pkl_tag_for_shift "${QUAL_TRAIN_DATA_SHIFT}")"
CANON_SHIFTTAG="${CANON_TAG#exp1_}"
CANON_PKL="${DUO_ROOT}/generated_datasets/${CANON_TAG}"
UNISO_CAN="${UNISO_ROOT}/data_exp1_${CANON_SHIFTTAG}"
UNIV_DIR="${QUAL_TRAIN_ROOT}/dtrain_universal_seed${SEED}"
UNIV_CKPT="${UNIV_DIR}/checkpoints"
UNIV_TASK_EMB="${UNIV_DIR}/task_text_embeds_TxE.npy"
UNIV_LOG="${UNIV_DIR}/logs"
EVAL_TRAIN_DOMAIN="${UNIV_DIR}/eval_train_domain"
UNIV_STAMP="${UNIV_DIR}/.pipeline_stamps"
CANON_QUAL="${BUNDLE_ROOT}/shift_${CANON_SHIFTTAG}"
CANON_FS="${CANON_QUAL}/fs_checkpoints"

export TRAIN_LOG_DIR="${UNIV_LOG}"
mkdir -p "${UNIV_LOG}" "${UNIV_CKPT}" "${UNISO_CAN}" "${UNIV_STAMP}" "${CANON_QUAL}" "${CANON_FS}"

echo "[info] === Quality exp1: universal D_train train + per-shift D_test FS + eval ==="
echo "[info] TRAIN_D_X=${TRAIN_D_X} TEST_D_X=${TEST_D_X} LATENT_DIM=${LATENT_DIM} QUAL_EXP1_SUFFIX=${QUAL_EXP1_SUFFIX}"
echo "[info] QUAL_TRAIN_DATA_SHIFT=${QUAL_TRAIN_DATA_SHIFT} -> ${CANON_PKL}"
echo "[info] universal dir=${UNIV_DIR}  QUAL_TEST_SHIFTS=${QUAL_TEST_SHIFTS}"
echo "[info] WANDB_MODE=${WANDB_MODE} WANDB_PROJECT=${WANDB_PROJECT}"
echo "[info] QUAL_TRAIN_CONFIG=${QUAL_TRAIN_CONFIG} finetune_lr=${QUAL_FINETUNE_LR} finetune_steps=${QUAL_N_FINETUNE} SKIP_MAIN_TRAIN=${SKIP_MAIN_TRAIN}"

# ----- 0) run_exp1 for canonical gap (D_train PKL anchor) -----
shopt -s nullglob
_mcan=( "${CANON_PKL}"/train_merged_h*.pkl )
shopt -u nullglob
if ((${#_mcan[@]})) && [[ "${FORCE_DATA}" != "1" ]]; then
  echo "[skip] data canon ${CANON_TAG} (train_merged exists)"
else
  echo "[step] run_exp1 canon shift=${QUAL_TRAIN_DATA_SHIFT} -> ${CANON_PKL}"
  "${PYTHON}" comparisonExperiment/experiment1/run_exp1.py \
    --out_root "${OUT_ROOT_EXP1}" \
    --test_shifts "${QUAL_TRAIN_DATA_SHIFT}" \
    --uniso_data_dir "${UNISO_CAN}" \
    --train_d_x "${TRAIN_D_X}" \
    --test_d_x "${TEST_D_X}" \
    --d_pad "${D_PAD}" \
    --latent_dim "${LATENT_DIM}" \
    --horizon "${HORIZON}" \
    --seed "${SEED}" \
    --train_pool_best_frac "${TRAIN_POOL_BEST_FRAC}" \
    --pkl_suffix "${QUAL_EXP1_SUFFIX}" \
    --branin_domain "${EXP1_BRANIN_DOMAIN}" \
    --family_mode "${EXP1_FAMILY_MODE}" \
    --similarity_blend "${EXP1_SIMILARITY_BLEND}" \
    --text_encoder_model "${TEXT_ENCODER}" \
    --obs_noise_base_std "${EXP1_OBS_NOISE_BASE_STD:-0.03}" \
    --obs_noise_high_std "${EXP1_OBS_NOISE_HIGH_STD:-0.40}" \
    --obs_noise_high_dims "${EXP1_OBS_NOISE_HIGH_DIMS:-1}"
fi

# ----- 1) run_exp1 for every eval test_shift -----
IFS=',' read -r -a SHIFT_ARR <<< "${QUAL_TEST_SHIFTS}"
for shift in "${SHIFT_ARR[@]}"; do
  shift="$(echo "${shift}" | xargs)"
  [[ -z "${shift}" ]] && continue
  PKL_TAG="$(_pkl_tag_for_shift "${shift}")"
  SHIFT_TAG="${PKL_TAG#exp1_}"
  PKL_DIR="${DUO_ROOT}/generated_datasets/${PKL_TAG}"
  UNISO_GAP="${UNISO_ROOT}/data_exp1_${SHIFT_TAG}"
  QUAL_OUT="${BUNDLE_ROOT}/shift_${SHIFT_TAG}"
  STAMP="${QUAL_OUT}/.pipeline_stamps"
  mkdir -p "${UNISO_GAP}" "${QUAL_OUT}" "${STAMP}"
  shopt -s nullglob
  _mg=( "${PKL_DIR}"/train_merged_h*.pkl )
  shopt -u nullglob
  if ((${#_mg[@]})) && [[ "${FORCE_DATA}" != "1" ]]; then
    echo "[skip] data ${PKL_TAG} (train_merged exists)"
  else
    echo "[step] run_exp1 eval shift=${shift} -> ${PKL_DIR}"
    "${PYTHON}" comparisonExperiment/experiment1/run_exp1.py \
      --out_root "${OUT_ROOT_EXP1}" \
      --test_shifts "${shift}" \
      --uniso_data_dir "${UNISO_GAP}" \
      --train_d_x "${TRAIN_D_X}" \
      --test_d_x "${TEST_D_X}" \
      --d_pad "${D_PAD}" \
      --latent_dim "${LATENT_DIM}" \
      --horizon "${HORIZON}" \
      --seed "${SEED}" \
      --train_pool_best_frac "${TRAIN_POOL_BEST_FRAC}" \
      --pkl_suffix "${QUAL_EXP1_SUFFIX}" \
      --branin_domain "${EXP1_BRANIN_DOMAIN}" \
      --family_mode "${EXP1_FAMILY_MODE}" \
      --similarity_blend "${EXP1_SIMILARITY_BLEND}" \
      --text_encoder_model "${TEXT_ENCODER}" \
      --obs_noise_base_std "${EXP1_OBS_NOISE_BASE_STD:-0.03}" \
      --obs_noise_high_std "${EXP1_OBS_NOISE_HIGH_STD:-0.40}" \
      --obs_noise_high_dims "${EXP1_OBS_NOISE_HIGH_DIMS:-1}"
    touch "${STAMP}/data_ok"
  fi
done

# ----- 2) embeddings (canonical UniSO, universal path) -----
if [[ -f "${UNIV_TASK_EMB}" && "${FORCE_EMBED}" != "1" ]]; then
  echo "[skip] build_task_text_embeds (${UNIV_TASK_EMB})"
else
  [[ "${FORCE_EMBED}" == "1" ]] && : > "${UNIV_LOG}/build_task_text_embeds.log"
  echo "[step] build_task_text_embeds -> ${UNIV_TASK_EMB}"
  "${PYTHON}" -m QualityExperiment.build_task_text_embeds \
    --uniso_data_dir "${UNISO_CAN}" \
    --n_train_tasks "${NTASK}" \
    --text_encoder_model "${TEXT_ENCODER}" \
    --out_npy "${UNIV_TASK_EMB}" \
    >> "${UNIV_LOG}/build_task_text_embeds.log" 2>&1
  [[ -s "${UNIV_TASK_EMB}" ]] || { echo "[error] empty embeddings" >&2; exit 1; }
fi

# ----- 3) main train only (D_train), universal ckpt dir -----
if [[ "${FORCE_TRAIN}" == "1" ]]; then
  : > "${UNIV_LOG}/train_exp1_checkpoints_main.log"
fi
_WB=()
if [[ "${WANDB_LOG_TRAIN}" == "1" && -n "${WANDB_PROJECT}" ]]; then
  _WB+=(--wandb_project "${WANDB_PROJECT}" --wandb_group "quality_exp1_dtrain_universal_seed${SEED}")
fi
echo "[step] train_exp1_checkpoints main only -> ${UNIV_CKPT} (pkl_dir=${CANON_TAG})"
_MAIN_SKIP=()
if [[ "${SKIP_MAIN_TRAIN}" == "1" ]]; then
  _MAIN_SKIP+=(--skip_main)
fi
TRAIN_METRICS_CSV_DIR="${UNIV_DIR}/train_metrics" \
  "${PYTHON}" -m QualityExperiment.train_exp1_checkpoints \
    --pkl_dir "${CANON_PKL}" \
    --task_text_embeds_npy "${UNIV_TASK_EMB}" \
    --out_dir "${UNIV_CKPT}" \
    --horizon "${HORIZON}" \
    --context_length_train "${QUAL_CTX_TRAIN}" \
    --context_length_fewshot "${QUAL_CTX_FS}" \
    --mt_num_tasks "${NTASK}" \
    --zs_task_idx 0 \
    --device "${DEVICE}" \
    --skip_finetune \
    "${_MAIN_SKIP[@]}" \
    $( [[ "${TRAIN_STEP_TARGET}" != "0" ]] && echo --train_step_target "${TRAIN_STEP_TARGET}" ) \
    $( [[ "${FORCE_TRAIN}" == "1" ]] && echo --force_train ) \
    $( [[ "${SKIP_PROXY_DIFFUSION:-0}" == "1" ]] && echo --skip_proxy ) \
    $( [[ "${FORCE_PROXY:-0}" == "1" && "${SKIP_PROXY_DIFFUSION:-0}" != "1" ]] && echo --force_proxy ) \
    "${_WB[@]}" \
    >> "${UNIV_LOG}/train_exp1_checkpoints_main.log" 2>&1
for _ck in ckpt_st_duo.pt ckpt_st_text.pt ckpt_mt_label.pt ckpt_mt_text.pt; do
  [[ -f "${UNIV_CKPT}/${_ck}" ]] || { echo "[error] missing ${UNIV_CKPT}/${_ck}" >&2; exit 1; }
done

# ----- 4) FS finetune per test_shift (writes shift_*/fs_checkpoints) -----
_fs_one_shift() {
  local _shift="$1"
  local _PKL_TAG _SHIFT_TAG _PKL_DIR _FS_DIR _STAMP _LOG
  _PKL_TAG="$(_pkl_tag_for_shift "${_shift}")"
  _SHIFT_TAG="${_PKL_TAG#exp1_}"
  _PKL_DIR="${DUO_ROOT}/generated_datasets/${_PKL_TAG}"
  _FS_DIR="${BUNDLE_ROOT}/shift_${_SHIFT_TAG}/fs_checkpoints"
  _STAMP="${BUNDLE_ROOT}/shift_${_SHIFT_TAG}/.pipeline_stamps"
  _LOG="${UNIV_LOG}/train_exp1_fs_${_SHIFT_TAG}.log"
  mkdir -p "${_FS_DIR}" "${_STAMP}"
  if [[ "${FORCE_FINETUNE}" == "1" ]]; then
    rm -f "${_STAMP}/finetune_ok"
  fi
  if [[ -f "${_STAMP}/finetune_ok" && "${FORCE_FINETUNE}" != "1" ]]; then
    echo "[skip] fs finetune shift=${_shift} (${_FS_DIR})"
    return 0
  fi
  [[ "${FORCE_FINETUNE}" == "1" ]] && : > "${_LOG}"
  echo "[step] train_exp1_checkpoints FS shift=${_shift} -> ${_FS_DIR}"
  _WBF=()
  if [[ "${WANDB_LOG_TRAIN}" == "1" && -n "${WANDB_PROJECT}" ]]; then
    _WBF+=(--wandb_project "${WANDB_PROJECT}" --wandb_group "quality_exp1_fs_${_SHIFT_TAG}_seed${SEED}")
  fi
  TRAIN_METRICS_CSV_DIR="${UNIV_DIR}/train_metrics_fs_${_SHIFT_TAG}" \
    "${PYTHON}" -m QualityExperiment.train_exp1_checkpoints \
      --pkl_dir "${CANON_PKL}" \
      --pkl_dir_finetune "${_PKL_DIR}" \
      --task_text_embeds_npy "${UNIV_TASK_EMB}" \
      --out_dir "${UNIV_CKPT}" \
      --finetune_out_dir "${_FS_DIR}" \
      --horizon "${HORIZON}" \
      --context_length_train "${QUAL_CTX_TRAIN}" \
      --context_length_fewshot "${QUAL_CTX_FS}" \
      --mt_num_tasks "${NTASK}" \
      --zs_task_idx 0 \
      --device "${DEVICE}" \
      --skip_main \
      --finetune_steps "${QUAL_N_FINETUNE}" \
      --finetune_lr "${QUAL_FINETUNE_LR}" \
      $( [[ "${FINETUNE_ADD_STEPS}" != "0" ]] && echo --finetune_add_steps "${FINETUNE_ADD_STEPS}" ) \
      $( [[ "${FORCE_FINETUNE}" == "1" ]] && echo --force_finetune ) \
      $( [[ -n "${FINETUNE_PRESERVE_VARIANTS:-}" ]] && echo --finetune_preserve_variants "${FINETUNE_PRESERVE_VARIANTS}" ) \
      $( [[ "${SKIP_PROXY_DIFFUSION:-0}" == "1" ]] && echo --skip_proxy ) \
      $( [[ "${FORCE_PROXY:-0}" == "1" && "${SKIP_PROXY_DIFFUSION:-0}" != "1" ]] && echo --force_proxy ) \
      "${_WBF[@]}" \
      >> "${_LOG}" 2>&1
  touch "${_STAMP}/finetune_ok"
}

for shift in "${SHIFT_ARR[@]}"; do
  shift="$(echo "${shift}" | xargs)"
  [[ -z "${shift}" ]] && continue
  _fs_one_shift "${shift}"
done

# ----- 4b) proxies only (no diffusion): per-task D_train + colocated main/fs -----
_quality_proxies_main() {
  local _pfx=()
  [[ "${FORCE_PROXY:-0}" == "1" ]] && _pfx+=(--force_proxy)
  echo "[step] train_quality_proxies main + D_train per-task -> ${UNIV_CKPT}"
  mkdir -p "${UNIV_LOG}"
  "${PYTHON}" -m QualityExperiment.train_quality_proxies \
    --pkl_dir "${CANON_PKL}" \
    --ckpt_dir "${UNIV_CKPT}" \
    --task_text_embeds_npy "${UNIV_TASK_EMB}" \
    --horizon "${HORIZON}" \
    --context_length_train "${QUAL_CTX_TRAIN}" \
    --context_length_fewshot "${QUAL_CTX_FS}" \
    --mt_num_tasks "${NTASK}" \
    --device "${DEVICE}" \
    --train_domain \
    --colocated_main \
    "${_pfx[@]}" \
    >> "${UNIV_LOG}/train_quality_proxies.log" 2>&1
}

_quality_proxies_fs_shift() {
  local _shift="$1"
  local _PKL_TAG _SHIFT_TAG _PKL_DIR _FS_DIR _pfx=()
  _PKL_TAG="$(_pkl_tag_for_shift "${_shift}")"
  _SHIFT_TAG="${_PKL_TAG#exp1_}"
  _PKL_DIR="${DUO_ROOT}/generated_datasets/${_PKL_TAG}"
  _FS_DIR="${BUNDLE_ROOT}/shift_${_SHIFT_TAG}/fs_checkpoints"
  if [[ -n "${QUAL_FS_BUNDLE_ROOT:-}" ]]; then
    local _ALT="${QUAL_FS_BUNDLE_ROOT}/shift_${_SHIFT_TAG}/fs_checkpoints"
    if [[ -d "${_ALT}" ]]; then
      _FS_DIR="${_ALT}"
    fi
  fi
  [[ -d "${_FS_DIR}" ]] || { echo "[skip] fs proxies shift=${_shift}: no ${_FS_DIR}"; return 0; }
  [[ "${FORCE_PROXY:-0}" == "1" ]] && _pfx+=(--force_proxy)
  echo "[step] train_quality_proxies fs shift=${_shift} -> ${_FS_DIR}"
  "${PYTHON}" -m QualityExperiment.train_quality_proxies \
    --pkl_dir "${CANON_PKL}" \
    --pkl_dir_finetune "${_PKL_DIR}" \
    --ckpt_dir "${UNIV_CKPT}" \
    --fs_ckpt_dir "${_FS_DIR}" \
    --task_text_embeds_npy "${UNIV_TASK_EMB}" \
    --horizon "${HORIZON}" \
    --context_length_train "${QUAL_CTX_TRAIN}" \
    --context_length_fewshot "${QUAL_CTX_FS}" \
    --mt_num_tasks "${NTASK}" \
    --device "${DEVICE}" \
    --colocated_fs \
    "${_pfx[@]}" \
    >> "${UNIV_LOG}/train_quality_proxies_fs_${_SHIFT_TAG}.log" 2>&1
}

if [[ "${SKIP_PROXY_DIFFUSION:-0}" == "1" ]]; then
  _quality_proxies_main
  for shift in "${SHIFT_ARR[@]}"; do
    shift="$(echo "${shift}" | xargs)"
    [[ -z "${shift}" ]] && continue
    _quality_proxies_fs_shift "${shift}"
  done
fi

# ----- 5) D_train domain eval once (uses canonical gap FS + UniSO) -----
if [[ -f "${UNIV_DIR}/.train_domain_eval_ok" && "${FORCE_SUITE}" != "1" && "${FORCE_TRAIN_DOMAIN_EVAL}" != "1" ]]; then
  echo "[skip] train_domain (${EVAL_TRAIN_DOMAIN})"
else
  [[ "${FORCE_SUITE}" == "1" || "${FORCE_TRAIN_DOMAIN_EVAL}" == "1" ]] && : > "${UNIV_LOG}/run_quality_suite_train_domain.log"
  echo "[step] run_quality_suite train_domain -> ${EVAL_TRAIN_DOMAIN}"
  mkdir -p "${EVAL_TRAIN_DOMAIN}"
  _LS_TD=()
  [[ "${NO_LANDSCAPE_FIGURE}" == "1" ]] && _LS_TD+=(--no_landscape_figure)
  "${PYTHON}" -m QualityExperiment.run_quality_suite \
    --uniso_data_dir "${UNISO_CAN}" \
    --pkl_dir "${CANON_PKL}" \
    --horizon "${HORIZON}" \
    --context_length_train "${QUAL_CTX_TRAIN}" \
    --context_length_fewshot "${QUAL_CTX_FS}" \
    --phases train_domain \
    --task_text_embeds_npy "${UNIV_TASK_EMB}" \
    --mt_num_tasks "${NTASK}" \
    --text_encoder_model "${TEXT_ENCODER}" \
    --local_out_dir "${EVAL_TRAIN_DOMAIN}" \
    --ckpt_st_duo "${UNIV_CKPT}/ckpt_st_duo.pt" \
    --ckpt_st_text "${UNIV_CKPT}/ckpt_st_text.pt" \
    --ckpt_mt_label "${UNIV_CKPT}/ckpt_mt_label.pt" \
    --ckpt_mt_text "${UNIV_CKPT}/ckpt_mt_text.pt" \
    --ckpt_st_duo_fs "${CANON_FS}/ckpt_st_duo_fs.pt" \
    --ckpt_st_text_fs "${CANON_FS}/ckpt_st_text_fs.pt" \
    --ckpt_mt_label_fs "${CANON_FS}/ckpt_mt_label_fs.pt" \
    --ckpt_mt_text_fs "${CANON_FS}/ckpt_mt_text_fs.pt" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_group "quality_exp1_train_domain_seed${SEED}" \
    --device "${DEVICE}" \
    "${_LS_TD[@]}" \
    >> "${UNIV_LOG}/run_quality_suite_train_domain.log" 2>&1
  compgen -G "${EVAL_TRAIN_DOMAIN}/quality_trace_*.npz" >/dev/null || { echo "[error] no train_domain npz" >&2; exit 1; }
  touch "${UNIV_DIR}/.train_domain_eval_ok"
fi

REF_TRAIN_DOMAIN_ART="${QUAL_REF_TRAIN_DOMAIN:-${EVAL_TRAIN_DOMAIN}}"
FIRST_UNISO="${UNISO_CAN}"

# ----- 6) shift eval per test_shift -----
MERGE_ARTS=()
for shift in "${SHIFT_ARR[@]}"; do
  shift="$(echo "${shift}" | xargs)"
  [[ -z "${shift}" ]] && continue
  PKL_TAG="$(_pkl_tag_for_shift "${shift}")"
  SHIFT_TAG="${PKL_TAG#exp1_}"
  PKL_DIR="${DUO_ROOT}/generated_datasets/${PKL_TAG}"
  UNISO_GAP="${UNISO_ROOT}/data_exp1_${SHIFT_TAG}"
  QUAL_OUT="${BUNDLE_ROOT}/shift_${SHIFT_TAG}"
  STAMP="${QUAL_OUT}/.pipeline_stamps"
  ART="${QUAL_OUT}/artifacts_${PKL_TAG}_seed${SEED}"
  FS_DIR="${QUAL_OUT}/fs_checkpoints"
  if [[ -n "${QUAL_FS_BUNDLE_ROOT:-}" ]]; then
    _FS_ALT="${QUAL_FS_BUNDLE_ROOT}/shift_${SHIFT_TAG}/fs_checkpoints"
    if [[ -d "${_FS_ALT}" ]]; then
      FS_DIR="${_FS_ALT}"
      echo "[info] shift=${shift} reuse fs ckpts from ${FS_DIR}"
    fi
  fi
  mkdir -p "${QUAL_OUT}" "${STAMP}" "${ART}"

  if [[ -f "${STAMP}/suite_ok" && "${FORCE_SUITE}" != "1" ]]; then
    echo "[skip] shift suite (${ART})"
  else
    [[ "${FORCE_SUITE}" == "1" ]] && { rm -f "${STAMP}/suite_ok"; : > "${QUAL_OUT}/run_quality_suite_shift.log"; }
    echo "[step] run_quality_suite shift=${shift} -> ${ART}"
    _LS1=()
    [[ "${NO_LANDSCAPE_FIGURE}" == "1" ]] && _LS1+=(--no_landscape_figure)
    "${PYTHON}" -m QualityExperiment.run_quality_suite \
      --uniso_data_dir "${UNISO_GAP}" \
      --pkl_dir "${PKL_DIR}" \
      --horizon "${HORIZON}" \
      --context_length_train "${QUAL_CTX_TRAIN}" \
      --context_length_fewshot "${QUAL_CTX_FS}" \
      --phases shift_zero_shot,shift_few_shot_tail10p,shift_few_shot_tail20p,shift_few_shot_tail50p \
      --task_text_embeds_npy "${UNIV_TASK_EMB}" \
      --mt_num_tasks "${NTASK}" \
      --text_encoder_model "${TEXT_ENCODER}" \
      --local_out_dir "${ART}" \
      --ckpt_st_duo "${UNIV_CKPT}/ckpt_st_duo.pt" \
      --ckpt_st_text "${UNIV_CKPT}/ckpt_st_text.pt" \
      --ckpt_mt_label "${UNIV_CKPT}/ckpt_mt_label.pt" \
      --ckpt_mt_text "${UNIV_CKPT}/ckpt_mt_text.pt" \
      --ckpt_st_duo_fs "${FS_DIR}/ckpt_st_duo_fs.pt" \
      --ckpt_st_text_fs "${FS_DIR}/ckpt_st_text_fs.pt" \
      --ckpt_mt_label_fs "${FS_DIR}/ckpt_mt_label_fs.pt" \
      --ckpt_mt_text_fs "${FS_DIR}/ckpt_mt_text_fs.pt" \
      --wandb_project "${WANDB_PROJECT}" \
      --wandb_group "quality_exp1_${SHIFT_TAG}" \
      --device "${DEVICE}" \
      "${_LS1[@]}" \
      >> "${QUAL_OUT}/run_quality_suite_shift.log" 2>&1
    compgen -G "${ART}/quality_trace_*.npz" >/dev/null || { echo "[error] no shift npz ${ART}" >&2; exit 1; }
    touch "${STAMP}/suite_ok"
  fi
  MERGE_ARTS+=("${ART}")
done

OUT_MERGED="${QUAL_MERGED_OUT:-${BUNDLE_ROOT}/analysis_table/quality_best_by_task_merged.tex}"
echo "[step] export merged LaTeX -> ${OUT_MERGED}"
"${PYTHON}" -m QualityExperiment.export_quality_latex_table \
  --merge_artifacts_dirs "${MERGE_ARTS[@]}" \
  --reference_train_artifacts_dir "${REF_TRAIN_DOMAIN_ART}" \
  --reference_uniso_data_dir "${FIRST_UNISO}" \
  --out_tex "${OUT_MERGED}"

trap - ERR
echo "[done] universal_train=${UNIV_DIR} ref_uniso=${FIRST_UNISO} merged_tex=${OUT_MERGED}"

#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# English doc: Restore v3.5 step-500 mt_text fs ckpts and re-run shift eval + merged LaTeX only.
# 中文注释: 修复 v4 被 force_finetune 覆盖的 mt_text；不重训其它模型。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/wandb_login.sh
source "${DUO_ROOT}/scripts/wandb_login.sh"

export PYTHON="${PYTHON:-${HOME}/anaconda3/envs/gtg/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QUAL_BUNDLE_ROOT="${QUAL_BUNDLE_ROOT:-${DUO_ROOT}/results/quality_bundle_4}"
export QUAL_TRAIN_ROOT="${QUAL_TRAIN_ROOT:-${DUO_ROOT}/results/quality_training_3}"
export SEED="${SEED:-0}"
export HORIZON="${HORIZON:-64}"
export NTASK="${NTASK:-5}"
export TEXT_ENCODER="${TEXT_ENCODER:-sentence-transformers/all-MiniLM-L6-v2}"
export DEVICE="${DEVICE:-cuda}"
export WANDB_PROJECT="${WANDB_PROJECT:-duo-quality-suite}"

UNIV="${QUAL_TRAIN_ROOT}/dtrain_universal_seed${SEED}"
UNIV_CKPT="${UNIV}/checkpoints"
TASK_EMB="${UNIV}/task_text_embeds_TxE.npy"
REF_TRAIN="${UNIV}/eval_train_domain"
MERGED="${QUAL_BUNDLE_ROOT}/analysis_table/quality_best_by_task_merged.tex"

echo "[step] restore mt_text fs ckpts from v3.5 step_500"
bash "${SCRIPT_DIR}/seed_v4_fs_mt_text_from_v35.sh"

echo "[step] copy v3.5 step-500 mt_text eval npz (exact match to v3.5 sweep table)"
bash "${SCRIPT_DIR}/copy_v35_mt_text_eval_to_v4.sh"

IFS=',' read -r -a SHIFTS <<< "${QUAL_TEST_SHIFTS:-sim_low,sim_mid,sim_high}"
MERGE_ARTS=()

if [[ "${REFRESH_MT_TEXT_REEVAL:-0}" == "1" ]]; then
  echo "[info] REFRESH_MT_TEXT_REEVAL=1: re-run suite for all shifts (other methods only if npz kept)"
else
  echo "[info] skip suite re-eval; mt_text npz copied from v3.5"
  for shift in "${SHIFTS[@]}"; do
    shift="$(echo "${shift}" | xargs)"
    [[ -z "${shift}" ]] && continue
    MERGE_ARTS+=("${QUAL_BUNDLE_ROOT}/shift_${shift}_3/artifacts_exp1_${shift}_3_seed${SEED}")
  done
  echo "[step] export merged LaTeX -> ${MERGED}"
  mkdir -p "$(dirname "${MERGED}")"
  "${PYTHON}" -m QualityExperiment.export_quality_latex_table \
    --merge_artifacts_dirs "${MERGE_ARTS[@]}" \
    --reference_train_artifacts_dir "${REF_TRAIN}" \
    --reference_uniso_data_dir "${DUO_ROOT}/../UniSO/data_exp1_sim_low_3" \
    --out_tex "${MERGED}"
  echo "[done] ${MERGED}"
  exit 0
fi

for shift in "${SHIFTS[@]}"; do
  shift="$(echo "${shift}" | xargs)"
  [[ -z "${shift}" ]] && continue
  TAG="exp1_${shift}_3"
  SHIFT_TAG="${TAG#exp1_}"
  PKL_DIR="${DUO_ROOT}/generated_datasets/${TAG}"
  UNISO="${DUO_ROOT}/../UniSO/data_exp1_${SHIFT_TAG}"
  QUAL_OUT="${QUAL_BUNDLE_ROOT}/shift_${SHIFT_TAG}"
  ART="${QUAL_OUT}/artifacts_${TAG}_seed${SEED}"
  FS_DIR="${QUAL_OUT}/fs_checkpoints"
  mkdir -p "${ART}" "${QUAL_OUT}"
  rm -f "${QUAL_OUT}/.pipeline_stamps/suite_ok"
  : > "${QUAL_OUT}/run_quality_suite_shift_refresh.log"

  echo "[step] re-eval shift=${shift} -> ${ART}"
  "${PYTHON}" -m QualityExperiment.run_quality_suite \
    --uniso_data_dir "${UNISO}" \
    --pkl_dir "${PKL_DIR}" \
    --horizon "${HORIZON}" \
    --context_length_train 32 \
    --context_length_fewshot 16 \
    --phases shift_zero_shot,shift_few_shot_tail10p,shift_few_shot_tail20p,shift_few_shot_tail50p \
    --task_text_embeds_npy "${TASK_EMB}" \
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
    --wandb_group "quality_v4_refresh_${SHIFT_TAG}" \
    --device "${DEVICE}" \
    --no_landscape_figure \
    >> "${QUAL_OUT}/run_quality_suite_shift_refresh.log" 2>&1

  touch "${QUAL_OUT}/.pipeline_stamps/suite_ok"
  MERGE_ARTS+=("${ART}")
done

echo "[step] export merged LaTeX -> ${MERGED}"
mkdir -p "$(dirname "${MERGED}")"
"${PYTHON}" -m QualityExperiment.export_quality_latex_table \
  --merge_artifacts_dirs "${MERGE_ARTS[@]}" \
  --reference_train_artifacts_dir "${REF_TRAIN}" \
  --reference_uniso_data_dir "${DUO_ROOT}/../UniSO/data_exp1_sim_low_3" \
  --out_tex "${MERGED}"

echo "[done] ${MERGED}"

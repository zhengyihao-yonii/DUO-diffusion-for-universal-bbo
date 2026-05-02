#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# English doc: End-to-end QualityExperiment on exp1. Optimizer steps = rounds * steps_per_round
# (default: 100 rounds * 100 = 10000 main steps; 50 * 100 = 5000 finetune steps).
# 中文注释: 「轮」由 TRAIN_ROUNDS/FINETUNE_ROUNDS 指定；每轮对应梯度步数 = STEPS_PER_ROUND（默认 100）。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DUO_ROOT}"

PYTHON="${PYTHON:-python3}"

# ----- user-tunable -----
export SEED="${SEED:-0}"
# D_train 个数 = TRAIN_D_X 逗号分隔项数（与 run_exp1 --train_d_x 一致）
export TRAIN_D_X="${TRAIN_D_X:-10,12,14,18,20}"
export TEST_D_X="${TEST_D_X:-16}"
export LATENT_DIM="${LATENT_DIM:-16}"
# DUO 通用 zero-pad 维（VAE 输入维）；须 >= 所有原生 d_x
export D_PAD="${D_PAD:-32}"
# 与 run_exp1 目录命名一致（0.5 -> exp1_gap0p500）
export GAP_LIST="${GAP_LIST:-0,0.5}"
export PKL_GAP_DIR_NAME="${PKL_GAP_DIR_NAME:-exp1_gap0p500}"
# 轮数；实际 Trainer 步数 = 轮数 * STEPS_PER_ROUND
export TRAIN_ROUNDS="${TRAIN_ROUNDS:-100}"
export FINETUNE_ROUNDS="${FINETUNE_ROUNDS:-50}"
export STEPS_PER_ROUND="${STEPS_PER_ROUND:-100}"
TRAIN_STEPS=$((TRAIN_ROUNDS * STEPS_PER_ROUND))
FINETUNE_STEPS=$((FINETUNE_ROUNDS * STEPS_PER_ROUND))

export UNISO_DATA_DIR="${UNISO_DATA_DIR:-${DUO_ROOT}/../UniSO/data}"
export OUT_ROOT_EXP1="${OUT_ROOT_EXP1:-results/comparison1/exp1}"
export QUAL_OUT="${QUAL_OUT:-results/quality_full_run}"
export TEXT_ENCODER="${TEXT_ENCODER:-sentence-transformers/all-MiniLM-L6-v2}"
export WANDB_PROJECT="${WANDB_PROJECT:-duo-quality-suite}"
export WANDB_GROUP="${WANDB_GROUP:-exp1_quality_full}"
export DEVICE="${DEVICE:-cuda}"
# 1=跳过数据生成（已跑过 run_exp1）
export SKIP_DATA="${SKIP_DATA:-0}"
# 1=跳过 quartet 训练（已有 ckpt）
export SKIP_TRAIN="${SKIP_TRAIN:-0}"
# 1=跳过 run_quality_suite
export SKIP_SUITE="${SKIP_SUITE:-0}"

mkdir -p "${QUAL_OUT}"

NTASK="$(echo "${TRAIN_D_X}" | awk -F',' '{print NF}')"

echo "[info] DUO_ROOT=${DUO_ROOT}"
echo "[info] TRAIN_D_X=${TRAIN_D_X} (NTASK=${NTASK}) TEST_D_X=${TEST_D_X} D_PAD=${D_PAD} LATENT_DIM=${LATENT_DIM} PKL_GAP_DIR_NAME=${PKL_GAP_DIR_NAME}"
echo "[info] train: ${TRAIN_ROUNDS} rounds x ${STEPS_PER_ROUND} steps/round = ${TRAIN_STEPS} steps"
echo "[info] finetune: ${FINETUNE_ROUNDS} rounds x ${STEPS_PER_ROUND} steps/round = ${FINETUNE_STEPS} steps"

if [[ "${SKIP_DATA}" != "1" ]]; then
  echo "[step] run_exp1.py (gaps=${GAP_LIST}, seed=${SEED}, train_d_x=${TRAIN_D_X})"
  "${PYTHON}" comparisonExperiment/experiment1/run_exp1.py \
    --out_root "${OUT_ROOT_EXP1}" \
    --gaps "${GAP_LIST}" \
    --uniso_data_dir "${UNISO_DATA_DIR}" \
    --train_d_x "${TRAIN_D_X}" \
    --test_d_x "${TEST_D_X}" \
    --d_pad "${D_PAD}" \
    --latent_dim "${LATENT_DIM}" \
    --seed "${SEED}"
else
  echo "[skip] data generation (SKIP_DATA=1)"
fi

PKL_DIR="${DUO_ROOT}/generated_datasets/${PKL_GAP_DIR_NAME}"
TASK_EMB="${QUAL_OUT}/task_text_embeds_TxE.npy"

echo "[step] build_task_text_embeds -> ${TASK_EMB}"
"${PYTHON}" -m QualityExperiment.build_task_text_embeds \
  --uniso_data_dir "${UNISO_DATA_DIR}" \
  --n_train_tasks "${NTASK}" \
  --text_encoder_model "${TEXT_ENCODER}" \
  --out_npy "${TASK_EMB}"

CKPT_DIR="${QUAL_OUT}/checkpoints"
if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo "[step] train quartet + FS (${TRAIN_STEPS} / ${FINETUNE_STEPS} optimizer steps)"
  "${PYTHON}" -m QualityExperiment.train_exp1_checkpoints \
    --pkl_dir "${PKL_DIR}" \
    --task_text_embeds_npy "${TASK_EMB}" \
    --out_dir "${CKPT_DIR}" \
    --horizon 32 \
    --mt_num_tasks "${NTASK}" \
    --train_steps "${TRAIN_STEPS}" \
    --finetune_steps "${FINETUNE_STEPS}" \
    --zs_task_idx 0 \
    --batch_size 32 \
    --lr 2e-4 \
    --grad_accum 2 \
    --device "${DEVICE}"
else
  echo "[skip] training (SKIP_TRAIN=1)"
fi

LOCAL_ART="${QUAL_OUT}/artifacts_${PKL_GAP_DIR_NAME}_seed${SEED}"

if [[ "${SKIP_SUITE}" != "1" ]]; then
  echo "[step] run_quality_suite"
  "${PYTHON}" -m QualityExperiment.run_quality_suite \
    --uniso_data_dir "${UNISO_DATA_DIR}" \
    --pkl_dir "${PKL_DIR}" \
    --phases train_domain,shift_zero_shot,shift_few_shot \
    --task_text_embeds_npy "${TASK_EMB}" \
    --mt_num_tasks "${NTASK}" \
    --text_encoder_model "${TEXT_ENCODER}" \
    --local_out_dir "${LOCAL_ART}" \
    --ckpt_st_duo "${CKPT_DIR}/ckpt_st_duo.pt" \
    --ckpt_st_text "${CKPT_DIR}/ckpt_st_text.pt" \
    --ckpt_mt_label "${CKPT_DIR}/ckpt_mt_label.pt" \
    --ckpt_mt_text "${CKPT_DIR}/ckpt_mt_text.pt" \
    --ckpt_st_duo_fs "${CKPT_DIR}/ckpt_st_duo_fs.pt" \
    --ckpt_st_text_fs "${CKPT_DIR}/ckpt_st_text_fs.pt" \
    --ckpt_mt_label_fs "${CKPT_DIR}/ckpt_mt_label_fs.pt" \
    --ckpt_mt_text_fs "${CKPT_DIR}/ckpt_mt_text_fs.pt" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_group "${WANDB_GROUP}" \
    --device "${DEVICE}"
else
  echo "[skip] run_quality_suite (SKIP_SUITE=1)"
fi

echo "[done] checkpoints: ${CKPT_DIR}"
echo "[done] wandb project: ${WANDB_PROJECT} group: ${WANDB_GROUP}"
echo "[done] local artifacts: ${LOCAL_ART}"

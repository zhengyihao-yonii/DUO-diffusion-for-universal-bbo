#!/usr/bin/env bash
# Full-multitask smoke: run evaluate.py with random diffusion weights (no checkpoint) for seeds 0..7.
# Verifies the inference stack (sampling, multitask conditioning, oracle scoring) without trained weights.
#
# Defaults align with typical full-MT experiments (same 9-task CSV as run_multitask.sh _FULL_MT_TASKS_CSV;
# trajectory overrides from examples/traj_params_per_task_example2.json when present).
#
# Usage:
#   bash run_eval_random_diffusion_inference_smoke.sh
#   bash run_eval_random_diffusion_inference_smoke.sh --latent_dim 64
#   SMOKE_SLUG=my_try1 bash run_eval_random_diffusion_inference_smoke.sh
#
# Env:
#   PYTHON            default python3
#   TRAIN_TASKS       default 9-task full MT (comma CSV, canonical order)
#   TRAJ_PARAMS_JSON  default ${_ROOT}/examples/traj_params_per_task_example2.json if file exists, else empty
#   N_TRAJ K EPS      default 1000 50 0.05
#   HORIZON CTX_LEN   default 64 32
#   FRAC SIGMA        default 1.0 0.0
#   LATENT_DIM        default 32
#   USE_TEXT_CONDITION / MULTITASK_TEXT_ONLY  default 1 / 1（与 text-only 全任务实验一致；设 0 关闭）
#   TRAIN_TIMESTEP_BIAS_POWER  default 0.5
#   LEARNING_RATE     default 0.0002
#   START_SEED        default 0
#   NUM_SEEDS         default 8  -> runs START_SEED .. START_SEED+NUM_SEEDS-1
#   SMOKE_ROOT        default results/smoke/${SMOKE_SLUG}; SMOKE_SLUG default random_diffusion_full_mt
#   WANDB_DISABLED    default 1
#
set -euo pipefail

_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${_ROOT}"
export PYTHONPATH="${_ROOT}:${PYTHONPATH:-}"
# shellcheck source=scripts/wandb_login.sh
source "${_ROOT}/scripts/wandb_login.sh"

PYTHON="${PYTHON:-python3}"
export WANDB_DISABLED="${WANDB_DISABLED:-1}"

# 与 run_multitask.sh 中全任务 9 任务字典序一致
_TRAIN_TASKS_DEFAULT="ant,dkitty,gtopx2,gtopx3,gtopx4,gtopx6,superconductor,tfbind10,tfbind8"
TRAIN_TASKS="${TRAIN_TASKS:-${_TRAIN_TASKS_DEFAULT}}"

N_TRAJ="${N_TRAJ:-1000}"
K="${K:-50}"
EPS="${EPS:-0.05}"
HORIZON="${HORIZON:-64}"
CTX_LEN="${CTX_LEN:-32}"
FRAC="${FRAC:-1.0}"
SIGMA="${SIGMA:-0.0}"
LATENT_DIM="${LATENT_DIM:-32}"

# 与常见全任务 text-only 训练/评估默认一致（可按需覆盖为 0）
USE_TEXT_CONDITION="${USE_TEXT_CONDITION:-1}"
MULTITASK_TEXT_ONLY="${MULTITASK_TEXT_ONLY:-1}"
TRAIN_TIMESTEP_BIAS_POWER="${TRAIN_TIMESTEP_BIAS_POWER:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"

START_SEED="${START_SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-8}"

SMOKE_SLUG="${SMOKE_SLUG:-random_diffusion_full_mt}"
SMOKE_ROOT="${SMOKE_ROOT:-${_ROOT}/results/smoke/${SMOKE_SLUG}}"

_TRAJ_DEFAULT="${_ROOT}/examples/traj_params_per_task_example2.json"
if [[ -z "${TRAJ_PARAMS_JSON:-}" ]]; then
  if [[ -f "${_TRAJ_DEFAULT}" ]]; then
    TRAJ_PARAMS_JSON="${_TRAJ_DEFAULT}"
  else
    TRAJ_PARAMS_JSON=""
  fi
fi

mkdir -p "${SMOKE_ROOT}"

_meta="${SMOKE_ROOT}/meta.env"
{
  echo "SMOKE_SLUG=${SMOKE_SLUG}"
  echo "TRAIN_TASKS=${TRAIN_TASKS}"
  echo "START_SEED=${START_SEED}"
  echo "NUM_SEEDS=${NUM_SEEDS}"
  echo "N_TRAJ=${N_TRAJ} K=${K} EPS=${EPS}"
  echo "HORIZON=${HORIZON} CTX_LEN=${CTX_LEN}"
  echo "FRAC=${FRAC} SIGMA=${SIGMA} LATENT_DIM=${LATENT_DIM}"
  echo "USE_TEXT_CONDITION=${USE_TEXT_CONDITION} MULTITASK_TEXT_ONLY=${MULTITASK_TEXT_ONLY}"
  echo "TRAIN_TIMESTEP_BIAS_POWER=${TRAIN_TIMESTEP_BIAS_POWER} LEARNING_RATE=${LEARNING_RATE}"
  echo "TRAJ_PARAMS_JSON=${TRAJ_PARAMS_JSON}"
  echo "WANDB_DISABLED=${WANDB_DISABLED}"
  echo "PYTHON=${PYTHON}"
  echo "EXTRA_ARGS=$*"
} >"${_meta}"

_manifest="${SMOKE_ROOT}/manifest.log"
echo "[smoke] root=${SMOKE_ROOT}" | tee -a "${_manifest}"
echo "[smoke] wrote ${_meta}" | tee -a "${_manifest}"

_base_args=(
  --train_tasks "${TRAIN_TASKS}"
  --n_traj "${N_TRAJ}"
  --k "${K}"
  --eps "${EPS}"
  --horizon "${HORIZON}"
  --ctx_len "${CTX_LEN}"
  --frac "${FRAC}"
  --sigma "${SIGMA}"
  --latent_dim "${LATENT_DIM}"
  --train_timestep_bias_power "${TRAIN_TIMESTEP_BIAS_POWER}"
  --learning_rate "${LEARNING_RATE}"
  --random_diffusion_weights
  --proxy_filter 0
)
if [[ "${USE_TEXT_CONDITION}" == "1" ]]; then
  _base_args+=(--use_text_condition)
fi
if [[ "${MULTITASK_TEXT_ONLY}" == "1" ]]; then
  _base_args+=(--multitask_text_only)
fi
if [[ -n "${TRAJ_PARAMS_JSON}" ]]; then
  _base_args+=(--traj_params_json "${TRAJ_PARAMS_JSON}")
fi

for ((i = 0; i < NUM_SEEDS; i++)); do
  _s=$((START_SEED + i))
  _out="${SMOKE_ROOT}/seed_${_s}"
  mkdir -p "${_out}"
  _log="${_out}/evaluate.log"
  echo "[smoke] seed=${_s} log=${_log}" | tee -a "${_manifest}"
  # 用户传入的额外参数在前；--seed 放最后保证本脚本控制的 seed 生效
  "${PYTHON}" "${_ROOT}/evaluate.py" \
    "${_base_args[@]}" \
    "$@" \
    --seed "${_s}" \
    >"${_log}" 2>&1
done

echo "[smoke] all ${NUM_SEEDS} seeds OK under ${SMOKE_ROOT}" | tee -a "${_manifest}"

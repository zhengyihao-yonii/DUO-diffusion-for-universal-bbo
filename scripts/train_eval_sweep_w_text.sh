#!/usr/bin/env bash
# Multitask + text-conditioned training per seed, then sweep condition_guidance_w_text (text CFG) per seed.
#
# Seeds (priority order):
#   1) SEEDS="0 1 2" or SEEDS="0,1,2"
#   2) NUM_SEEDS (or NUM_RUNS) + START_SEED (range)
#   3) SEED (single) or START_SEED (default 0)
#
# Logs: results/eval_sweep_w_text/<SWEEP_ARCHIVE_SLUG>/seed<SEED>/
# SWEEP_ARCHIVE_SLUG is aligned with train.py/evaluate.py hyper dir when possible.
#
# LATENT_DIM defaults to 32; can be overridden via env or CLI --latent_dim.
# TRAIN_TIMESTEP_BIAS_POWER / TRAIN_LOSS_MIN_SNR_GAMMA default to 0.0 (disabled).
# TRAIN_LEARNING_RATE optional; when set, passes --learning_rate and appends _lr... to paths.
# PROXY_FILTER optional 0/1; when set, passes --proxy_filter.
# DIFFUSION_CKPT_TRAIN_STEPS optional int: passed to evaluate.py as --diffusion_ckpt_train_steps
#   (load state_{N}.pt; default eval loads latest state_*.pt in checkpoint dir).
# train.py 默认 resume：同 MODEL_ROOT 下若 checkpoint/ 已有 state_*.pt 且步数 < TRAIN_EPOCHS 对应总步数，
#   会从最新 state 续训；若要同路径从零重训：TRAIN_EXTRA 前加 --retrain 或 export 后手写 python train.py ... --retrain
# Wandb 续同一条曲线：第一次训练记下 run id，续跑前 export WANDB_RUN_ID=<id> WANDB_RESUME=allow
#   （日志里 metrics 的 step 与 trainer.step 一致时会接在原有 step 轴后面；见 https://docs.wandb.ai/guides/runs/resuming ）
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Avoid nounset surprises: treat as optional empty string.
DIFFUSION_CKPT_TRAIN_STEPS="${DIFFUSION_CKPT_TRAIN_STEPS:-}"

# ---------- CLI parsing (before LATENT_DIM default) ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --latent_dim=*)
      LATENT_DIM="${1#*=}"
      shift
      ;;
    --latent_dim)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: --latent_dim requires an integer argument" >&2
        exit 1
      fi
      LATENT_DIM="$2"
      shift 2
      ;;
    -h | --help)
      sed -n '1,28p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1 (supported: --latent_dim <int>, --latent_dim=<int>, -h)" >&2
      exit 1
      ;;
  esac
done

canonical_train_tasks_token() {
  local csv="$1"
  local -a parts=()
  local IFS=,
  read -ra parts <<< "${csv// /}"
  local -a nonempty=()
  local p
  for p in "${parts[@]}"; do
    [[ -z "${p// /}" ]] && continue
    nonempty+=("${p// /}")
  done
  if [[ ${#nonempty[@]} -le 1 ]]; then
    echo "${nonempty[0]:-}"
    return
  fi
  local sorted
  sorted="$(printf '%s\n' "${nonempty[@]}" | LC_ALL=C sort)"
  echo "${sorted//$'\n'/_}"
}

# ========= Config =========
TRAIN_TASKS="${TRAIN_TASKS:-ant,dkitty,gtopx2,gtopx3,gtopx4,gtopx6,superconductor,tfbind10,tfbind8}"
FRAC="${FRAC:-1.0}"
SIGMA="${SIGMA:-0.0}"
N_TRAJ="${N_TRAJ:-1000}"
K="${K:-50}"
EPS="${EPS:-0.05}"
HORIZON="${HORIZON:-64}"
# Seed selection (see header docs).
SEED="${SEED:-}"
START_SEED="${START_SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-${NUM_RUNS:-1}}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-500}"
TEXT_ENCODER="${TEXT_ENCODER:-sentence-transformers/all-MiniLM-L6-v2}"

TRAJ_PARAMS_JSON="${TRAJ_PARAMS_JSON:-}"
LATENT_DIM="${LATENT_DIM:-32}"
TRAIN_TIMESTEP_BIAS_POWER="${TRAIN_TIMESTEP_BIAS_POWER:-0.0}"
TRAIN_LOSS_MIN_SNR_GAMMA="${TRAIN_LOSS_MIN_SNR_GAMMA:-0.0}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-}"
TRAIN_HALF_TBIAS_FRAC="${TRAIN_HALF_TBIAS_FRAC:-0.7}"
TRAIN_HALF_LR_MULT="${TRAIN_HALF_LR_MULT:-1.0}"

if [[ -n "${W_VALUES_OVERRIDE:-}" ]]; then
  read -ra W_VALUES <<< "${W_VALUES_OVERRIDE}"
else
  W_VALUES=(0.0 0.8 1.2 2.0)
fi

EVAL_SWEEP_DIR="${EVAL_SWEEP_DIR:-results/eval_sweep_w_text}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
LOG_TO_TERMINAL="${LOG_TO_TERMINAL:-0}"

PROXY_FILTER_EXTRA=()
if [[ -n "${PROXY_FILTER:-}" ]]; then
  PROXY_FILTER_EXTRA=(--proxy_filter "${PROXY_FILTER}")
fi

LR_EXTRA=()
if [[ -n "${TRAIN_LEARNING_RATE}" ]]; then
  LR_EXTRA=(--learning_rate "${TRAIN_LEARNING_RATE}")
fi

# Optional discrete CE (tfbind8/10): set DUO_DISCRETE_CE_LAMBDA (and optionally DUO_DISCRETE_CE_TASK_NAMES).
# When RUN_SUFFIX is empty, we auto-append _ce... to keep outputs separated.
RUN_SUFFIX="${RUN_SUFFIX:-}"
if [[ -z "${RUN_SUFFIX}" && -n "${DUO_DISCRETE_CE_LAMBDA:-}" ]]; then
  # Only append suffix when lambda > 0 (lambda=0 keeps historical path).
  _lam_raw="${DUO_DISCRETE_CE_LAMBDA// /}"
  # Use python3 -c to avoid heredoc/fi parsing pitfalls in some shells.
  if python3 -c "import sys
try:
    v = float('${_lam_raw}')
except Exception:
    sys.exit(1)
sys.exit(0 if v > 0.0 else 2)" >/dev/null 2>&1; then
    RUN_SUFFIX="_ce${_lam_raw}"
  fi
fi

# If caller passed --run_suffix via TRAIN_EXTRA, reflect it in archive slug as well.
# This keeps results/... aligned with trained_models/... without requiring manual RUN_SUFFIX.
if [[ -z "${RUN_SUFFIX}" && -n "${TRAIN_EXTRA:-}" ]]; then
  case " ${TRAIN_EXTRA} " in
    *" --run_suffix "*)
      # shellcheck disable=SC2206
      _te_words=(${TRAIN_EXTRA})
      for ((i = 0; i < ${#_te_words[@]}; i++)); do
        if [[ "${_te_words[$i]}" == "--run_suffix" && $((i + 1)) -lt ${#_te_words[@]} ]]; then
          RUN_SUFFIX="${_te_words[$((i + 1))]}"
          break
        fi
      done
      ;;
    *" --run_suffix="*)
      # Grab last occurrence; simplest robust parse for " --run_suffix=_ce0.005 " style.
      RUN_SUFFIX="${TRAIN_EXTRA##*--run_suffix=}"
      RUN_SUFFIX="${RUN_SUFFIX%% *}"
      ;;
  esac
fi

# Task-scheduler runs: append a canonical suffix so outputs never collide with baseline.
if [[ -n "${DUO_TASK_SCHEDULER:-}" ]]; then
  _ts="${DUO_TASK_SCHEDULER// /}"
  _ts="${_ts,,}"
  if [[ "${_ts}" == "1" || "${_ts}" == "true" || "${_ts}" == "yes" || "${_ts}" == "on" ]]; then
    if [[ -z "${RUN_SUFFIX}" ]]; then
      RUN_SUFFIX="_tsched"
    elif [[ "${RUN_SUFFIX}" != *"_tsched"* ]]; then
      RUN_SUFFIX="${RUN_SUFFIX}_tsched"
    fi
  fi
fi

# ---------- Build SEED list ----------
SEED_LIST=()
if [[ -n "${SEEDS:-}" ]]; then
  _sn="${SEEDS//,/ }"
  read -ra SEED_LIST <<< "${_sn}"
elif [[ "${NUM_SEEDS}" -gt 1 ]]; then
  _ns="${NUM_SEEDS}"
  for ((i = 0; i < _ns; i++)); do
    SEED_LIST+=("$((START_SEED + i))")
  done
else
  if [[ -n "${SEED}" ]]; then
    SEED_LIST=("${SEED}")
  else
    SEED_LIST=("${START_SEED}")
  fi
fi

if [[ ${#SEED_LIST[@]} -eq 0 ]]; then
  echo "ERROR: unable to build seed list (check SEEDS / NUM_SEEDS / SEED)" >&2
  exit 1
fi

export PROJECT_ROOT
TRAIN_TASKS_TOKEN="$(canonical_train_tasks_token "${TRAIN_TASKS}")"

SWEEP_ARCHIVE_SLUG=""
_print_hyper_args=(
  python "${PROJECT_ROOT}/scripts/print_multitask_ckpt_hyper_dir.py"
  --train_tasks "${TRAIN_TASKS}"
  --frac "${FRAC}"
  --sigma "${SIGMA}"
  --n_traj "${N_TRAJ}"
  --k "${K}"
  --eps "${EPS}"
  --horizon "${HORIZON}"
  --hyper_suffix "${RUN_SUFFIX:-}"
  --latent_dim "${LATENT_DIM}"
  --train_timestep_bias_power "${TRAIN_TIMESTEP_BIAS_POWER}"
  --train_loss_min_snr_gamma "${TRAIN_LOSS_MIN_SNR_GAMMA}"
  --train_half_timestep_bias_frac "${TRAIN_HALF_TBIAS_FRAC}"
  --train_half_lr_mult "${TRAIN_HALF_LR_MULT}"
)
if [[ ${#LR_EXTRA[@]} -gt 0 ]]; then
  _print_hyper_args+=("${LR_EXTRA[@]}")
fi
if [[ -n "${TRAJ_PARAMS_JSON}" ]]; then
  _print_hyper_args+=(--traj_params_json "${TRAJ_PARAMS_JSON}")
fi
if _py_hyper="$("${_print_hyper_args[@]}" 2>/dev/null)" && [[ -n "${_py_hyper}" ]]; then
  SWEEP_ARCHIVE_SLUG="${_py_hyper}"
fi
if [[ -z "${SWEEP_ARCHIVE_SLUG}" ]]; then
  echo "[train_eval_sweep_w_text] WARN: print_multitask_ckpt_hyper_dir failed; using fallback slug (may not match checkpoint dir)" >&2
  _eps_slug="${EPS//./p}"
  _tpj_slug=""
  if [[ -n "${TRAJ_PARAMS_JSON}" ]]; then
    _tpj_slug="_$(basename "${TRAJ_PARAMS_JSON}")"
    _tpj_slug="${_tpj_slug// /_}"
  fi
  SWEEP_ARCHIVE_SLUG="${TRAIN_TASKS_TOKEN}_frac${FRAC}_sigma${SIGMA}_n${N_TRAJ}x${HORIZON}_k${K}_eps${_eps_slug}_lat${LATENT_DIM}${RUN_SUFFIX:-}${_tpj_slug}"
fi
MODEL_ROOT="${EVAL_SWEEP_DIR}/${SWEEP_ARCHIVE_SLUG}"
mkdir -p "${MODEL_ROOT}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
{
  echo "# train_eval_sweep_w_text"
  echo "# started ${RUN_ID}"
  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "MODEL_ROOT=${MODEL_ROOT}"
  echo "TRAIN_TASKS=${TRAIN_TASKS}"
  echo "FRAC=${FRAC} SIGMA=${SIGMA}"
  echo "SEED_LIST=${SEED_LIST[*]}"
  echo "START_SEED=${START_SEED} NUM_SEEDS=${NUM_SEEDS} SEEDS=${SEEDS:-}"
  echo "TRAJ_PARAMS_JSON=${TRAJ_PARAMS_JSON:-}"
  echo "LATENT_DIM=${LATENT_DIM}"
  echo "TRAIN_TIMESTEP_BIAS_POWER=${TRAIN_TIMESTEP_BIAS_POWER}"
  echo "TRAIN_LOSS_MIN_SNR_GAMMA=${TRAIN_LOSS_MIN_SNR_GAMMA}"
  echo "TRAIN_LEARNING_RATE=${TRAIN_LEARNING_RATE:-}"
  echo "TRAIN_HALF_TBIAS_FRAC=${TRAIN_HALF_TBIAS_FRAC}"
  echo "TRAIN_HALF_LR_MULT=${TRAIN_HALF_LR_MULT}"
  echo "PROXY_FILTER=${PROXY_FILTER:-}"
  echo "DUO_DISCRETE_CE_LAMBDA=${DUO_DISCRETE_CE_LAMBDA:-}"
  echo "DUO_DISCRETE_CE_TASK_NAMES=${DUO_DISCRETE_CE_TASK_NAMES:-}"
  echo "RUN_SUFFIX=${RUN_SUFFIX:-}"
  echo "W_VALUES=${W_VALUES[*]}"
  echo "TRAIN_EPOCHS=${TRAIN_EPOCHS}"
  echo "DIFFUSION_CKPT_TRAIN_STEPS=${DIFFUSION_CKPT_TRAIN_STEPS:-}"
  echo "SKIP_TRAIN=${SKIP_TRAIN} SKIP_EVAL=${SKIP_EVAL}"
  echo "SWEEP_ARCHIVE_SLUG=${SWEEP_ARCHIVE_SLUG}"
  echo "---"
} > "${MODEL_ROOT}/run_meta_${RUN_ID}.env"

for SEED in "${SEED_LIST[@]}"; do
  LOG_ROOT="${MODEL_ROOT}/seed${SEED}"
  mkdir -p "${LOG_ROOT}"
  SWEEP_LOG="${LOG_ROOT}/sweep.log"

  _log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    printf '[%s] %s\n' "${ts}" "$*" >> "${SWEEP_LOG}"
    if [[ "${LOG_TO_TERMINAL}" == "1" ]]; then
      printf '[seed %s][%s] %s\n' "${SEED}" "${ts}" "$*"
    fi
  }

  _log "Log dir: ${LOG_ROOT} (seed logs; archive root: ${MODEL_ROOT})"

  _TRAIN_EXTRA=(
    python train.py
    --train_tasks "${TRAIN_TASKS}"
    --frac "${FRAC}"
    --sigma "${SIGMA}"
    --n_traj "${N_TRAJ}"
    --k "${K}"
    --eps "${EPS}"
    --horizon "${HORIZON}"
    --seed "${SEED}"
    --use_text_condition
    --multitask_text_only
    --text_encoder_model "${TEXT_ENCODER}"
    --train_epochs "${TRAIN_EPOCHS}"
  )
  if [[ -n "${RUN_SUFFIX}" ]]; then
    _TRAIN_EXTRA+=(--run_suffix "${RUN_SUFFIX}")
  fi
  if [[ -n "${TRAJ_PARAMS_JSON}" ]]; then
    _TRAIN_EXTRA+=(--traj_params_json "${TRAJ_PARAMS_JSON}")
  fi
  _TRAIN_EXTRA+=(--latent_dim "${LATENT_DIM}")
  _TRAIN_EXTRA+=(--train_timestep_bias_power "${TRAIN_TIMESTEP_BIAS_POWER}")
  _TRAIN_EXTRA+=(--train_loss_min_snr_gamma "${TRAIN_LOSS_MIN_SNR_GAMMA}")
  _TRAIN_EXTRA+=(--train_half_timestep_bias_frac "${TRAIN_HALF_TBIAS_FRAC}")
  _TRAIN_EXTRA+=(--train_half_lr_mult "${TRAIN_HALF_LR_MULT}")
  _TRAIN_EXTRA+=("${LR_EXTRA[@]}")
  _TRAIN_EXTRA+=("${PROXY_FILTER_EXTRA[@]}")

  _EVAL_BASE=(
    python evaluate.py
    --train_tasks "${TRAIN_TASKS}"
    --frac "${FRAC}"
    --sigma "${SIGMA}"
    --n_traj "${N_TRAJ}"
    --k "${K}"
    --eps "${EPS}"
    --horizon "${HORIZON}"
    --seed "${SEED}"
    --use_text_condition
    --multitask_text_only
    --text_encoder_model "${TEXT_ENCODER}"
  )
  if [[ -n "${RUN_SUFFIX}" ]]; then
    _EVAL_BASE+=(--run_suffix "${RUN_SUFFIX}")
  fi
  if [[ -n "${TRAJ_PARAMS_JSON}" ]]; then
    _EVAL_BASE+=(--traj_params_json "${TRAJ_PARAMS_JSON}")
  fi
  _EVAL_BASE+=(--latent_dim "${LATENT_DIM}")
  _EVAL_BASE+=(--train_epochs "${TRAIN_EPOCHS}")
  _EVAL_BASE+=(--train_timestep_bias_power "${TRAIN_TIMESTEP_BIAS_POWER}")
  _EVAL_BASE+=(--train_loss_min_snr_gamma "${TRAIN_LOSS_MIN_SNR_GAMMA}")
  _EVAL_BASE+=(--train_half_timestep_bias_frac "${TRAIN_HALF_TBIAS_FRAC}")
  _EVAL_BASE+=(--train_half_lr_mult "${TRAIN_HALF_LR_MULT}")
  _EVAL_BASE+=("${LR_EXTRA[@]}")
  _EVAL_BASE+=("${PROXY_FILTER_EXTRA[@]}")
  if [[ -n "${DIFFUSION_CKPT_TRAIN_STEPS}" ]]; then
    _EVAL_BASE+=(--diffusion_ckpt_train_steps "${DIFFUSION_CKPT_TRAIN_STEPS}")
  fi

  if [[ "${SKIP_TRAIN}" != "1" ]]; then
    _log "=== [1/2] Training seed=${SEED} (${LOG_ROOT}/train.log) ==="
    _train_ec=0
    "${_TRAIN_EXTRA[@]}" >> "${LOG_ROOT}/train.log" 2>&1 || _train_ec=$?
    if [[ "${_train_ec}" -ne 0 ]]; then
      _log "ERROR: train exit_code=${_train_ec}; see ${LOG_ROOT}/train.log; abort"
      exit "${_train_ec}"
    fi
  else
    _log "=== [1/2] SKIP_TRAIN=1; skip training seed=${SEED} ==="
  fi

  if [[ "${SKIP_EVAL}" != "1" ]]; then
    _log "=== [2/2] Evaluation sweep w_text (${LOG_ROOT}/eval_w*.log) ==="
    for W in "${W_VALUES[@]}"; do
      _wfile="${W//./p}"
      _log "--- w_text=${W} -> ${LOG_ROOT}/eval_w${_wfile}.log ---"
      # capture exit code: avoid set -e exiting on evaluate non-zero
      _ev=0
      "${_EVAL_BASE[@]}" --condition_guidance_w_text "${W}" >> "${LOG_ROOT}/eval_w${_wfile}.log" 2>&1 || _ev=$?
      if [[ "${_ev}" -ne 0 ]]; then
        _log "WARN: evaluate w=${W} exit_code=${_ev}; see ${LOG_ROOT}/eval_w${_wfile}.log (continue)"
      fi
    done
  fi
  if [[ "${SKIP_EVAL}" == "1" ]]; then
    _log "=== [2/2] SKIP_EVAL=1; skip eval seed=${SEED} ==="
  fi

  _log "Done seed=${SEED}. Project root: ${PROJECT_ROOT}"
done

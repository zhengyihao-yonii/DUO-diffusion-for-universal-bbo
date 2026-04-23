#!/usr/bin/env bash
# 多任务 + 文本条件：按 seed 训练（可多次），再对每个 seed 扫 condition_guidance_w_text（text CFG）。
#
# 随机种子（与 run_multitask 类似，三选一，优先级从上到下）:
#   1) SEEDS="0 1 2" 或 SEEDS="0,1,2"  — 显式列表
#   2) NUM_SEEDS（或 NUM_RUNS）+ START_SEED — 连续 seed: START_SEED..START_SEED+NUM_SEEDS-1
#   3) 仅 SEED — 单次；未设 SEED 时用 START_SEED（默认 0）。例：NUM_SEEDS=1 START_SEED=1 只写 START_SEED 即可
#
# 示例:
#   SEEDS="0 1 2" ./scripts/train_eval_sweep_w_text.sh
#   NUM_SEEDS=3 START_SEED=0 ./scripts/train_eval_sweep_w_text.sh
#   NUM_RUNS=3 START_SEED=1 ./scripts/train_eval_sweep_w_text.sh
#
# 日志目录：归档 train.log / eval_w*.log / sweep.log；默认与 trained_models/multi_* 下 RUN.prefix 中段一致。
#   results/eval_sweep_w_text/<SWEEP_ARCHIVE_SLUG>/seed<SEED>/
# SWEEP_ARCHIVE_SLUG 默认由 scripts/print_multitask_ckpt_hyper_dir.py 输出（mt_<hex>_textcond_mttextonly + RUN_SUFFIX + _latent{d}），
#   与 train.py / evaluate.py 的 multitask 超参目录名一致；若 Python 失败则回退为任务名 + 标量超参的长 slug。
#   同一组参数多次运行仍写入同一目录，便于补跑部分 w（eval 日志 >> 追加）。
# 每次运行在本目录下追加 run_meta_<时间戳>.env。
# LATENT_DIM  默认 32；须与 train/eval 的 --latent_dim 一致。可用环境变量 LATENT_DIM=64 或命令行：
#   ./scripts/train_eval_sweep_w_text.sh --latent_dim 64
#   （命令行优先于环境变量中的 LATENT_DIM）
#
# TRAIN_TIMESTEP_BIAS_POWER / TRAIN_LOSS_MIN_SNR_GAMMA  默认 0.0；非零时传给 train.py（小 t 偏斜 / min-SNR）。
#
# PROXY_FILTER  可选 0/1；若 export 则 train/evaluate 追加 --proxy_filter（不设则默认 1）。
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---------- 解析命令行（须在 LATENT_DIM 默认值之前）----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --latent_dim=*)
      LATENT_DIM="${1#*=}"
      shift
      ;;
    --latent_dim)
      if [[ -z "${2:-}" ]]; then
        echo "错误: --latent_dim 需要整数参数" >&2
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
      echo "未知参数: $1（支持: --latent_dim <int>、--latent_dim=<int>、-h）" >&2
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

# ========= 可按需修改 =========
TRAIN_TASKS="${TRAIN_TASKS:-ant,dkitty,gtopx2,gtopx3,gtopx4,gtopx6,superconductor,tfbind10,tfbind8}"
FRAC="${FRAC:-1.0}"
SIGMA="${SIGMA:-0.0}"
N_TRAJ="${N_TRAJ:-1000}"
K="${K:-50}"
EPS="${EPS:-0.05}"
HORIZON="${HORIZON:-64}"
# SEED 空=未显式指定；NUM_SEEDS==1 时单次 seed 用 START_SEED（与下面 SEED_LIST 一致）
SEED="${SEED:-}"
START_SEED="${START_SEED:-0}"
NUM_SEEDS="${NUM_SEEDS:-${NUM_RUNS:-1}}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-200}"
TEXT_ENCODER="${TEXT_ENCODER:-sentence-transformers/all-MiniLM-L6-v2}"

TRAJ_PARAMS_JSON="${TRAJ_PARAMS_JSON:-}"
LATENT_DIM="${LATENT_DIM:-32}"
TRAIN_TIMESTEP_BIAS_POWER="${TRAIN_TIMESTEP_BIAS_POWER:-0.0}"
TRAIN_LOSS_MIN_SNR_GAMMA="${TRAIN_LOSS_MIN_SNR_GAMMA:-0.0}"

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

# 离散任务辅助 CE（tfbind8/10）：只需设置环境变量 DUO_DISCRETE_CE_LAMBDA（以及可选 DUO_DISCRETE_CE_TASK_NAMES）。
# 为避免覆盖默认实验，自动将该 lambda 追加到 hyper 目录名与 RUN.prefix（train/evaluate 的 --run_suffix）。
RUN_SUFFIX="${RUN_SUFFIX:-}"
if [[ -z "${RUN_SUFFIX}" && -n "${DUO_DISCRETE_CE_LAMBDA:-}" ]]; then
  # 只有 lambda > 0 才加后缀；lambda=0 时保持与历史实验同一路径（不新增目录）
  _lam_raw="${DUO_DISCRETE_CE_LAMBDA// /}"
  if python3 - <<PY >/dev/null 2>&1
import sys
try:
    v = float("${_lam_raw}")
except Exception:
    sys.exit(1)
sys.exit(0 if v > 0.0 else 2)
PY
  then
    RUN_SUFFIX="_ce${_lam_raw}"
  fi
fi

# ---------- 解析 SEED 列表 ----------
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
  echo "错误: 无法得到 seed 列表（请检查 SEEDS / NUM_SEEDS / SEED）" >&2
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
)
if [[ -n "${TRAJ_PARAMS_JSON}" ]]; then
  _print_hyper_args+=(--traj_params_json "${TRAJ_PARAMS_JSON}")
fi
if _py_hyper="$("${_print_hyper_args[@]}" 2>/dev/null)" && [[ -n "${_py_hyper}" ]]; then
  SWEEP_ARCHIVE_SLUG="${_py_hyper}"
fi
if [[ -z "${SWEEP_ARCHIVE_SLUG}" ]]; then
  echo "[train_eval_sweep_w_text] 警告: print_multitask_ckpt_hyper_dir 失败，使用回退 slug（与 checkpoint 中段可能不一致）" >&2
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
  echo "PROXY_FILTER=${PROXY_FILTER:-}"
  echo "DUO_DISCRETE_CE_LAMBDA=${DUO_DISCRETE_CE_LAMBDA:-}"
  echo "DUO_DISCRETE_CE_TASK_NAMES=${DUO_DISCRETE_CE_TASK_NAMES:-}"
  echo "RUN_SUFFIX=${RUN_SUFFIX:-}"
  echo "W_VALUES=${W_VALUES[*]}"
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

  _log "本目录: ${LOG_ROOT}（本 seed 的训练/评估日志；归档根: ${MODEL_ROOT}）"

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
  _EVAL_BASE+=(--train_timestep_bias_power "${TRAIN_TIMESTEP_BIAS_POWER}")
  _EVAL_BASE+=(--train_loss_min_snr_gamma "${TRAIN_LOSS_MIN_SNR_GAMMA}")
  _EVAL_BASE+=("${PROXY_FILTER_EXTRA[@]}")

  if [[ "${SKIP_TRAIN}" != "1" ]]; then
    _log "=== [1/2] Training seed=${SEED}（${LOG_ROOT}/train.log）==="
    _train_ec=0
    "${_TRAIN_EXTRA[@]}" >> "${LOG_ROOT}/train.log" 2>&1 || _train_ec=$?
    if [[ "${_train_ec}" -ne 0 ]]; then
      _log "错误: train 退出码 ${_train_ec}，见 ${LOG_ROOT}/train.log；中止本脚本"
      exit "${_train_ec}"
    fi
  else
    _log "=== [1/2] SKIP_TRAIN=1，跳过训练 seed=${SEED} ==="
  fi

  if [[ "${SKIP_EVAL}" != "1" ]]; then
    _log "=== [2/2] Evaluation sweep w_text（${LOG_ROOT}/eval_w*.log）==="
    for W in "${W_VALUES[@]}"; do
      _wfile="${W//./p}"
      _log "--- w_text=${W} → ${LOG_ROOT}/eval_w${_wfile}.log ---"
      # 捕获退出码：避免 set -e 在 evaluate 非零时直接退出，导致 sweep.log 无 Done、终端无提示
      _ev=0
      "${_EVAL_BASE[@]}" --condition_guidance_w_text "${W}" >> "${LOG_ROOT}/eval_w${_wfile}.log" 2>&1 || _ev=$?
      if [[ "${_ev}" -ne 0 ]]; then
        _log "警告: evaluate w=${W} 退出码 ${_ev}，详见 ${LOG_ROOT}/eval_w${_wfile}.log（继续后续 w）"
      fi
    done
  fi
  if [[ "${SKIP_EVAL}" == "1" ]]; then
    _log "=== [2/2] SKIP_EVAL=1，跳过评估 seed=${SEED} ==="
  fi

  _log "Done seed=${SEED}. Project root: ${PROJECT_ROOT}"
done

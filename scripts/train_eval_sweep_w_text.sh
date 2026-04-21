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
# 日志目录（与 checkpoint 中间段 ``mt_<16hex>_textcond_mttextonly`` 一致，见 print_multitask_ckpt_hyper_dir.py）:
#   results/eval_sweep_w_text/<CKPT_HYPER_DIR>/seed<SEED>/
# 同一套轨迹参数（同一 mt_*）下多次执行会**写入同一 seed 目录**（eval 日志用 >> 追加），便于补跑部分 w（如 W_VALUES_OVERRIDE="16.0 32.0"）。
# 每次运行会在 <CKPT_HYPER_DIR>/ 下追加 run_meta_<时间戳>.env 记录参数。
#
# 默认不把 checkpoint 下的 npz 复制到 results（SWEEP_COPY_NPZ=0）；需要本地留档时: SWEEP_COPY_NPZ=1 ./scripts/train_eval_sweep_w_text.sh
#
# PROXY_FILTER  可选 0/1；若 export 则 train/evaluate 追加 --proxy_filter（不设则默认 1）。
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

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

if [[ -n "${W_VALUES_OVERRIDE:-}" ]]; then
  read -ra W_VALUES <<< "${W_VALUES_OVERRIDE}"
else
  W_VALUES=(0.0 0.8 1.2 2.0)
fi

CKPT_HYPER_DIR="${CKPT_HYPER_DIR:-}"
EVAL_SWEEP_DIR="${EVAL_SWEEP_DIR:-results/eval_sweep_w_text}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
LOG_TO_TERMINAL="${LOG_TO_TERMINAL:-0}"
# 是否把 checkpoint 目录下与当前 w 匹配的 performance/samples *.npz 复制到 seed 目录下的 npz_w*/（默认 0 不复制，减小 results 体积）
SWEEP_COPY_NPZ="${SWEEP_COPY_NPZ:-0}"

PROXY_FILTER_EXTRA=()
if [[ -n "${PROXY_FILTER:-}" ]]; then
  PROXY_FILTER_EXTRA=(--proxy_filter "${PROXY_FILTER}")
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
_multi_base="trained_models/multi_${TRAIN_TASKS_TOKEN}_frac${FRAC}_sigma${SIGMA}"

_PRINT_MT=(
  python3 "${PROJECT_ROOT}/scripts/print_multitask_ckpt_hyper_dir.py"
  --train_tasks "${TRAIN_TASKS}"
  --frac "${FRAC}"
  --sigma "${SIGMA}"
  --n_traj "${N_TRAJ}"
  --k "${K}"
  --eps "${EPS}"
  --horizon "${HORIZON}"
)
if [[ -n "${TRAJ_PARAMS_JSON}" ]]; then
  _PRINT_MT+=(--traj_params_json "${TRAJ_PARAMS_JSON}")
fi

# 超参目录名（与 train/eval 的 RUN.prefix 中间段一致，如 mt_911054c35daad7e0_textcond_mttextonly）
# 只取以 mt_ 开头的行：避免 import 时误打到 stdout 的提示混进 ``$(...)`` 变成错误目录名（曾见 colab 警告）
if [[ -z "${CKPT_HYPER_DIR}" ]]; then
  _hyper_lines="$("${_PRINT_MT[@]}" 2>&1)" || true
  CKPT_HYPER_DIR="$(printf '%s\n' "${_hyper_lines}" | grep -E '^mt_' | tail -n 1)"
  if [[ -z "${CKPT_HYPER_DIR}" && -n "${_hyper_lines}" ]]; then
    echo "错误: print_multitask_ckpt_hyper_dir 未得到 mt_* 行，完整输出：" >&2
    printf '%s\n' "${_hyper_lines}" >&2
  fi
fi
if [[ -z "${CKPT_HYPER_DIR}" && -d "${_multi_base}" ]]; then
  _cand=()
  while IFS= read -r -d '' d; do _cand+=("$d"); done < <(find "${_multi_base}" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null || true)
  if [[ ${#_cand[@]} -eq 1 ]]; then
    CKPT_HYPER_DIR="$(basename "${_cand[0]}")"
  fi
fi
if [[ -z "${CKPT_HYPER_DIR}" ]]; then
  echo "错误: 无法确定 CKPT_HYPER_DIR（print_multitask_ckpt_hyper_dir 失败且 trained_models 下无唯一目录）；可手动设置 CKPT_HYPER_DIR=" >&2
  exit 1
fi

MODEL_ROOT="${EVAL_SWEEP_DIR}/${CKPT_HYPER_DIR}"
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
  echo "PROXY_FILTER=${PROXY_FILTER:-}"
  echo "W_VALUES=${W_VALUES[*]}"
  echo "SKIP_TRAIN=${SKIP_TRAIN} SKIP_EVAL=${SKIP_EVAL}"
  echo "CKPT_HYPER_DIR=${CKPT_HYPER_DIR}"
  echo "SWEEP_COPY_NPZ=${SWEEP_COPY_NPZ}"
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

  _log "本目录: ${LOG_ROOT}（本 seed 的训练/评估日志）"
  if [[ -n "${CKPT_HYPER_DIR:-}" ]]; then
    _log "CKPT_HYPER_DIR=${CKPT_HYPER_DIR}（checkpoint: ${_multi_base}/${CKPT_HYPER_DIR}/seed${SEED}/）"
  fi

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
  if [[ -n "${TRAJ_PARAMS_JSON}" ]]; then
    _TRAIN_EXTRA+=(--traj_params_json "${TRAJ_PARAMS_JSON}")
  fi
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
  if [[ -n "${TRAJ_PARAMS_JSON}" ]]; then
    _EVAL_BASE+=(--traj_params_json "${TRAJ_PARAMS_JSON}")
  fi
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

      if [[ "${SWEEP_COPY_NPZ}" == "1" ]]; then
        if [[ -n "${CKPT_HYPER_DIR:-}" ]]; then
          _ckpt_root="${_multi_base}/${CKPT_HYPER_DIR}/seed${SEED}"
          _dst="${LOG_ROOT}/npz_w${_wfile}"
          mkdir -p "${_dst}"
          _wtag="$(python3 -c "print(f\"_wtext{float('${W}'):g}\")")"
          shopt -s nullglob
          _npz=( "${_ckpt_root}"/performance_*"${_wtag}"*.npz "${_ckpt_root}"/samples_*"${_wtag}"*.npz )
          shopt -u nullglob
          if [[ ${#_npz[@]} -gt 0 ]]; then
            cp -f "${_npz[@]}" "${_dst}/" || true
            _log "已复制 npz 到 ${_dst}/"
          else
            _log "未找到 ${_ckpt_root}/*${_wtag}*.npz"
          fi
        else
          _log "SWEEP_COPY_NPZ=1 但无法确定 CKPT_HYPER_DIR，跳过 npz 复制"
        fi
      fi
    done
  fi
  if [[ "${SKIP_EVAL}" == "1" ]]; then
    _log "=== [2/2] SKIP_EVAL=1，跳过评估 seed=${SEED} ==="
  fi

  _log "Done seed=${SEED}. Project root: ${PROJECT_ROOT}"
done

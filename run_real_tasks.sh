#!/usr/bin/env bash
#
# =============================================================================
# 真实任务：Zero-shot / Few-shot 统一入口
# =============================================================================
#
# 用法:
#   bash run_real_tasks.sh [选项...] [MODE] [TASK]
#
#   MODE   all | zero_shot | few_shot
#          默认 all：先 zero-shot 再 few-shot
#   TASK   all | lunar_lander | robot_push | rover
#          默认 all：三个任务依次执行
#
# 随机种子（与 scripts/train_eval_sweep_w_text.sh 中 NUM_SEEDS + START_SEED 语义一致）:
#   NUM_SEEDS + START_SEED  — 连续 seed: START_SEED .. START_SEED+NUM_SEEDS-1
#   Zero-shot 与 Few-shot 共用
# 常用选项（与 train.py / evaluate.py 一致）:
#   --pretrained_mt_hex 911054c35daad7e0
#   --pretrained_diffusion_seed 0
#   --n_traj 100 --k 20 --eps 0.05
#   --horizon 64 --frac 1.0 --sigma 0.0
#   --num_seeds 5  # 重复次数（默认 1）
#   --start_seed 0 # 第一个 seed（默认 0）
#   PROXY_FILTER   可选 0/1（环境变量）：仅 few-shot 的 train/evaluate 会传 --proxy_filter（不设则默认 1）。
#                  Zero-shot 阶段始终关闭 proxy（--proxy_filter 0），不训 proxy、评估时不筛选。
#   SKIP_TRAIN_IF_CKPT  默认 1：few-shot 若 trained_models/.../fewshot_ft.../seed*/checkpoint/ 已有
#                  state.pt 或 state_*.pt，则跳过 train.py 仅跑 evaluate（重跑失败评估时不必重训）。
#                  FORCE_FEWSHOT_TRAIN=1 强制重新微调。
#   --eval-only 或 EVAL_ONLY=1：只跑 evaluate.py；zero-shot / few-shot 均不跑 construct、不跑 train。
#                  few-shot 需已有同目录下 checkpoint；可用 BATCH_RUN 指定批次，RESULTS 指向已有实验根目录。
#   --latent_dim D  与 train/evaluate/construct 一致（默认 32）；非 32 时 few-shot 默认 vae 元数据为
#                  generated_datasets/multi_*/vae_info_latent{D}.p，且 _fewshot_trained_ckpt_dir 含 _latent{D}。
#   TRAIN_TIMESTEP_BIAS_POWER / TRAIN_LOSS_MIN_SNR_GAMMA  环境变量，默认 0.0；few-shot train.py 会传入对应 CLI（扩散训练可选改进）。
#   CONDITION_GUIDANCE_W_TEXT  文本 CFG 权重（--condition_guidance_w_text），默认 8（与 eval_sweep_w_text 中 w=8 对齐）。
#                  也可 --condition_guidance_w_text <x> 或 --w_text <x>。
#   --pretrained_vae_info /abs/path/vae_info.p   # few-shot；不设则见下方自动解析
#   --pretrained_multitask_train_tasks 'ant,dkitty,...'  # 与多任务预训练 CSV 一致，用于默认 VAE 路径
#
# Few-shot 与 PRETRAINED_VAE_INFO:
#   多任务 VAE 产物路径为 generated_datasets/multi_<token>_frac<f>_sigma<s>/vae_info.p
#   与扩散的 mt_<hex>_textcond_mttextonly 目录无关（hex 只标在 trained_models 下）。
#   若未传 --pretrained_vae_info，脚本会按 PRETRAINED_MULTITASK_TRAIN_TASKS + FRAC + SIGMA
#   拼出默认路径；文件存在则自动使用，否则报错并提示手动指定。
#
# 结果目录（与 run_singletask → results/single_task、run_multitask → results/multi_task 并列）:
#   REAL_TASK_RESULTS_ROOT  默认 PROJECT/results/real_task；可用 --results-root 或环境变量覆盖。
#   Zero-shot:  $REAL_TASK_RESULTS_ROOT/zero_shot/<task>_frac<F>_sigma<S>/
#                 <n_traj>x<h>_k<k>_eps<eps>[_ret]_mt<hex>_pds<seed>_zs/run<N>_seed<S>/evaluate.log
#   真实任务 few-shot：评估前需要 construct 生成的单任务轨迹 pkl（Step 1）。
#   真实任务 zero-shot：不构造轨迹 pkl；evaluate 固定无上下文采样（README）。
#   Few-shot 默认 FEWSHOT_K=128、FEWSHOT_MODE=worst（见 README）。
#   Few-shot:   $REAL_TASK_RESULTS_ROOT/few_shot/<task>_frac.../fs_<tag>/
#                 <hyper>_mt<hex>_ft/construct_trajectories.log + run<N>_seed*/{train,evaluate}.log
#   若 export RESULTS=/某次实验的完整目录，则仅 Few-shot 时覆盖该任务的 base_dir（与旧版一致，便于单任务重跑）。
#
# Weights & Biases（默认在线同步；网络不稳可用离线）:
#   --wandb-offline     等价于 export GTG_WANDB_OFFLINE=1 WANDB_MODE=offline（train/evaluate 内 wandb 仅写本地）
#   GTG_WANDB_OFFLINE=1 或 WANDB_MODE=offline  同上，可与 sweep 脚本一致
#   WANDB_INIT_TIMEOUT=300  在线模式 init 超时秒数（默认 300）
#   WANDB_DISABLED=1        完全跳过 wandb.init
#
# 仍可用环境变量覆盖（与选项二选一，命令行优先于本脚本默认值，建议在 shell 里 export 的仍生效于未写的选项）:
#   解析方式：先读环境变量，再解析命令行覆盖，最后对空值填默认。
#
# =============================================================================

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_SCRIPT_DIR/scripts/gpu_env.sh" ]]; then
  # shellcheck source=scripts/gpu_env.sh
  source "$_SCRIPT_DIR/scripts/gpu_env.sh"
fi

PROJECT="${PROJECT:-$_SCRIPT_DIR}"
PYTHON="${PYTHON:-python3}"
cd "$PROJECT"
# 重定向到 evaluate.log/train.log 时默认块缓冲，崩溃前看不到最后几步；需要实时日志时保持为 1
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# ---------- 先保留环境变量，再由命令行覆盖 ----------
POSITIONAL=()
NUM_SEEDS_CLI=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pretrained_mt_hex)
      PRETRAINED_MT_HEX="$2"
      shift 2
      ;;
    --pretrained_diffusion_seed)
      PRETRAINED_DIFFUSION_SEED="$2"
      shift 2
      ;;
    --pretrained_vae_info)
      PRETRAINED_VAE_INFO="$2"
      shift 2
      ;;
    --pretrained_multitask_train_tasks)
      PRETRAINED_MULTITASK_TRAIN_TASKS="$2"
      shift 2
      ;;
    --n_traj)
      N_TRAJ="$2"
      shift 2
      ;;
    --k)
      K="$2"
      shift 2
      ;;
    --eps)
      EPS="$2"
      shift 2
      ;;
    --horizon)
      HORIZON="$2"
      shift 2
      ;;
    --frac)
      FRAC="$2"
      shift 2
      ;;
    --sigma)
      SIGMA="$2"
      shift 2
      ;;
    --num_seeds)
      NUM_SEEDS_CLI="$2"
      shift 2
      ;;
    --start_seed)
      START_SEED="$2"
      shift 2
      ;;
    --train_epochs)
      TRAIN_EPOCHS="$2"
      shift 2
      ;;
    --results-root)
      REAL_TASK_RESULTS_ROOT="$2"
      shift 2
      ;;
    --wandb-offline)
      GTG_WANDB_OFFLINE_CLI=1
      shift
      ;;
    --eval-only)
      EVAL_ONLY=1
      shift
      ;;
    --latent_dim)
      LATENT_DIM="$2"
      shift 2
      ;;
    --condition_guidance_w_text | --w_text)
      CONDITION_GUIDANCE_W_TEXT="$2"
      shift 2
      ;;
    -h | --help)
      sed -n '1,130p' "$0"
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        POSITIONAL+=("$1")
        shift
      done
      break
      ;;
    -*)
      echo "未知选项: $1（见 $0 --help）" >&2
      exit 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ "${GTG_WANDB_OFFLINE_CLI:-0}" == "1" ]]; then
  export GTG_WANDB_OFFLINE=1
  export WANDB_MODE=offline
  echo "[wandb] 离线模式（--wandb-offline）→ GTG_WANDB_OFFLINE=1, WANDB_MODE=offline"
fi

MODE="${POSITIONAL[0]:-all}"
TASK_ARG="${POSITIONAL[1]:-all}"

# ---------- 模型默认 ----------
PRETRAINED_MT_HEX="${PRETRAINED_MT_HEX:-911054c35daad7e0}"
PRETRAINED_DIFFUSION_SEED="${PRETRAINED_DIFFUSION_SEED:-0}"

# 与 diffuser/utils/real_task_transfer.py DEFAULT_PRETRAINED_MULTITASK_CSV 一致（用于默认 vae_info 路径）
PRETRAINED_MULTITASK_TRAIN_TASKS="${PRETRAINED_MULTITASK_TRAIN_TASKS:-ant,dkitty,gtopx2,gtopx3,gtopx4,gtopx6,superconductor,tfbind10,tfbind8}"

# ---------- 轨迹默认（100 / 20 / 0.05）----------
N_TRAJ="${N_TRAJ:-100}"
K="${K:-20}"
EPS="${EPS:-0.05}"
HORIZON="${HORIZON:-64}"
FRAC="${FRAC:-1.0}"
SIGMA="${SIGMA:-0.0}"

# ---------- 结果根目录（与 results/single_task、results/multi_task 并列）----------
REAL_TASK_RESULTS_ROOT="${REAL_TASK_RESULTS_ROOT:-$PROJECT/results/real_task}"

# 文本 classifier-free guidance（与 evaluate.py / train.py --condition_guidance_w_text 一致；默认 8）
CONDITION_GUIDANCE_W_TEXT="${CONDITION_GUIDANCE_W_TEXT:-8}"

# ---------- Few-shot 可选变量 ----------
# 默认：在合并后的 JSON 上取 y 最小的 128 个点（worst；越大越好）。
# 若需全量点（不子采样）：export FEWSHOT_K= 为空字符串，或在本脚本执行环境中显式置空。
# 仅当 FEWSHOT_K 完全未设置（unset）时使用默认 128。
PRETRAINED_VAE_INFO="${PRETRAINED_VAE_INFO:-}"
if [[ -z "${FEWSHOT_K+x}" ]]; then
  FEWSHOT_K=128
fi
FEWSHOT_MODE="${FEWSHOT_MODE:-worst}"
CONSTRUCT_SEED="${CONSTRUCT_SEED:-0}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-50}"
FINETUNE_LR="${FINETUNE_LR:-3e-5}"
START_SEED="${START_SEED:-0}"
if [[ -n "${NUM_SEEDS_CLI:-}" ]]; then
  NUM_SEEDS="$NUM_SEEDS_CLI"
else
  NUM_SEEDS="${NUM_SEEDS:-1}"
fi
if ! [[ "$NUM_SEEDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误: NUM_SEEDS 须为正整数，当前: ${NUM_SEEDS}" >&2
  exit 1
fi
# Few-shot 扩散微调默认 epoch；与 train.py 一致。
TRAIN_EPOCHS="${TRAIN_EPOCHS:-500}"
LATENT_DIM="${LATENT_DIM:-32}"
TRAIN_TIMESTEP_BIAS_POWER="${TRAIN_TIMESTEP_BIAS_POWER:-0.0}"
TRAIN_LOSS_MIN_SNR_GAMMA="${TRAIN_LOSS_MIN_SNR_GAMMA:-0.0}"
_DIFF_TRAIN_SUF="$(
  cd "$PROJECT" && PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -c \
    'import sys; from diffuser.utils.multitask_canon import diffusion_train_path_suffix; print(diffusion_train_path_suffix(float(sys.argv[1]), float(sys.argv[2])))' \
    "${TRAIN_TIMESTEP_BIAS_POWER}" "${TRAIN_LOSS_MIN_SNR_GAMMA}"
)"
EVAL_ONLY="${EVAL_ONLY:-0}"
USE_RETURNS="${USE_RETURNS:-0}"
AUTO_CONTINUE="${AUTO_CONTINUE:-0}"
CPU_THREADS="${CPU_THREADS:-}"

TASKS_ALL=(lunar_lander robot_push rover)

# few-shot：未手动指定时，按「多任务数据集目录」拼 vae_info.p（与 mt_<hex> 扩散目录无关）
_resolve_pretrained_vae_info_if_needed() {
  if [[ -n "${PRETRAINED_VAE_INFO}" ]]; then
    if [[ ! -f "$PRETRAINED_VAE_INFO" ]]; then
      echo "错误: PRETRAINED_VAE_INFO 不是有效文件: $PRETRAINED_VAE_INFO" >&2
      exit 1
    fi
    return 0
  fi
  local _cand
  _cand="$(
    FRAC="$FRAC" SIGMA="$SIGMA" PROJECT="$PROJECT" LATENT_DIM="$LATENT_DIM" \
      PRETRAINED_MULTITASK_TRAIN_TASKS="$PRETRAINED_MULTITASK_TRAIN_TASKS" \
      "$PYTHON" -c "
import importlib.util, os, sys
project = os.environ['PROJECT']
sys.path.insert(0, project)
path = os.path.join(project, 'diffuser/utils/multitask_canon.py')
spec = importlib.util.spec_from_file_location('multitask_canon', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from diffuser.utils.vae_layout import resolve_generated_vae_info_path
csv = os.environ['PRETRAINED_MULTITASK_TRAIN_TASKS']
frac = float(os.environ['FRAC'])
sigma = float(os.environ['SIGMA'])
ld = int(os.environ.get('LATENT_DIM', '32'))
tok = mod.multitask_path_token(mod.canonical_train_tasks_csv(csv))
base = os.path.join(project, f'generated_datasets/multi_{tok}_frac{frac}_sigma{sigma}')
p = resolve_generated_vae_info_path(base, ld)
print(p or '')
"
  )"
  if [[ -n "$_cand" ]] && [[ -f "$_cand" ]]; then
    PRETRAINED_VAE_INFO="$_cand"
    echo "[auto] PRETRAINED_VAE_INFO=$PRETRAINED_VAE_INFO"
    echo "      （multi 数据集目录 = PRETRAINED_MULTITASK_TRAIN_TASKS + FRAC/SIGMA；与 --pretrained_mt_hex 无关）"
    return 0
  fi
  echo "错误: few-shot 需要多任务 vae 元数据（vae_info.p 或 vae_info_latent*.p）。未指定 --pretrained_vae_info，且默认路径不存在：" >&2
  echo "      $_cand" >&2
  echo "      请先生成多任务 VAE，或显式传入 --pretrained_vae_info / export PRETRAINED_VAE_INFO=..." >&2
  exit 1
}

# 真实任务 zero-shot：evaluate 固定无上下文轨迹，不依赖单任务轨迹 pkl；需要时自行运行 construct_trajectories.py。

_max_batch_from_dirs() {
  local root="$1"
  local m=0
  shopt -s nullglob
  local d
  for d in "$root"/run*_seed*/; do
    [[ -d "$d" ]] || continue
    local b
    b=$(basename "$d")
    if [[ "$b" =~ ^run([0-9]+)_seed ]]; then
      local n="${BASH_REMATCH[1]}"
      if [[ "$n" =~ ^[0-9]+$ ]] && ((10#$n > m)); then
        m=$((10#$n))
      fi
    fi
  done
  shopt -u nullglob
  echo "$m"
}

# 与 train.py 单任务 --real_task_text_only_finetune 的 RUN.prefix 中段一致（_fewshot_ft + 条件后缀）
_fewshot_trained_ckpt_dir() {
  local t="$1" seed="$2"
  local _ret="" _txt _mto _lat=""
  if [[ "${LATENT_DIM:-32}" != "32" ]]; then
    _lat="_latent${LATENT_DIM}"
  fi
  if [[ "${USE_RETURNS:-0}" == "1" ]]; then
    _ret="${GTG_RETCOND_PATH_INFIX:-_retcond}"
  fi
  _txt="${GTG_TEXTCOND_PATH_INFIX:-_textcond}"
  _mto="${GTG_MTTEXTONLY_PATH_INFIX:-_mttextonly}"
  echo "${PROJECT}/trained_models/${t}_frac${FRAC}_sigma${SIGMA}/${N_TRAJ}x${HORIZON}_k${K}_eps${EPS}_fewshot_ft${_ret}${_txt}${_mto}${_DIFF_TRAIN_SUF:-}${_lat}/seed${seed}/checkpoint"
}

_tasks_to_run() {
  case "$TASK_ARG" in
    all | "")
      printf '%s\n' "${TASKS_ALL[@]}"
      ;;
    lunar_lander | robot_push | rover)
      echo "$TASK_ARG"
      ;;
    -h | --help)
      sed -n '1,55p' "$0"
      exit 0
      ;;
    *)
      echo "用法: $0 [all|zero_shot|few_shot] [all|lunar_lander|robot_push|rover]" >&2
      exit 1
      ;;
  esac
}

RETURNS_EXTRA=()
if [[ "${USE_RETURNS}" == "1" ]]; then
  RETURNS_EXTRA=(--returns_condition --include_returns)
fi

CPU_EXTRA=()
if [[ -n "${CPU_THREADS}" ]]; then
  CPU_EXTRA=(--cpu_threads "$CPU_THREADS")
fi

PROXY_FILTER_FS_EXTRA=()
if [[ -n "${PROXY_FILTER:-}" ]]; then
  PROXY_FILTER_FS_EXTRA=(--proxy_filter "${PROXY_FILTER}")
fi

W_TEXT_EXTRA=(--condition_guidance_w_text "${CONDITION_GUIDANCE_W_TEXT}")

_run_zero_shot_one() {
  local t="$1"
  local zs_n="${NUM_SEEDS}"
  if ! [[ "$zs_n" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误: NUM_SEEDS 须为正整数，当前: ${zs_n}" >&2
    return 1
  fi
  local _ST_FRAC_SIG="${t}_frac${FRAC}_sigma${SIGMA}"
  local _ST_HYPER="${N_TRAJ}x${HORIZON}_k${K}_eps${EPS}"
  if [[ "${USE_RETURNS}" == "1" ]]; then
    _ST_HYPER="${_ST_HYPER}${RESULTS_SUFFIX:-_ret}"
  fi
  local _mt_tag="mt${PRETRAINED_MT_HEX}_pds${PRETRAINED_DIFFUSION_SEED}_zs"
  local base_dir="${REAL_TASK_RESULTS_ROOT}/zero_shot/${_ST_FRAC_SIG}/${_ST_HYPER}_${_mt_tag}"
  mkdir -p "$base_dir"
  local BATCH_STATE_FILE="$base_dir/.gtg_pipeline_batch"
  local br
  if [[ "${EVAL_ONLY}" == "1" ]]; then
    if [[ -n "${BATCH_RUN:-}" ]]; then
      br="${BATCH_RUN}"
    elif [[ -f "$BATCH_STATE_FILE" ]]; then
      br=$(cat "$BATCH_STATE_FILE")
    else
      br=$(_max_batch_from_dirs "$base_dir")
      [[ -z "$br" || "$br" == "0" ]] && br=1
    fi
    echo "[批次] Zero-shot EVAL_ONLY=1 → run${br}_seed*（共 ${zs_n} 个 seed）"
  else
    local prev=0
    if [[ -f "$BATCH_STATE_FILE" ]]; then
      prev=$(cat "$BATCH_STATE_FILE")
    else
      prev=$(_max_batch_from_dirs "$base_dir")
    fi
    br=$((prev + 1))
    echo "[批次] Zero-shot BATCH_RUN=$br  |  seed 从 ${START_SEED} 起共 ${zs_n} 次（NUM_SEEDS + START_SEED，同 sweep）"
  fi

  echo "=========================================="
  echo "[Zero-shot] TASK=$t  PRETRAINED_MT_HEX=$PRETRAINED_MT_HEX  n_traj=$N_TRAJ k=$K eps=$EPS"
  echo "  condition_guidance_w_text=${CONDITION_GUIDANCE_W_TEXT}"
  echo "  根目录: $base_dir"
  echo "=========================================="

  echo "[Zero-shot] 真实任务：跳过轨迹 pkl / construct（采样固定无上下文；需本地 generated_datasets/.../vae_info.p 等供解码）"

  local zi
  local _ok=0
  for ((zi = 0; zi < zs_n; zi++)); do
    local seed=$((START_SEED + zi))
    local run_dir="$base_dir/run${br}_seed${seed}"
    mkdir -p "$run_dir"
    echo "--- Zero-shot seed=${seed} ($((zi + 1))/${zs_n}) → $run_dir ---"
    if "$PYTHON" evaluate.py \
      --train_tasks "$t" \
      --real_task_zero_shot_eval \
      --pretrained_mt_hex "$PRETRAINED_MT_HEX" \
      --pretrained_multitask_train_tasks "$PRETRAINED_MULTITASK_TRAIN_TASKS" \
      --pretrained_diffusion_seed "$PRETRAINED_DIFFUSION_SEED" \
      --n_traj "$N_TRAJ" \
      --k "$K" \
      --eps "$EPS" \
      --horizon "$HORIZON" \
      --frac "$FRAC" \
      --sigma "$SIGMA" \
      --seed "$seed" \
      --latent_dim "$LATENT_DIM" \
      "${RETURNS_EXTRA[@]}" \
      "${CPU_EXTRA[@]}" \
      "${W_TEXT_EXTRA[@]}" \
      --proxy_filter 0 \
      >"$run_dir/evaluate.log" 2>&1; then
      _ok=$((_ok + 1))
      echo "  完成: $run_dir/evaluate.log"
    else
      echo "Zero-shot 评估失败: $run_dir/evaluate.log" >&2
    fi
  done
  if [[ "${EVAL_ONLY}" != "1" ]] && [[ "$_ok" -gt 0 ]]; then
    echo "$br" >"$BATCH_STATE_FILE"
  fi
  if [[ "$_ok" -ne "$zs_n" ]]; then
    echo "警告: Zero-shot 本批成功 ${_ok}/${zs_n}" >&2
    return 1
  fi
  return 0
}

_run_few_shot_one() {
  local t="$1"
  local start_seed="${START_SEED}"

  if [[ ! -f "$PRETRAINED_VAE_INFO" ]]; then
    echo "错误: 文件不存在: $PRETRAINED_VAE_INFO" >&2
    exit 1
  fi

  if [[ -n "${FEWSHOT_K:-}" ]]; then
    _FS_TAG="k${FEWSHOT_K}_${FEWSHOT_MODE}"
  else
    _FS_TAG="all_${FEWSHOT_MODE}"
  fi

  _ST_FRAC_SIG="${t}_frac${FRAC}_sigma${SIGMA}"
  _ST_HYPER="${N_TRAJ}x${HORIZON}_k${K}_eps${EPS}"
  if [[ "${USE_RETURNS}" == "1" ]]; then
    _ST_HYPER="${_ST_HYPER}${RESULTS_SUFFIX:-_ret}"
  fi
  _ST_HYPER="${_ST_HYPER}${_DIFF_TRAIN_SUF:-}"
  _ST_HYPER="${_ST_HYPER}_mt${PRETRAINED_MT_HEX}_ft"

  # RESULTS=完整路径 时覆盖本任务的 base_dir（单任务重跑）；否则与 single_task 类似写入 real_task/few_shot
  base_dir="${RESULTS:-${REAL_TASK_RESULTS_ROOT}/few_shot/${_ST_FRAC_SIG}/fs_${_FS_TAG}/${_ST_HYPER}}"
  mkdir -p "$base_dir"
  BATCH_STATE_FILE="$base_dir/.gtg_pipeline_batch"

  if [[ "${AUTO_CONTINUE}" == "1" ]]; then
    max_s=-1
    shopt -s nullglob
    for d in "$base_dir"/run*_seed*/; do
      [[ -d "$d" ]] || continue
      base=$(basename "$d")
      if [[ "$base" =~ _seed([0-9]+)$ ]]; then
        s="${BASH_REMATCH[1]}"
        if [[ "$s" =~ ^[0-9]+$ ]] && ((10#$s > max_s)); then
          max_s=$((10#$s))
        fi
      fi
    done
    shopt -u nullglob
    if [[ $max_s -ge 0 ]]; then
      start_seed=$((max_s + 1))
      echo "[AUTO_CONTINUE] 最大已有 seed=$max_s → START_SEED=$start_seed，NUM_SEEDS=$NUM_SEEDS"
    fi
  fi

  if [[ "${EVAL_ONLY}" == "1" ]]; then
    if [[ -n "${BATCH_RUN:-}" ]]; then
      :
    elif [[ -f "$BATCH_STATE_FILE" ]]; then
      BATCH_RUN=$(cat "$BATCH_STATE_FILE")
    else
      BATCH_RUN=$(_max_batch_from_dirs "$base_dir")
      [[ -z "$BATCH_RUN" || "$BATCH_RUN" == "0" ]] && BATCH_RUN=1
    fi
    echo "[批次] EVAL_ONLY=1，使用 run${BATCH_RUN}_seed*"
  else
    prev=0
    if [[ -f "$BATCH_STATE_FILE" ]]; then
      prev=$(cat "$BATCH_STATE_FILE")
    else
      prev=$(_max_batch_from_dirs "$base_dir")
    fi
    BATCH_RUN=$((prev + 1))
    echo "[批次] BATCH_RUN=$BATCH_RUN"
  fi

  echo "=========================================="
  echo "[Few-shot] TASK=$t  PRETRAINED_MT_HEX=$PRETRAINED_MT_HEX  n_traj=$N_TRAJ k=$K eps=$EPS horizon=$HORIZON"
  echo "  TRAIN_EPOCHS=$TRAIN_EPOCHS"
  echo "  TRAIN_TIMESTEP_BIAS_POWER=$TRAIN_TIMESTEP_BIAS_POWER  TRAIN_LOSS_MIN_SNR_GAMMA=$TRAIN_LOSS_MIN_SNR_GAMMA"
  echo "  NUM_SEEDS=$NUM_SEEDS  START_SEED=$start_seed  （seed: $start_seed .. $((start_seed + NUM_SEEDS - 1))）"
  echo "  PRETRAINED_VAE_INFO=$PRETRAINED_VAE_INFO"
  echo "  condition_guidance_w_text=${CONDITION_GUIDANCE_W_TEXT}"
  echo "  RESULTS=$base_dir"
  echo "=========================================="

  CONSTRUCT_EXTRA=()
  if [[ -n "${FEWSHOT_K:-}" ]]; then
    CONSTRUCT_EXTRA+=(--fewshot_k "$FEWSHOT_K")
  fi
  CONSTRUCT_EXTRA+=(--fewshot_mode "$FEWSHOT_MODE")
  CONSTRUCT_EXTRA+=(--pretrained_vae_info "$PRETRAINED_VAE_INFO")
  CONSTRUCT_EXTRA+=(--finetune_epochs "$FINETUNE_EPOCHS")
  CONSTRUCT_EXTRA+=(--finetune_lr "$FINETUNE_LR")
  CONSTRUCT_EXTRA+=(--seed "$CONSTRUCT_SEED")

  if [[ "${EVAL_ONLY}" != "1" ]]; then
    echo "=== Step 1: construct_trajectories ==="
    construct_log="$base_dir/construct_trajectories.log"
    $PYTHON "$PROJECT/construct_trajectories.py" \
      --task "$t" \
      --frac "$FRAC" \
      --sigma "$SIGMA" \
      --n_traj "$N_TRAJ" \
      --k "$K" \
      --eps "$EPS" \
      --horizon "$HORIZON" \
      --latent_dim "$LATENT_DIM" \
      "${CONSTRUCT_EXTRA[@]}" \
      "${CPU_EXTRA[@]}" \
      >"$construct_log" 2>&1
    construct_status=$?
    if [[ $construct_status -ne 0 ]]; then
      echo "轨迹构建失败: $construct_log" >&2
      exit 1
    fi
    echo "日志: $construct_log"
  else
    echo "=== EVAL_ONLY=1：跳过 construct ==="
  fi

  _TE_EXTRA=(--train_epochs "$TRAIN_EPOCHS")

  echo
  echo "=== Step 2: train（real_task_text_only_finetune）+ evaluate ==="

  for ((run = 0; run < NUM_SEEDS; run++)); do
    seed=$((start_seed + run))
    run_dir="$base_dir/run${BATCH_RUN}_seed${seed}"
    mkdir -p "$run_dir"

    echo "--- run${BATCH_RUN}_seed${seed} ($((run + 1))/$NUM_SEEDS) ---"

    _fs_ckpt="$(_fewshot_trained_ckpt_dir "$t" "$seed")"
    _skip_train=0
    if [[ "${EVAL_ONLY}" != "1" ]] && [[ "${SKIP_TRAIN_IF_CKPT:-1}" == "1" ]] && [[ "${FORCE_FEWSHOT_TRAIN:-0}" != "1" ]]; then
      if [[ -f "${_fs_ckpt}/state.pt" ]] || compgen -G "${_fs_ckpt}/state_*.pt" > /dev/null 2>&1; then
        _skip_train=1
        echo "  [few-shot] 已有扩散 checkpoint，跳过训练: ${_fs_ckpt}"
      fi
    fi

    if [[ "${EVAL_ONLY}" != "1" ]] && [[ "$_skip_train" -eq 0 ]]; then
      $PYTHON "$PROJECT/train.py" \
        --train_tasks "$t" \
        --real_task_text_only_finetune \
        --pretrained_mt_hex "$PRETRAINED_MT_HEX" \
        --pretrained_multitask_train_tasks "$PRETRAINED_MULTITASK_TRAIN_TASKS" \
        --pretrained_diffusion_seed "$PRETRAINED_DIFFUSION_SEED" \
        --n_traj "$N_TRAJ" \
        --k "$K" \
        --eps "$EPS" \
        --seed "$seed" \
        --horizon "$HORIZON" \
        --frac "$FRAC" \
        --sigma "$SIGMA" \
        --latent_dim "$LATENT_DIM" \
        --train_timestep_bias_power "$TRAIN_TIMESTEP_BIAS_POWER" \
        --train_loss_min_snr_gamma "$TRAIN_LOSS_MIN_SNR_GAMMA" \
        "${RETURNS_EXTRA[@]}" \
        "${_TE_EXTRA[@]}" \
        "${CPU_EXTRA[@]}" \
        "${W_TEXT_EXTRA[@]}" \
        "${PROXY_FILTER_FS_EXTRA[@]}" \
        >"$run_dir/train.log" 2>&1 || {
        echo "训练失败: $run_dir/train.log" >&2
        continue
      }
    elif [[ "${EVAL_ONLY}" == "1" ]]; then
      echo "  跳过训练（EVAL_ONLY=1）"
    elif [[ "$_skip_train" -eq 1 ]]; then
      echo "  跳过训练（SKIP_TRAIN_IF_CKPT，checkpoint 已存在）"
    fi

    $PYTHON "$PROJECT/evaluate.py" \
      --train_tasks "$t" \
      --real_task_text_only_finetune \
      --pretrained_multitask_train_tasks "$PRETRAINED_MULTITASK_TRAIN_TASKS" \
      --n_traj "$N_TRAJ" \
      --k "$K" \
      --eps "$EPS" \
      --seed "$seed" \
      --horizon "$HORIZON" \
      --frac "$FRAC" \
      --sigma "$SIGMA" \
      --latent_dim "$LATENT_DIM" \
      --train_timestep_bias_power "$TRAIN_TIMESTEP_BIAS_POWER" \
      --train_loss_min_snr_gamma "$TRAIN_LOSS_MIN_SNR_GAMMA" \
      "${RETURNS_EXTRA[@]}" \
      "${CPU_EXTRA[@]}" \
      "${W_TEXT_EXTRA[@]}" \
      "${PROXY_FILTER_FS_EXTRA[@]}" \
      >"$run_dir/evaluate.log" 2>&1 || {
      echo "评估失败: $run_dir/evaluate.log" >&2
      continue
    }

    echo "  完成: $run_dir"
  done

  if [[ "${EVAL_ONLY}" != "1" ]]; then
    echo "$BATCH_RUN" >"$BATCH_STATE_FILE"
  fi

  echo "Few-shot 结果目录: $base_dir/"
}

mapfile -t _TASK_LIST < <(_tasks_to_run)

case "$MODE" in
  all | both)
    echo ">>> 阶段 A：Zero-shot"
    for t in "${_TASK_LIST[@]}"; do
      _run_zero_shot_one "$t"
    done
    echo ""
    echo ">>> 阶段 B：Few-shot"
    _resolve_pretrained_vae_info_if_needed
    for t in "${_TASK_LIST[@]}"; do
      _run_few_shot_one "$t"
    done
    ;;
  zero_shot | zs)
    for t in "${_TASK_LIST[@]}"; do
      _run_zero_shot_one "$t"
    done
    ;;
  few_shot | fs | few-shot)
    _resolve_pretrained_vae_info_if_needed
    for t in "${_TASK_LIST[@]}"; do
      _run_few_shot_one "$t"
    done
    ;;
  -h | --help)
    sed -n '1,80p' "$0"
    exit 0
    ;;
  *)
    echo "用法: $0 [选项] [all|zero_shot|few_shot] [all|lunar_lander|robot_push|rover]" >&2
    exit 1
    ;;
esac

echo ""
echo "=== 全部完成 ==="

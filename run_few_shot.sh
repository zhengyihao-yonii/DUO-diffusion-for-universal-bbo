#!/usr/bin/env bash
#
# =============================================================================
# GTGdfgo：真实任务 few-shot 全流程（单任务）
# =============================================================================
#
# 顺序：construct_trajectories（含 few-shot 数据 + 预训练 VAE 微调）→ train.py → evaluate.py
#
# 用法 A（推荐：环境变量）:
#   export PRETRAINED_VAE_INFO=/path/to/multi_xxx/vae_info.p   # 必填：仅 Design-Bench 等多任务预训练产物
#   export TASK=lunar_lander                                  # 三选一: lunar_lander | robot_push | rover
#   # 按需改 FEWSHOT_K / FEWSHOT_MODE / 轨迹与训练超参，然后：
#   bash run_few_shot.sh
#
# 用法 B（第一个参数作为预训练 vae_info.p）:
#   bash run_few_shot.sh /path/to/multi_xxx/vae_info.p
#
# 可选环境变量见下方「参数说明」。
#
# =============================================================================
# 参数说明（环境变量，均有默认值或条件默认）
# =============================================================================
#
# --- 必填 / 核心路径 ---
# PRETRAINED_VAE_INFO   多任务预训练得到的 vae_info.p（例如 generated_datasets/multi_dkitty_ant_.../vae_info.p）。
#                       用于在 few-shot 真实数据上微调 VAE（_rwft 目录），不可省略。
# TASK                  真实任务短名：lunar_lander | robot_push | rover（默认 lunar_lander）。
#
# --- Few-shot 子集（仅 real-world JSON 数据）---
# FEWSHOT_K             取多少个点；留空或不 export 表示「全量」点（等价 CLI 不传 --fewshot_k）。
# FEWSHOT_MODE          all | random | worst（默认 random）。
#                       - all：不子采样；random：随机 K 个；worst：y 最小的 K 个（假设 y 越大越好）。
# CONSTRUCT_SEED        构造轨迹与 few-shot 采样用的随机种子（默认 0）。与下面训练用的 START_SEED 可分开。
#
# --- VAE 微调（construct 内调 train_vae）---
# FINETUNE_EPOCHS       微调轮数（默认 50）。
# FINETUNE_LR           微调学习率（默认 3e-5）。
#
# --- 与标准单任务流水线一致的轨迹 / 数据比例 ---
# FRAC                  数据比例，与仓库其它脚本一致（默认 1.0）。
# SIGMA                 噪声（默认 0.0）。
# N_TRAJ                合成轨迹条数（默认 2000）。
# K                     轨迹每步近邻候选数（默认 50）。
# EPS                   轨迹价值容忍（默认 0.05）。
# HORIZON               轨迹长度，需与扩散 horizon 一致（默认 64）。
#
# --- 扩散训练与重复实验 ---
# NUM_RUNS              同一超参下重复次数（不同 seed）（默认 1）。
# START_SEED            第一次训练使用的 seed（默认 0）；第 i 次为 START_SEED+i。
# TRAIN_EPOCHS          传给 train.py --train_epochs（默认 200）。
#
# --- 可选开关（与 run_singletask.sh 对齐）---
# EVAL_ONLY             设为 1 时跳过 construct 与 train，只对已有 run*_seed* 重写 evaluate.log。
# USE_RETURNS           设为 1 时 train/evaluate 增加 --returns_condition --include_returns。
# AUTO_CONTINUE         设为 1 时根据已有 run*_seed* 自动把 START_SEED 设为 max_seed+1。
# CPU_THREADS           传给 construct / train / evaluate 的 --cpu_threads（限制 BLAS 等线程数）。
#
# --- 路径 ---
# PYTHON                Python 可执行文件（默认 /home/xk/anaconda3/envs/gtg/bin/python）。
# PROJECT               GTGdfgo 根目录（默认本脚本所在目录）。
# RESULTS               结果根目录；默认
#                       $PROJECT/results/few_shot/<TASK>_frac<F>_sigma<S>/fs_<tag>/<N_TRAJ>x<H>_k<K>_eps<E>/
#                       其中 fs_<tag>：若设了 FEWSHOT_K 则为 k<K>_<MODE>，否则为 all_<MODE>。
#
# GPU：可 export CUDA_VISIBLE_DEVICES=0 或 source scripts/gpu_env.sh 使用 GPU_ID。
#
# =============================================================================

set -uo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/gpu_env.sh
source "$_SCRIPT_DIR/scripts/gpu_env.sh"

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

# ---------- 默认 Python / 工程根 ----------
PYTHON="${PYTHON:-/home/xk/anaconda3/envs/gtg/bin/python}"
PROJECT="${PROJECT:-$_SCRIPT_DIR}"

# ---------- 第一个参数：预训练 vae_info.p ----------
PRETRAINED_VAE_INFO="${1:-${PRETRAINED_VAE_INFO:-}}"
if [[ -z "$PRETRAINED_VAE_INFO" ]]; then
  echo "错误: 必须设置 PRETRAINED_VAE_INFO（多任务预训练 vae_info.p），或通过第一个参数传入。" >&2
  echo "示例: PRETRAINED_VAE_INFO=/path/to/multi_.../vae_info.p bash $0" >&2
  echo "或:   bash $0 /path/to/multi_.../vae_info.p" >&2
  exit 1
fi
if [[ ! -f "$PRETRAINED_VAE_INFO" ]]; then
  echo "错误: 文件不存在: $PRETRAINED_VAE_INFO" >&2
  exit 1
fi

# ---------- 任务与 few-shot ----------
TASK="${TASK:-lunar_lander}"
FEWSHOT_MODE="${FEWSHOT_MODE:-random}"
CONSTRUCT_SEED="${CONSTRUCT_SEED:-0}"

# ---------- 轨迹与数据 ----------
FRAC="${FRAC:-1.0}"
SIGMA="${SIGMA:-0.0}"
N_TRAJ="${N_TRAJ:-2000}"
K="${K:-50}"
EPS="${EPS:-0.05}"
HORIZON="${HORIZON:-64}"

# ---------- VAE 微调 ----------
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-50}"
FINETUNE_LR="${FINETUNE_LR:-3e-5}"

# ---------- 训练重复 ----------
NUM_RUNS="${NUM_RUNS:-1}"
START_SEED="${START_SEED:-0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-200}"

# ---------- 结果目录 tag ----------
if [[ -n "${FEWSHOT_K:-}" ]]; then
  _FS_TAG="k${FEWSHOT_K}_${FEWSHOT_MODE}"
else
  _FS_TAG="all_${FEWSHOT_MODE}"
fi

_ST_FRAC_SIG="${TASK}_frac${FRAC}_sigma${SIGMA}"
_ST_HYPER="${N_TRAJ}x${HORIZON}_k${K}_eps${EPS}"
if [[ "${USE_RETURNS:-0}" == "1" ]]; then
  _ST_HYPER="${_ST_HYPER}${RESULTS_SUFFIX:-_ret}"
fi

base_dir="${RESULTS:-$PROJECT/results/few_shot/${_ST_FRAC_SIG}/fs_${_FS_TAG}/${_ST_HYPER}}"
mkdir -p "$base_dir"
BATCH_STATE_FILE="$base_dir/.gtg_pipeline_batch"

if [[ "${AUTO_CONTINUE:-0}" == "1" ]]; then
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
    START_SEED=$((max_s + 1))
    echo "[AUTO_CONTINUE] 最大已有 seed=$max_s → 本批 START_SEED=$START_SEED，共 NUM_RUNS=$NUM_RUNS 次"
  fi
fi

if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
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

echo "=== GTGdfgo few-shot real-world ==="
echo "  TASK=$TASK  PRETRAINED_VAE_INFO=$PRETRAINED_VAE_INFO"
echo "  FEWSHOT_K=${FEWSHOT_K:-"(全量)"}  FEWSHOT_MODE=$FEWSHOT_MODE  CONSTRUCT_SEED=$CONSTRUCT_SEED"
echo "  FINETUNE_EPOCHS=$FINETUNE_EPOCHS  FINETUNE_LR=$FINETUNE_LR"
echo "  n_traj=$N_TRAJ k=$K eps=$EPS horizon=$HORIZON frac=$FRAC sigma=$SIGMA"
echo "  NUM_RUNS=$NUM_RUNS START_SEED=$START_SEED TRAIN_EPOCHS=$TRAIN_EPOCHS"
echo "  RESULTS=$base_dir"
echo "  EVAL_ONLY=${EVAL_ONLY:-0}"
echo

cd "$PROJECT"

CONSTRUCT_EXTRA=()
if [[ -n "${FEWSHOT_K:-}" ]]; then
  CONSTRUCT_EXTRA+=(--fewshot_k "$FEWSHOT_K")
fi
CONSTRUCT_EXTRA+=(--fewshot_mode "$FEWSHOT_MODE")
CONSTRUCT_EXTRA+=(--pretrained_vae_info "$PRETRAINED_VAE_INFO")
CONSTRUCT_EXTRA+=(--finetune_epochs "$FINETUNE_EPOCHS")
CONSTRUCT_EXTRA+=(--finetune_lr "$FINETUNE_LR")
CONSTRUCT_EXTRA+=(--seed "$CONSTRUCT_SEED")

CPU_EXTRA=()
if [[ -n "${CPU_THREADS:-}" ]]; then
  CPU_EXTRA+=(--cpu_threads "$CPU_THREADS")
fi

if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
  echo "=== EVAL_ONLY=1：跳过 construct 与 train ==="
else
  echo "=== Step 1: construct_trajectories（few-shot + VAE 微调）==="
  construct_log="$base_dir/construct_trajectories.log"
  # shellcheck disable=SC2086
  $PYTHON "$PROJECT/construct_trajectories.py" \
    --task "$TASK" \
    --frac "$FRAC" \
    --sigma "$SIGMA" \
    --n_traj "$N_TRAJ" \
    --k "$K" \
    --eps "$EPS" \
    --horizon "$HORIZON" \
    "${CONSTRUCT_EXTRA[@]}" \
    "${CPU_EXTRA[@]}" \
    >"$construct_log" 2>&1
  construct_status=$?
  if [[ $construct_status -ne 0 ]]; then
    echo "轨迹构建失败: $construct_log" >&2
    exit 1
  fi
  echo "日志: $construct_log"
fi

RETURNS_EXTRA=()
if [[ "${USE_RETURNS:-0}" == "1" ]]; then
  RETURNS_EXTRA=(--returns_condition --include_returns)
fi

_TE_EXTRA=(--train_epochs "$TRAIN_EPOCHS")

echo
echo "=== Step 2: 训练 + 评估（$NUM_RUNS 次）==="

for ((run = 0; run < NUM_RUNS; run++)); do
  seed=$((START_SEED + run))
  run_dir="$base_dir/run${BATCH_RUN}_seed${seed}"
  mkdir -p "$run_dir"

  echo "--- run${BATCH_RUN}_seed${seed} ($((run + 1))/$NUM_RUNS) ---"

  if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
    $PYTHON "$PROJECT/train.py" \
      --task "$TASK" \
      --n_traj "$N_TRAJ" \
      --k "$K" \
      --eps "$EPS" \
      --seed "$seed" \
      --horizon "$HORIZON" \
      --frac "$FRAC" \
      --sigma "$SIGMA" \
      "${RETURNS_EXTRA[@]}" \
      "${_TE_EXTRA[@]}" \
      "${CPU_EXTRA[@]}" \
      >"$run_dir/train.log" 2>&1 || { echo "训练失败: $run_dir/train.log"; continue; }
  else
    echo "  跳过训练（EVAL_ONLY=1）"
  fi

  $PYTHON "$PROJECT/evaluate.py" \
    --task "$TASK" \
    --n_traj "$N_TRAJ" \
    --k "$K" \
    --eps "$EPS" \
    --seed "$seed" \
    --horizon "$HORIZON" \
    --frac "$FRAC" \
    --sigma "$SIGMA" \
    "${RETURNS_EXTRA[@]}" \
    "${CPU_EXTRA[@]}" \
    >"$run_dir/evaluate.log" 2>&1 || { echo "评估失败: $run_dir/evaluate.log"; continue; }

  echo "  完成: $run_dir"
done

if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  echo "$BATCH_RUN" >"$BATCH_STATE_FILE"
fi

echo
echo "全部结束。结果: $base_dir/"

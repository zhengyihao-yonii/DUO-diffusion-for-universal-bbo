#!/usr/bin/env bash
#
# GTGdfgo 单任务：轨迹（含 VAE）只构建一次，再按不同 seed 多次 train.py + evaluate.py。
# 与 run_multitask.sh 的区别：仅单个任务名，参数形式为「单任务专用」。
#
# 无需单独先跑 train_vae.py：construct_trajectories.py 会调用 train_vae.main。
#
# 用法:
#   bash run_multiple_times.sh <task_name> <num_runs> <n_traj> <k> <eps> [horizon] [frac] [sigma]
#
# 示例:
#   bash run_multiple_times.sh dkitty 3 4000 20 0.01
#   bash run_multiple_times.sh dkitty 3 4000 20 0.01 64 1.0 0.0
#
# 环境变量（可选）:
#   PYTHON         Python，默认 /home/xk/anaconda3/envs/gtg/bin/python
#   PROJECT        GTGdfgo 根目录，默认本脚本所在目录
#   RESULTS        结果目录，默认 PROJECT/results/${task}_multiple_runs
#                  USE_RETURNS=1 时默认在目录名后追加 RESULTS_SUFFIX（默认 _retcond），与无 returns 结果分开
#   RESULTS_SUFFIX 仅 USE_RETURNS=1 且未显式设置 RESULTS 时生效；设为空可取消默认后缀
#   START_SEED     本批第一次使用的随机种子，默认 0
#   AUTO_CONTINUE  默认 0。设为 1 时在 RESULTS 下扫描 run*_seed*，从 max_seed+1 起跑本批
#   EVAL_ONLY        默认 0。设为 1 时跳过轨迹构建与 train，只对已有 run*_seed* 目录重写 evaluate.log
#                    （需与之前成功训练时使用相同的 task/num_runs/START_SEED/参数，以便找到 checkpoint）
#   BATCH_RUN        可选，仅 EVAL_ONLY=1 时生效：指定要重评的批次号 N（runN_seed*）。不设则使用
#                    $RESULTS/.gtg_pipeline_batch 中记录的「上一轮完整流水线」批次号。
#   USE_RETURNS  设为 1 时，train/evaluate 追加 --returns_condition --include_returns（显式标量 return 条件）
#   单任务仅基础 VAE + 轨迹 + 扩散，无文本条件；结果目录默认 results/<task>_multiple_runs[_retcond]。
#   CUDA_VISIBLE_DEVICES / GPU_ID  见脚本中部「GPU」注释与 scripts/gpu_env.sh。
#   CPU_THREADS      可选，限制 OpenMP/BLAS/PyTorch CPU 线程数；等价于 train/evaluate/construct 的 --cpu_threads N。
#
# 目录命名：同一次 bash 内为 run{N}_seed{s},{s+1},...（仅 seed 变）；N 仅在「又一次完整 bash（非 EVAL_ONLY）」
#          结束后 +1，并写入 .gtg_pipeline_batch。无该文件时会根据已有 run*_seed* 推断最大批次。
#
# GPU（多卡 / 避免 OOM）:
#   - 推荐: 运行前 export CUDA_VISIBLE_DEVICES=1
#   - 或: GPU_ID=2 bash run_singletask.sh ...
#   - 或: 取消注释下一行:
# export CUDA_VISIBLE_DEVICES=0

set -uo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/gpu_env.sh
source "$_SCRIPT_DIR/scripts/gpu_env.sh"

# 扫描 RESULTS 下 run 前缀的最大批次号（run123_seed456 -> 123）
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

if [[ $# -lt 5 ]]; then
  echo "用法: bash run_multiple_times.sh <task_name> <num_runs> <n_traj> <k> <eps> [horizon] [frac] [sigma]"
  echo "示例: bash run_multiple_times.sh dkitty 3 4000 20 0.01"
  echo ""
  echo "环境变量 START_SEED、AUTO_CONTINUE、EVAL_ONLY 见脚本头部注释。"
  exit 1
fi

task_name="$1"
num_runs="$2"
n_traj="$3"
k="$4"
eps="$5"
HORIZON="${6:-64}"
FRAC="${7:-1.0}"
SIGMA="${8:-0.0}"

PYTHON="${PYTHON:-/home/xk/anaconda3/envs/gtg/bin/python}"
PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
_rs=""
if [[ "${USE_RETURNS:-0}" == "1" ]]; then
  _rs="${RESULTS_SUFFIX:-_retcond}"
fi
base_dir="${RESULTS:-$PROJECT/results/${task_name}_multiple_runs${_rs}}"
mkdir -p "$base_dir"
BATCH_STATE_FILE="$base_dir/.gtg_pipeline_batch"

START_SEED="${START_SEED:-0}"
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
    echo "[AUTO_CONTINUE] 已有目录中最大 seed=$max_s，本批从 START_SEED=$START_SEED 起共 $num_runs 次"
  else
    echo "[AUTO_CONTINUE] 未找到已有 run*_seed* 目录，本批从 START_SEED=$START_SEED 开始"
  fi
fi

# 批次号 N：一次完整流水线共用一个 runN_*；EVAL_ONLY 时复用已记录批次（或 BATCH_RUN）
if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
  if [[ -n "${BATCH_RUN:-}" ]]; then
    :
  elif [[ -f "$BATCH_STATE_FILE" ]]; then
    BATCH_RUN=$(cat "$BATCH_STATE_FILE")
  else
    BATCH_RUN=$(_max_batch_from_dirs "$base_dir")
    [[ -z "$BATCH_RUN" || "$BATCH_RUN" == "0" ]] && BATCH_RUN=1
  fi
  echo "[批次] EVAL_ONLY=1，使用 run${BATCH_RUN}_seed*（可设置 BATCH_RUN 覆盖）"
else
  prev=0
  if [[ -f "$BATCH_STATE_FILE" ]]; then
    prev=$(cat "$BATCH_STATE_FILE")
  else
    prev=$(_max_batch_from_dirs "$base_dir")
  fi
  BATCH_RUN=$((prev + 1))
  echo "[批次] 本流水线批次号 BATCH_RUN=$BATCH_RUN（目录 run${BATCH_RUN}_seed<seed>）"
fi

echo "=== GTGdfgo 单任务: $task_name | num_runs=$num_runs | start_seed=$START_SEED (seed: $START_SEED .. $((START_SEED + num_runs - 1))) ==="
echo "  n_traj=$n_traj k=$k eps=$eps horizon=$HORIZON frac=$FRAC sigma=$SIGMA"
echo "  EVAL_ONLY=${EVAL_ONLY:-0}  BATCH_RUN=${BATCH_RUN}"
echo "  PROJECT=$PROJECT"
echo "  RESULTS=$base_dir"
echo

cd "$PROJECT"

if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
  echo "=== EVAL_ONLY=1：跳过轨迹构建与训练，仅重写各 run 目录下的 evaluate.log ==="
else
  echo "=== Step 1: 合成轨迹（内含 VAE 训练/加载）==="
  construct_log="$base_dir/construct_trajectories.log"
  $PYTHON "$PROJECT/construct_trajectories.py" \
    --task "$task_name" \
    --frac "$FRAC" \
    --sigma "$SIGMA" \
    --n_traj "$n_traj" \
    --k "$k" \
    --eps "$eps" \
    --horizon "$HORIZON" \
    >"$construct_log" 2>&1
  construct_status=$?
  if [[ $construct_status -ne 0 ]]; then
    echo "轨迹构建失败，查看日志: $construct_log"
    exit 1
  fi
  echo "日志: $construct_log"
fi

echo
# 避免在 echo "..." 内嵌套 $([ "..." == "1" ])，部分 bash 会错误匹配引号导致文末 EOF 报错
_step2_mode="训练 + 评估"
if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
  _step2_mode="仅评估"
fi
echo "=== Step 2: 多次${_step2_mode}（本批 ${num_runs} 次）==="

RETURNS_EXTRA=()
if [[ "${USE_RETURNS:-0}" == "1" ]]; then
  RETURNS_EXTRA=(--returns_condition --include_returns)
  echo "[USE_RETURNS] 启用 returns_condition + include_returns"
fi

for ((run = 0; run < num_runs; run++)); do
  seed=$((START_SEED + run))
  run_dir="$base_dir/run${BATCH_RUN}_seed${seed}"
  mkdir -p "$run_dir"

  echo "--- 本批第 $((run + 1))/$num_runs  (run${BATCH_RUN}_seed${seed}) ---"

  if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
    echo "  训练..."
    $PYTHON "$PROJECT/train.py" \
      --task "$task_name" \
      --n_traj "$n_traj" \
      --k "$k" \
      --eps "$eps" \
      --seed "$seed" \
      --horizon "$HORIZON" \
      --frac "$FRAC" \
      --sigma "$SIGMA" \
      "${RETURNS_EXTRA[@]}" \
      >"$run_dir/train.log" 2>&1 || { echo "训练失败: $run_dir/train.log"; continue; }
  else
    echo "  跳过训练（EVAL_ONLY=1），覆盖 evaluate.log"
  fi

  echo "  评估..."
  $PYTHON "$PROJECT/evaluate.py" \
    --task "$task_name" \
    --n_traj "$n_traj" \
    --k "$k" \
    --eps "$eps" \
    --seed "$seed" \
    --horizon "$HORIZON" \
    --frac "$FRAC" \
    --sigma "$SIGMA" \
    "${RETURNS_EXTRA[@]}" \
    >"$run_dir/evaluate.log" 2>&1 || { echo "评估失败: $run_dir/evaluate.log"; continue; }

  echo "  完成: $run_dir"
done

if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  echo "$BATCH_RUN" >"$BATCH_STATE_FILE"
  _next_batch=$((BATCH_RUN + 1))
  echo "[批次] 已写入 ${BATCH_STATE_FILE} -> ${BATCH_RUN}（下次完整流水线将使用 run${_next_batch}_*）"
fi

echo
echo "全部完成。结果目录: ${base_dir}/"

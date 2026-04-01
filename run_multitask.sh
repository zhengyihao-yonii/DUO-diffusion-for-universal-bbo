#!/usr/bin/env bash
#
# 多任务 GTGdfgo 流水线：一次性构建 VAE+合成轨迹，再按种子多次训练扩散并可选评估。
#
# 无需单独先跑 train_vae.py：construct_trajectories.py 内部会调用 train_vae.main，
# 按 multi_{tasks}_..._dim{fixed_dim} 目录训练或加载统一 VAE，再生成混合轨迹。
# 若你只想预训练 VAE、暂不建轨迹，可手动: python train_vae.py --tasks "a,b" --frac ... --sigma ...
#
# 用法:
#   bash run_multitask.sh <train_tasks> <num_runs> <n_traj> <k> <eps> [eval_task] [horizon] [frac] [sigma]
#
# 示例:
#   bash run_multitask.sh "dkitty,ant" 3 1000 50 0.05 dkitty 64 1.0 0.0
#
# 环境变量（可选）:
#   PYTHON         Python 解释器，默认 /home/xk/anaconda3/envs/gtg/bin/python
#   PROJECT        项目根目录（GTGdfgo），默认本脚本所在目录
#   RESULTS        日志与汇总输出目录
#   EVAL_ALL       设为 0 时，evaluate 只评 EVAL_TASK 一个任务（传 --eval_only_first）。
#                  默认（未设置或非 0）下，多任务会自动评估 train_tasks 中全部任务（见 evaluate.py）。
#   START_SEED     本批第一次运行使用的随机种子，默认 0。设为 3 且 NUM_RUNS=2 则使用 seed 3、4。
#   AUTO_CONTINUE  默认 0。仅当设为 1 时，在 RESULTS 下扫描已有 run*_seed<数字> 目录取最大 seed，
#                  本批从 max_seed+1 开始（并覆盖 START_SEED）；为 0 或未设置时始终用 START_SEED。
#   EVAL_ONLY        默认 0。设为 1 时跳过轨迹构建与 train，只对已有 run 目录重写 evaluate.log。
#   BATCH_RUN        可选，仅 EVAL_ONLY=1：指定批次号 N；否则读 $RESULTS/.gtg_pipeline_batch。
#
# 同一次 bash：run{N}_seed* 仅 seed 递增；仅「完整流水线」结束后 N 写入 .gtg_pipeline_batch 并在下次 +1。

set -euo pipefail

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
  echo "用法: bash run_multitask.sh <train_tasks> <num_runs> <n_traj> <k> <eps> [eval_task] [horizon] [frac] [sigma]"
  echo "示例: bash run_multitask.sh \"dkitty,ant\" 3 1000 50 0.05 dkitty 64 1.0 0.0"
  echo ""
  echo "环境变量 START_SEED（默认0）、AUTO_CONTINUE（默认0；设为1则从已有最大 seed 后继续）见脚本头部注释。"
  exit 1
fi

TRAIN_TASKS="$1"
NUM_RUNS="$2"
N_TRAJ="$3"
K="$4"
EPS="$5"
EVAL_TASK="${6:-$(echo "$TRAIN_TASKS" | cut -d, -f1 | tr -d ' ')}"
HORIZON="${7:-64}"
FRAC="${8:-1.0}"
SIGMA="${9:-0.0}"

PYTHON="${PYTHON:-/home/xk/anaconda3/envs/gtg/bin/python}"
PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RESULTS="${RESULTS:-$PROJECT/results/multitask_$(echo "$TRAIN_TASKS" | tr ',' '_')}"

mkdir -p "$RESULTS"
BATCH_STATE_FILE="$RESULTS/.gtg_pipeline_batch"

# 起始 seed：默认 0；AUTO_CONTINUE=1 时根据已有 run 目录推断
START_SEED="${START_SEED:-0}"
if [[ "${AUTO_CONTINUE:-0}" == "1" ]]; then
  max_s=-1
  shopt -s nullglob
  for d in "$RESULTS"/run*_seed*/; do
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
    echo "[AUTO_CONTINUE] 已有目录中最大 seed=$max_s，本批从 START_SEED=$START_SEED 起共 $NUM_RUNS 次"
  else
    echo "[AUTO_CONTINUE] 未找到已有 run*_seed* 目录，本批从 START_SEED=$START_SEED 开始"
  fi
fi

if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
  if [[ -n "${BATCH_RUN:-}" ]]; then
    :
  elif [[ -f "$BATCH_STATE_FILE" ]]; then
    BATCH_RUN=$(cat "$BATCH_STATE_FILE")
  else
    BATCH_RUN=$(_max_batch_from_dirs "$RESULTS")
    [[ -z "$BATCH_RUN" || "$BATCH_RUN" == "0" ]] && BATCH_RUN=1
  fi
  echo "[批次] EVAL_ONLY=1，使用 run${BATCH_RUN}_seed*（可设置 BATCH_RUN 覆盖）"
else
  prev=0
  if [[ -f "$BATCH_STATE_FILE" ]]; then
    prev=$(cat "$BATCH_STATE_FILE")
  else
    prev=$(_max_batch_from_dirs "$RESULTS")
  fi
  BATCH_RUN=$((prev + 1))
  echo "[批次] 本流水线批次号 BATCH_RUN=$BATCH_RUN（目录 run${BATCH_RUN}_seed<seed>）"
fi

echo "=== 多任务配置 ==="
echo "  train_tasks: $TRAIN_TASKS"
echo "  eval_task (checkpoint 后缀): $EVAL_TASK"
echo "  num_runs: $NUM_RUNS  start_seed: $START_SEED  (本批 seed 范围: $START_SEED .. $((START_SEED + NUM_RUNS - 1)))"
echo "  n_traj: $N_TRAJ  k: $K  eps: $EPS  horizon: $HORIZON  frac: $FRAC  sigma: $SIGMA"
echo "  EVAL_ONLY=${EVAL_ONLY:-0}  BATCH_RUN=${BATCH_RUN}"
echo "  PROJECT: $PROJECT"
echo "  RESULTS: $RESULTS"
echo

cd "$PROJECT"

if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
  echo "=== EVAL_ONLY=1：跳过轨迹构建与训练，仅重写各 run 目录下的 evaluate.log ==="
else
  echo "=== Step 1: 合成轨迹（内含多任务 VAE 训练/加载）==="
  CONSTRUCT_LOG="$RESULTS/construct_trajectories.log"
  $PYTHON "$PROJECT/construct_trajectories.py" \
    --tasks "$TRAIN_TASKS" \
    --frac "$FRAC" \
    --sigma "$SIGMA" \
    --n_traj "$N_TRAJ" \
    --k "$K" \
    --eps "$EPS" \
    --horizon "$HORIZON" \
    >"$CONSTRUCT_LOG" 2>&1

  echo "日志: $CONSTRUCT_LOG"
fi

echo
echo "=== Step 2: 多次$([ "${EVAL_ONLY:-0}" == "1" ] && echo '仅评估' || echo '训练 + 评估')（本批 $NUM_RUNS 次，seed=$START_SEED..$((START_SEED + NUM_RUNS - 1))）==="
for ((run = 0; run < NUM_RUNS; run++)); do
  SEED=$((START_SEED + run))
  RUN_DIR="$RESULTS/run${BATCH_RUN}_seed${SEED}"
  mkdir -p "$RUN_DIR"

  echo "--- 本批第 $((run + 1))/$NUM_RUNS  (run${BATCH_RUN}_seed${SEED}) ---"

  if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
    echo "  训练扩散..."
    $PYTHON "$PROJECT/train.py" \
      --train_tasks "$TRAIN_TASKS" \
      --eval_task "$EVAL_TASK" \
      --n_traj "$N_TRAJ" \
      --k "$K" \
      --eps "$EPS" \
      --seed "$SEED" \
      --horizon "$HORIZON" \
      --frac "$FRAC" \
      --sigma "$SIGMA" \
      >"$RUN_DIR/train.log" 2>&1
  else
    echo "  跳过训练（EVAL_ONLY=1），覆盖 evaluate.log"
  fi

  echo "  评估..."
  EVAL_EXTRA=()
  if [[ "$TRAIN_TASKS" == *","* ]] && [[ "${EVAL_ALL:-1}" == "0" ]]; then
    EVAL_EXTRA+=(--eval_only_first)
  fi
  $PYTHON "$PROJECT/evaluate.py" \
    --train_tasks "$TRAIN_TASKS" \
    --eval_task "$EVAL_TASK" \
    --checkpoint_eval_task "$EVAL_TASK" \
    --n_traj "$N_TRAJ" \
    --k "$K" \
    --eps "$EPS" \
    --seed "$SEED" \
    --horizon "$HORIZON" \
    --frac "$FRAC" \
    --sigma "$SIGMA" \
    "${EVAL_EXTRA[@]}" \
    >"$RUN_DIR/evaluate.log" 2>&1

  echo "  完成: $RUN_DIR"
done

if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  echo "$BATCH_RUN" >"$BATCH_STATE_FILE"
  echo "[批次] 已写入 $BATCH_STATE_FILE -> $BATCH_RUN（下次完整流水线将使用 run$((BATCH_RUN + 1))_*）"
fi

echo
echo "全部完成。结果目录: $RESULTS"

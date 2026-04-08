#!/usr/bin/env bash
#
# 多任务 GTGdfgo 流水线：一次性构建 VAE+合成轨迹，再按种子多次训练扩散并可选评估。
#
# 无需单独先跑 train_vae.py：construct_trajectories.py 内部会调用 train_vae.main，
# 按 multi_{tasks}_..._dim{fixed_dim} 目录训练或加载统一 VAE，再生成混合轨迹。
# 若你只想预训练 VAE、暂不建轨迹，可手动: python train_vae.py --tasks "a,b" --frac ... --sigma ...
#
# 用法:
#   bash run_multitask.sh <train_tasks> <num_runs> <n_traj> <k> <eps> [horizon] [frac] [sigma]
#
# 示例:
#   bash run_multitask.sh "dkitty,ant" 3 1000 50 0.05 64 1.0 0.0
# 任务名顺序仅影响展示；目录 multi_ant_dkitty_* 与 "ant,dkitty" / "dkitty,ant" 共用。
#
# 环境变量（可选）:
#   PYTHON         Python 解释器，默认 /home/xk/anaconda3/envs/gtg/bin/python
#   PROJECT        项目根目录（GTGdfgo），默认本脚本所在目录
#   RESULTS        日志与汇总输出目录
#   RESULTS_SUFFIX 仅 USE_RETURNS=1 且未显式设置 RESULTS 时追加到默认路径（默认 _retcond）
#   EVAL_ALL       设为 0 时，多任务只评一个任务（见 --eval_only_first）；默认评 train_tasks 全部。
#   EVAL_SINGLE_TASK  仅当 EVAL_ALL=0 时有效：要评的那一个任务名（默认 train_tasks 字典序第一个）。
#   START_SEED     本批第一次运行使用的随机种子，默认 0。设为 3 且 NUM_RUNS=2 则使用 seed 3、4。
#   AUTO_CONTINUE  默认 0。仅当设为 1 时，在 RESULTS 下扫描已有 run*_seed<数字> 目录取最大 seed，
#                  本批从 max_seed+1 开始（并覆盖 START_SEED）；为 0 或未设置时始终用 START_SEED。
#   EVAL_ONLY        默认 0。设为 1 时跳过轨迹构建与 train，只对已有 run 目录重写 evaluate.log。
#   BATCH_RUN        可选，仅 EVAL_ONLY=1：指定批次号 N；否则读 $RESULTS/.gtg_pipeline_batch。
#   TRAIN_EXTRA      可选，追加传给 train.py 的参数（空格分隔，勿加引号包裹整条）。
#                    例: TRAIN_EXTRA="--condition_guidance_w_task 1.2 --condition_guidance_w_text 0.8"
#   EVAL_EXTRA_CMD   可选，追加传给 evaluate.py 的额外参数（勿与脚本内数组 EVAL_EXTRA 混淆）。
#   USE_RETURNS  设为 1 时，在 TRAIN_EXTRA / EVAL_EXTRA_CMD 基础上再追加
#                    --returns_condition --include_returns（显式标量 return 条件，与 config 手工改等价）
#   CUDA_VISIBLE_DEVICES  见脚本中部「GPU」注释；或设 GPU_ID（由 scripts/gpu_env.sh 处理）。
#
# 同一次 bash：run{N}_seed* 仅 seed 递增；仅「完整流水线」结束后 N 写入 .gtg_pipeline_batch 并在下次 +1。
#
# GPU（多卡 / 避免 OOM）:
#   - 推荐: 运行前 export CUDA_VISIBLE_DEVICES=0   （或 1,2,3，物理卡号）
#   - 或: GPU_ID=2 bash run_multitask.sh ...
#   - 或: 编辑本仓库 scripts/gpu_env.sh 上方的示例行，或在下面取消注释:
# export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/gpu_env.sh
source "$_SCRIPT_DIR/scripts/gpu_env.sh"

# 多任务：同一组任务不因命令行顺序不同而换目录（字典序排序后逗号连接）
_canon_train_tasks_csv() {
  echo "$1" | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -v '^$' | sort | paste -sd, -
}

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
  echo "用法: bash run_multitask.sh <train_tasks> <num_runs> <n_traj> <k> <eps> [horizon] [frac] [sigma]"
  echo "示例: bash run_multitask.sh \"dkitty,ant\" 3 1000 50 0.05 64 1.0 0.0"
  echo ""
  echo "环境变量 START_SEED（默认0）、AUTO_CONTINUE（默认0；设为1则从已有最大 seed 后继续）见脚本头部注释。"
  exit 1
fi

TRAIN_TASKS="$1"
NUM_RUNS="$2"
N_TRAJ="$3"
K="$4"
EPS="$5"

PYTHON="${PYTHON:-/home/xk/anaconda3/envs/gtg/bin/python}"
PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# 与 construct / train / evaluate 中 multi_* 目录一致（ant,dkitty 与 dkitty,ant 相同）
TRAIN_TASKS="$(_canon_train_tasks_csv "$TRAIN_TASKS")"
HORIZON="${6:-64}"
FRAC="${7:-1.0}"
SIGMA="${8:-0.0}"

_rs=""
if [[ "${USE_RETURNS:-0}" == "1" ]]; then
  _rs="${RESULTS_SUFFIX:-_retcond}"
fi
RESULTS="${RESULTS:-$PROJECT/results/multitask_$(echo "$TRAIN_TASKS" | tr ',' '_')${_rs}}"

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
echo "  评估: 默认评全部训练任务；EVAL_ALL=${EVAL_ALL:-1}（0=仅评一个，见 EVAL_SINGLE_TASK）"
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

if [[ "${USE_RETURNS:-0}" == "1" ]]; then
  TRAIN_EXTRA="${TRAIN_EXTRA:-} --returns_condition --include_returns"
  EVAL_EXTRA_CMD="${EVAL_EXTRA_CMD:-} --returns_condition --include_returns"
  echo "[USE_RETURNS] 已追加 --returns_condition --include_returns 至 train / evaluate"
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
    # shellcheck disable=SC2086
    $PYTHON "$PROJECT/train.py" \
      --train_tasks "$TRAIN_TASKS" \
      --n_traj "$N_TRAJ" \
      --k "$K" \
      --eps "$EPS" \
      --seed "$SEED" \
      --horizon "$HORIZON" \
      --frac "$FRAC" \
      --sigma "$SIGMA" \
      ${TRAIN_EXTRA:-} \
      >"$RUN_DIR/train.log" 2>&1
  else
    echo "  跳过训练（EVAL_ONLY=1），覆盖 evaluate.log"
  fi

  echo "  评估..."
  EVAL_EXTRA=()
  if [[ "$TRAIN_TASKS" == *","* ]] && [[ "${EVAL_ALL:-1}" == "0" ]]; then
    _single="${EVAL_SINGLE_TASK:-$(echo "$TRAIN_TASKS" | cut -d, -f1)}"
    EVAL_EXTRA+=(--eval_only_first --eval_task "$_single")
  fi
  # shellcheck disable=SC2086
  $PYTHON "$PROJECT/evaluate.py" \
    --train_tasks "$TRAIN_TASKS" \
    --n_traj "$N_TRAJ" \
    --k "$K" \
    --eps "$EPS" \
    --seed "$SEED" \
    --horizon "$HORIZON" \
    --frac "$FRAC" \
    --sigma "$SIGMA" \
    "${EVAL_EXTRA[@]}" \
    ${EVAL_EXTRA_CMD:-} \
    >"$RUN_DIR/evaluate.log" 2>&1

  echo "  完成: $RUN_DIR"
done

if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  echo "$BATCH_RUN" >"$BATCH_STATE_FILE"
  echo "[批次] 已写入 $BATCH_STATE_FILE -> $BATCH_RUN（下次完整流水线将使用 run$((BATCH_RUN + 1))_*）"
fi

echo
echo "全部完成。结果目录: $RESULTS"

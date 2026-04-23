#!/usr/bin/env bash
#
# DUO 单任务：轨迹（含 VAE）只构建一次，再按不同 seed 多次 train.py + evaluate.py。
# 与 run_multitask.sh 的区别：仅单个任务名，参数形式为「单任务专用」。
#
# 无需单独先跑 train_vae.py：construct_trajectories.py 会调用 train_vae.main。
#
# 用法:
#   bash run_singletask.sh <task_name> <num_runs> <n_traj> <k> <eps> [horizon] [frac] [sigma]
#
# 示例（无文本，与历史一致）:
#   bash run_singletask.sh dkitty 3 4000 20 0.01
#   bash run_singletask.sh dkitty 3 4000 20 0.01 64 1.0 0.0
#
# 与多任务 mt_*_textcond_mttextonly 对齐扩散步数：默认 TRAIN_EPOCHS=200 → n_train_steps = 200 × n_steps_per_epoch（各任务 config 内多为 100）= 20000。
# 与 examples/traj_params_per_task_example2.json 中某任务行对齐时，请把位置参数 n_traj/k/eps 设为该任务在 JSON 里的值（单任务不使用 --traj_params_json）。
# 示例（单任务 + 文本元信息，checkpoint 目录带 _textcond，与 --multitask_text_only 的多任务区别为仅 _textcond、无 _mttextonly）:
#   USE_TEXT_CONDITION=1 bash run_singletask.sh ant 1 1000 20 0.05 64 1.0 0.0
#
# 环境变量（可选）:
#   PYTHON         Python，默认 /home/xk/anaconda3/envs/gtg/bin/python
#   PROJECT        DUO 根目录，默认本脚本所在目录
#   RESULTS        结果目录，默认 PROJECT/results/single_task/<task>_frac<F>_sigma<S>/<n_traj>x<h>_k<k>_eps<eps>[_<RESULTS_SUFFIX>]
#                  与 scripts/analyze_eval_results.py 的 single_task 聚合布局一致。
#                  USE_RETURNS=1 时在「超参」目录名末追加 RESULTS_SUFFIX（默认 _ret），与无 returns 分开
#   RESULTS_SUFFIX 仅 USE_RETURNS=1 且未显式设置 RESULTS 时作用于超参目录名；设为空则不加后缀
#   START_SEED     本批第一次使用的随机种子，默认 0
#   AUTO_CONTINUE  默认 0。设为 1 时在 RESULTS 下扫描 run*_seed*，从 max_seed+1 起跑本批
#   EVAL_ONLY        默认 0。设为 1 时跳过轨迹构建与 train，只对已有 run*_seed* 目录重写 evaluate.log
#                    （需与之前成功训练时使用相同的 task/num_runs/START_SEED/参数，以便找到 checkpoint）
#   BATCH_RUN        可选，仅 EVAL_ONLY=1 时生效：指定要重评的批次号 N（runN_seed*）。不设则使用
#                    $RESULTS/.gtg_pipeline_batch 中记录的「上一轮完整流水线」批次号。
#   USE_RETURNS  设为 1 时，train/evaluate 追加 --returns_condition --include_returns（显式标量 return 条件）
#   TRAIN_EPOCHS  扩散训练 epoch 数，传给 train.py --train_epochs（n_train_steps = TRAIN_EPOCHS * config 内 n_steps_per_epoch）。
#                 默认 200；可 export TRAIN_EPOCHS=N 覆盖。与全任务多任务 text 训练对齐时请保持 TRAIN_EPOCHS 一致（如 200 → 20000 steps）。
#   USE_TEXT_CONDITION  设为 1 时 train/evaluate 追加 --use_text_condition（单任务 + task_metadata 文本条件；勿加 --multitask_text_only，该开关仅多任务或 real_task 微调）。
#                 结果超参目录名默认再追加 SINGLE_TASK_TEXTCOND_SUFFIX（默认 _textcond），避免与无文本实验混目录。
#   TEXT_ENCODER_MODEL  可选，sentence-transformers 模型目录或 Hub 名；未设置时若存在 ../models--sentence-transformers--all-MiniLM-L6-v2/... 快照则自动使用（与 run_multitask.sh 一致）。
#   TRAIN_EXTRA / EVAL_EXTRA_CMD  可选，额外传给 train.py / evaluate.py 的参数（例如 --condition_guidance_w_text 8.0）；勿与 USE_TEXT_CONDITION 重复传 --use_text_condition。
#   CUDA_VISIBLE_DEVICES / GPU_ID  见脚本中部「GPU」注释与 scripts/gpu_env.sh。
#   CPU_THREADS      可选，限制 OpenMP/BLAS/PyTorch CPU 线程数；等价于 train/evaluate/construct 的 --cpu_threads N。
#   PROXY_FILTER     可选 0/1；若 export 则 train/evaluate 追加 --proxy_filter（不设则默认 1：训练 proxy 并在评估中筛选）。
#   LATENT_DIM       默认 32；非 32 时传 --latent_dim，且 RESULTS 超参目录追加 _latent{d}（与 train.py RUN.prefix 一致）。
#   TRAIN_TIMESTEP_BIAS_POWER   默认 0.0（关闭）。>0 时 train.py --train_timestep_bias_power，训练离散 t 偏斜小 t（如 0.5）。
#   TRAIN_LOSS_MIN_SNR_GAMMA    默认 0.0（关闭）。>0 时 train.py --train_loss_min_snr_gamma，min-SNR 损失加权（如 5）。
#                 非零时 RESULTS 超参目录与 trained_models 中段会追加 _tsbias… / _msnr…（与 train.py 一致）。
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
  echo "用法: bash run_singletask.sh <task_name> <num_runs> <n_traj> <k> <eps> [horizon] [frac] [sigma]"
  echo "示例: bash run_singletask.sh dkitty 3 4000 20 0.01"
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
LATENT_DIM="${LATENT_DIM:-32}"
TRAIN_TIMESTEP_BIAS_POWER="${TRAIN_TIMESTEP_BIAS_POWER:-0.0}"
TRAIN_LOSS_MIN_SNR_GAMMA="${TRAIN_LOSS_MIN_SNR_GAMMA:-0.0}"

PYTHON="${PYTHON:-/home/xk/anaconda3/envs/gtg/bin/python}"
PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
_DIFF_TRAIN_SUF="$(
  cd "$PROJECT" && PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -c \
    'import sys; from diffuser.utils.multitask_canon import diffusion_train_path_suffix; print(diffusion_train_path_suffix(float(sys.argv[1]), float(sys.argv[2])))' \
    "${TRAIN_TIMESTEP_BIAS_POWER}" "${TRAIN_LOSS_MIN_SNR_GAMMA}"
)"
# 与 results/single_task/** 及 analyze_eval_results 约定一致：tasks_frac → hyper（含 n_traj x horizon）
_ST_FRAC_SIG="${task_name}_frac${FRAC}_sigma${SIGMA}"
_ST_HYPER="${n_traj}x${HORIZON}_k${k}_eps${eps}"
if [[ "${USE_RETURNS:-0}" == "1" ]]; then
  _ST_HYPER="${_ST_HYPER}${RESULTS_SUFFIX:-_ret}"
fi
if [[ "${USE_TEXT_CONDITION:-0}" == "1" ]]; then
  _ST_HYPER="${_ST_HYPER}${SINGLE_TASK_TEXTCOND_SUFFIX:-_textcond}"
fi
_ST_HYPER="${_ST_HYPER}${_DIFF_TRAIN_SUF}"
if [[ "${LATENT_DIM}" != "32" ]]; then
  _ST_HYPER="${_ST_HYPER}_latent${LATENT_DIM}"
fi
base_dir="${RESULTS:-$PROJECT/results/single_task/${_ST_FRAC_SIG}/${_ST_HYPER}}"
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

echo "=== DUO 单任务: $task_name | num_runs=$num_runs | start_seed=$START_SEED (seed: $START_SEED .. $((START_SEED + num_runs - 1))) ==="
echo "  n_traj=$n_traj k=$k eps=$eps horizon=$HORIZON frac=$FRAC sigma=$SIGMA  LATENT_DIM=${LATENT_DIM}"
echo "  USE_TEXT_CONDITION=${USE_TEXT_CONDITION:-0}  TRAIN_EPOCHS=${TRAIN_EPOCHS:-200}（对齐多任务时请与 mt 训练一致）"
echo "  TRAIN_TIMESTEP_BIAS_POWER=${TRAIN_TIMESTEP_BIAS_POWER}  TRAIN_LOSS_MIN_SNR_GAMMA=${TRAIN_LOSS_MIN_SNR_GAMMA}"
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
    --latent_dim "$LATENT_DIM" \
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

TRAIN_EPOCHS="${TRAIN_EPOCHS:-200}"
if ! [[ "$TRAIN_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "错误: TRAIN_EPOCHS 须为正整数，当前: ${TRAIN_EPOCHS}" >&2
  exit 1
fi
_TE_EXTRA=(--train_epochs "$TRAIN_EPOCHS")
echo "[TRAIN_EPOCHS] 扩散训练 ${_TE_EXTRA[*]}"

PROXY_FILTER_EXTRA=()
if [[ -n "${PROXY_FILTER:-}" ]]; then
  PROXY_FILTER_EXTRA=(--proxy_filter "${PROXY_FILTER}")
fi

# 离线 MiniLM：与 run_multitask.sh 相同默认路径
_TEXT_ENCODER_REL="../models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
if [[ -z "${TEXT_ENCODER_MODEL:-}" ]]; then
  _te_resolve="$(cd "$PROJECT" && realpath "$_TEXT_ENCODER_REL" 2>/dev/null || true)"
  if [[ -n "$_te_resolve" && -d "$_te_resolve" ]]; then
    TEXT_ENCODER_MODEL="$_te_resolve"
  fi
  unset _te_resolve
fi
unset _TEXT_ENCODER_REL

TEXT_TRAIN_EXTRA=()
TEXT_EVAL_EXTRA=()
if [[ "${USE_TEXT_CONDITION:-0}" == "1" ]]; then
  if [[ "${TRAIN_EXTRA:-}" != *"--use_text_condition"* ]]; then
    TEXT_TRAIN_EXTRA=(--use_text_condition)
  fi
  if [[ "${EVAL_EXTRA_CMD:-}" != *"--use_text_condition"* ]]; then
    TEXT_EVAL_EXTRA=(--use_text_condition)
  fi
  if [[ -n "${TEXT_ENCODER_MODEL:-}" ]] \
    && [[ "${TRAIN_EXTRA:-}" != *"--text_encoder_model"* ]] \
    && [[ "${EVAL_EXTRA_CMD:-}" != *"--text_encoder_model"* ]]; then
    TEXT_TRAIN_EXTRA+=(--text_encoder_model="${TEXT_ENCODER_MODEL}")
    TEXT_EVAL_EXTRA+=(--text_encoder_model="${TEXT_ENCODER_MODEL}")
    echo "[TEXT_ENCODER] --text_encoder_model=${TEXT_ENCODER_MODEL}"
  elif [[ "${TRAIN_EXTRA:-}" != *"--text_encoder_model"* ]] \
    && [[ "${EVAL_EXTRA_CMD:-}" != *"--text_encoder_model"* ]]; then
    echo "[TEXT_ENCODER] 未设置 TEXT_ENCODER_MODEL 且未找到默认快照；请 export TEXT_ENCODER_MODEL=... 或放置 MiniLM 快照" >&2
  fi
fi

_train_extra_arr=()
if [[ -n "${TRAIN_EXTRA:-}" ]]; then
  read -r -a _train_extra_arr <<< "$TRAIN_EXTRA"
fi
_eval_cmd_extra_arr=()
if [[ -n "${EVAL_EXTRA_CMD:-}" ]]; then
  read -r -a _eval_cmd_extra_arr <<< "$EVAL_EXTRA_CMD"
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
      --latent_dim "$LATENT_DIM" \
      --train_timestep_bias_power "$TRAIN_TIMESTEP_BIAS_POWER" \
      --train_loss_min_snr_gamma "$TRAIN_LOSS_MIN_SNR_GAMMA" \
      "${RETURNS_EXTRA[@]}" \
      "${_TE_EXTRA[@]}" \
      "${TEXT_TRAIN_EXTRA[@]}" \
      "${_train_extra_arr[@]}" \
      "${PROXY_FILTER_EXTRA[@]}" \
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
    --latent_dim "$LATENT_DIM" \
    --train_timestep_bias_power "$TRAIN_TIMESTEP_BIAS_POWER" \
    --train_loss_min_snr_gamma "$TRAIN_LOSS_MIN_SNR_GAMMA" \
    "${RETURNS_EXTRA[@]}" \
    "${TEXT_EVAL_EXTRA[@]}" \
    "${_eval_cmd_extra_arr[@]}" \
    "${PROXY_FILTER_EXTRA[@]}" \
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

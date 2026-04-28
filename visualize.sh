#!/usr/bin/env bash
#
# DUO：同一 task 下跑四种设置的采样 Oracle 曲线（wandb）：
# 多任务+文本 / 多任务+任务标签 / 单任务+文本 / 单任务基线（无 text、无 returns）。
#
# 用法:
#   bash visualize.sh <task_name> [seed]
#   bash visualize.sh <task_name> --multi_seeds <NUM_SEEDS> [START_SEED]
#
# 示例:
#   bash visualize.sh ant 0
#   bash visualize.sh ant --multi_seeds 5 0
#   TEXT_ENCODER_MODEL=/path/to/MiniLM bash visualize.sh dkitty 1
#
# 环境变量:
#   PYTHON          默认 ~/anaconda3/envs/gtg/bin/python（需 PyTorch）
#   PROJECT         仓库根目录，默认本脚本所在目录（重命名后为 DUO/）
#   MT_EXPECT_HEX   默认 911054c35daad7e0 — 与 traj 签名生成的 mt_<hex> 一致时起提示作用（可选校验）
#   SAMPLE_VIZ_STRIDE   默认 10
#   SAMPLE_VIZ_MAX_QUERIES 传给 evaluate（默认 512）
#   NUM_SEEDS       与 --multi_seeds 等价：>1 时对 START_SEED..START_SEED+NUM_SEEDS-1 各跑一遍，
#                   将各 tag 的曲线按 viz_step 取平均后写入 wandb（每组一根曲线）。
#   START_SEED      默认与单 seed 模式的 seed 一致；也可用 --multi_seeds 第三个参数指定。
#   VISUALIZE_RESULTS  本脚本主日志与各 eval 分日志的根目录；默认
#                      PROJECT/results/visualize/task_<TASK>_seed<SEED>/
#                      或 .../task_<TASK>_n<NUM_SEEDS>_s<START_SEED>/
#   DUMP_ROOT       多 seed 时 jsonl 目录（默认 $VISUALIZE_RESULTS/sample_viz_dump）
#   WANDB_RUN_GROUP_PREFIX 默认 duo_viz — 单 seed：${PREFIX}_${task}_seed${seed}
#                           多 seed：聚合 run 使用 ${PREFIX}_${task}_n${NUM_SEEDS}_avg
#   SKIP_*          SKIP_MT_TEXT=1 等可跳过某一种（用于缺 checkpoint 时）
#                   注意：mt_text 与 mt_task 对应 trained_models 下不同超参目录——
#                   mt_text → mt_<hex>_textcond_mttextonly/；mt_task → mt_<hex>/（仅 task 标签）。
#                   若只训练过前者，须训后者或置 SKIP_MT_TASK=1。
#   VISUALIZE_FORCE_EVAL  置 1 时即使 eval_<tag>_seed*.log 已存在也重新跑 evaluate；默认跳过已存在的分日志。
#   EXTRA_EVAL_FLAGS  追加到每次 evaluate.py 的参数（空格分隔字符串）
#   LATENT_DIM          默认 32；非 32 时传给 evaluate 与 print_multitask_ckpt_hyper_dir，与 trained_models 下 mt_*_latent{d} 一致。
#   TRAIN_TIMESTEP_BIAS_POWER / TRAIN_LOSS_MIN_SNR_GAMMA  默认 0；非零须与训练 checkpoint 路径一致（传给 evaluate 与 hyper 打印）。
#   VISUALIZE_TRAIN_EPOCHS  默认 1400：evaluate 优先加载 state_{epoch*n_steps_per_epoch}.pt（与 checkpoint-sweep 对齐）
#   VISUALIZE_N_STEPS_PER_EPOCH 默认 100（与训练一致）；仅用于说明与保持含义一致
#
# 日志：正常运行时脚本与 tqdm 等均写入 results/visualize/<slug>/visualize.log，各次 evaluate 另有
#       eval_<tag>_seed<seed>.log；多 seed 聚合写入 aggregate_wandb.log。用法错误仍打印到终端。
#
# 假设：与各设置对应的 checkpoint 已在 trained_models 下存在（轨迹参数与下列一致）。
#

set -euo pipefail

_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-${_PROJECT}}"
cd "$PROJECT"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/home/xk/anaconda3/envs/gtg/bin/python}"
if [[ $# -lt 1 ]]; then
  echo "用法: bash visualize.sh <task_name> [seed]" >&2
  echo "   或: bash visualize.sh <task_name> --multi_seeds <NUM_SEEDS> [START_SEED]" >&2
  echo "示例: bash visualize.sh ant 0" >&2
  echo "      bash visualize.sh ant --multi_seeds 5 0" >&2
  exit 1
fi

TASK="$1"
shift || exit 1

NUM_SEEDS="${NUM_SEEDS:-1}"
START_SEED="${START_SEED:-0}"

if [[ "${1:-}" == "--multi_seeds" ]]; then
  NUM_SEEDS="${2:?需要 NUM_SEEDS}"
  shift 2
  START_SEED="${1:-0}"
  shift || true
  SEED="$START_SEED"
else
  SEED="${1:-0}"
  START_SEED="$SEED"
  [[ $# -ge 1 ]] && shift || true
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --multi_seeds)
      NUM_SEEDS="${2:?}"
      shift 2
      START_SEED="${1:-0}"
      shift || true
      ;;
    *)
      echo "未知参数: $1" >&2
      exit 1
      ;;
  esac
done

TRAJ_JSON="${TRAJ_JSON:-$PROJECT/examples/traj_params_per_task_example2.json}"
FULL_MT="${FULL_MT:-ant,dkitty,gtopx2,gtopx3,gtopx4,gtopx6,superconductor,tfbind10,tfbind8}"
FRAC="${FRAC:-1.0}"
SIGMA="${SIGMA:-0.0}"
HORIZON="${HORIZON:-64}"
CTX_LEN="${CTX_LEN:-32}"
# 与 run_multitask / prepare_multitask_traj 一致：标量基线再被 JSON 按任务覆盖
N_TRAJ_MT="${N_TRAJ_MT:-1000}"
K_MT="${K_MT:-50}"
EPS_MT="${EPS_MT:-0.05}"

SAMPLE_VIZ_STRIDE="${SAMPLE_VIZ_STRIDE:-5}"
SAMPLE_VIZ_MAX_QUERIES="${SAMPLE_VIZ_MAX_QUERIES:-512}"
MT_EXPECT_HEX="${MT_EXPECT_HEX:-911054c35daad7e0}"
WANDB_RUN_GROUP_PREFIX="${WANDB_RUN_GROUP_PREFIX:-duo_viz}"
LATENT_DIM="${LATENT_DIM:-32}"
TRAIN_TIMESTEP_BIAS_POWER="${TRAIN_TIMESTEP_BIAS_POWER:-0.0}"
TRAIN_LOSS_MIN_SNR_GAMMA="${TRAIN_LOSS_MIN_SNR_GAMMA:-0.0}"
VISUALIZE_TRAIN_EPOCHS="${VISUALIZE_TRAIN_EPOCHS:-1400}"
VISUALIZE_N_STEPS_PER_EPOCH="${VISUALIZE_N_STEPS_PER_EPOCH:-100}"

read_n_k_eps() {
  "$PYTHON" - "$TRAJ_JSON" "$TASK" <<'PY'
import json, sys
path, task = sys.argv[1], sys.argv[2]
with open(path) as f:
    j = json.load(f)
row = j.get(task) or j["defaults"]
print(int(row["n_traj"]), int(row["k"]), float(row["eps"]))
PY
}

read -r N_TRAJ_ST K_ST EPS_ST <<<"$(read_n_k_eps)"

# 与 run_singletask 一样：主输出写入 results/，不在终端刷屏（用法类错误仍在 stderr 打印）
if [[ "$NUM_SEEDS" -gt 1 ]]; then
  _VSLUG="task_${TASK}_n${NUM_SEEDS}_s${START_SEED}"
else
  _VSLUG="task_${TASK}_seed${SEED}"
fi
VISUALIZE_RESULTS="${VISUALIZE_RESULTS:-$PROJECT/results/visualize/${_VSLUG}}"
mkdir -p "$VISUALIZE_RESULTS"
exec >"$VISUALIZE_RESULTS/visualize.log" 2>&1
echo "[visualize] 主日志（本文件）: $VISUALIZE_RESULTS/visualize.log"

if [[ "$NUM_SEEDS" -gt 1 ]]; then
  DUMP_ROOT="${DUMP_ROOT:-$VISUALIZE_RESULTS/sample_viz_dump}"
  mkdir -p "$DUMP_ROOT"
  export WANDB_DISABLED=1
  export WANDB_RUN_GROUP="${WANDB_RUN_GROUP_PREFIX}_${TASK}_n${NUM_SEEDS}_avg"
  EXTRA=(
    --sample_viz_stride "$SAMPLE_VIZ_STRIDE"
    --sample_viz_max_queries "$SAMPLE_VIZ_MAX_QUERIES"
    --sample_viz_dump_jsonl "$DUMP_ROOT"
  )
else
  # 单 seed：默认直接写 wandb（每个 tag 一个 run）。
  # 若设 VISUALIZE_SINGLE_SEED_AGG=1，则也走“dump → aggregate → wandb”的路径，
  # 生成一个包含四条曲线的聚合 run，便于在同一 group 里查找。
  export WANDB_RUN_GROUP="${WANDB_RUN_GROUP_PREFIX}_${TASK}_seed${SEED}"
  if [[ "${VISUALIZE_SINGLE_SEED_AGG:-1}" == "1" ]]; then
    DUMP_ROOT="${DUMP_ROOT:-$VISUALIZE_RESULTS/sample_viz_dump}"
    mkdir -p "$DUMP_ROOT"
    export WANDB_DISABLED=1
    EXTRA=(
      --sample_viz_stride "$SAMPLE_VIZ_STRIDE"
      --sample_viz_max_queries "$SAMPLE_VIZ_MAX_QUERIES"
      --sample_viz_dump_jsonl "$DUMP_ROOT"
    )
  else
    unset WANDB_DISABLED || true
    EXTRA=(--sample_viz_wandb --sample_viz_stride "$SAMPLE_VIZ_STRIDE" --sample_viz_max_queries "$SAMPLE_VIZ_MAX_QUERIES")
  fi
fi

# 离线 MiniLM（与 run_multitask 一致）
_TEXT_ENCODER_REL="../models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
if [[ -z "${TEXT_ENCODER_MODEL:-}" ]]; then
  _te_resolve="$(cd "$PROJECT" && realpath "$_TEXT_ENCODER_REL" 2>/dev/null || true)"
  if [[ -n "$_te_resolve" && -d "$_te_resolve" ]]; then
    export TEXT_ENCODER_MODEL="$_te_resolve"
  fi
fi
unset _TEXT_ENCODER_REL

echo "=== DUO visualize: task=$TASK ==="
if [[ "$NUM_SEEDS" -gt 1 ]]; then
  echo "multi-seed: NUM_SEEDS=$NUM_SEEDS START_SEED=$START_SEED → seeds $START_SEED..$((START_SEED + NUM_SEEDS - 1))"
  echo "DUMP_ROOT=$DUMP_ROOT"
else
  echo "single seed: SEED=$SEED"
fi
echo "VISUALIZE_RESULTS=$VISUALIZE_RESULTS"
echo "PROJECT=$PROJECT"
echo "WANDB_RUN_GROUP=$WANDB_RUN_GROUP"
echo "单任务轨迹（JSON）: n_traj=$N_TRAJ_ST k=$K_ST eps=$EPS_ST"
echo "多任务轨迹: n_traj=$N_TRAJ_MT k=$K_MT eps=$EPS_MT + $TRAJ_JSON  LATENT_DIM=${LATENT_DIM}"

# 可选：打印并校验 multitask slug 是否含期望 hex（需 torch）
if [[ -x "$PYTHON" ]] && "$PYTHON" -c "import torch" 2>/dev/null; then
  _hyper="$("$PYTHON" "$PROJECT/scripts/print_multitask_ckpt_hyper_dir.py" \
    --train_tasks "$FULL_MT" \
    --frac "$FRAC" \
    --sigma "$SIGMA" \
    --n_traj "$N_TRAJ_MT" \
    --k "$K_MT" \
    --eps "$EPS_MT" \
    --horizon "$HORIZON" \
    --latent_dim "$LATENT_DIM" \
    --train_timestep_bias_power "$TRAIN_TIMESTEP_BIAS_POWER" \
    --train_loss_min_snr_gamma "$TRAIN_LOSS_MIN_SNR_GAMMA" \
    ${TRAJ_JSON:+--traj_params_json "$TRAJ_JSON"} 2>/dev/null | tail -n 1)"
  echo "多任务 hyper 目录段（text+mttextonly 打印脚本）: ${_hyper:-<empty>}"
  if [[ -n "${_hyper:-}" ]] && [[ -n "$MT_EXPECT_HEX" ]]; then
    if [[ "$_hyper" != *"$MT_EXPECT_HEX"* ]]; then
      echo "[warn] 当前 traj 签名给出的目录不含期望 hex=$MT_EXPECT_HEX（训练若用不同 JSON/标量则正常）。仍继续。" >&2
    else
      echo "[ok] hyper 含 mt_${MT_EXPECT_HEX}（与默认预训练命名一致）。"
    fi
  fi
else
  echo "[warn] 无法 import torch，跳过 hyper 校验；请用含 PyTorch 的 PYTHON。"
fi

[[ -n "${EXTRA_EVAL_FLAGS:-}" ]] && read -r -a _EXTRA_USER <<< "$EXTRA_EVAL_FLAGS" || _EXTRA_USER=()

run_eval() {
  local tag="$1"
  shift
  local _elog="$VISUALIZE_RESULTS/eval_${tag}_seed${SEED:-0}.log"
  if [[ -f "$_elog" && "${VISUALIZE_FORCE_EVAL:-0}" != "1" ]]; then
    echo ""
    echo "---------- [$tag] seed=${SEED:-?} → 跳过（已有 $_elog），设 VISUALIZE_FORCE_EVAL=1 可强制重跑 ----------"
    return 0
  fi
  echo ""
  echo "---------- [$tag] seed=${SEED:-?} → eval 日志: $_elog ----------"
  # shellcheck disable=SC2086
  "$PYTHON" "$PROJECT/evaluate.py" \
    --latent_dim "$LATENT_DIM" \
    --train_timestep_bias_power "$TRAIN_TIMESTEP_BIAS_POWER" \
    --train_loss_min_snr_gamma "$TRAIN_LOSS_MIN_SNR_GAMMA" \
    --train_epochs "$VISUALIZE_TRAIN_EPOCHS" \
    "$@" \
    "${EXTRA[@]}" \
    --sample_viz_tag "$tag" \
    "${_EXTRA_USER[@]}" \
    >"$_elog" 2>&1
}

run_all_four() {
  # ① 多任务 + 文本 + multitask_text_only（mt_text）
  if [[ "${SKIP_MT_TEXT:-0}" != "1" ]]; then
    run_eval mt_text \
      --train_tasks "$FULL_MT" \
      --eval_task "$TASK" \
      --eval_only_first \
      --horizon "$HORIZON" --ctx_len "$CTX_LEN" --seed "$SEED" \
      --frac "$FRAC" --sigma "$SIGMA" \
      --n_traj "$N_TRAJ_MT" --k "$K_MT" --eps "$EPS_MT" \
      ${TRAJ_JSON:+--traj_params_json "$TRAJ_JSON"} \
      --use_text_condition --multitask_text_only \
      --condition_guidance_w_text 8.0 \
      --train_timestep_bias_power 0.5 \
      --learning_rate 0.0002 \
      --run_suffix _ce0.005
  fi

  # ② 多任务 + task 标签（无 text / 无 mttextonly）（mt_task）
  if [[ "${SKIP_MT_TASK:-0}" != "1" ]]; then
    run_eval mt_task \
      --train_tasks "$FULL_MT" \
      --eval_task "$TASK" \
      --eval_only_first \
      --horizon "$HORIZON" --ctx_len "$CTX_LEN" --seed "$SEED" \
      --frac "$FRAC" --sigma "$SIGMA" \
      --n_traj "$N_TRAJ_MT" --k "$K_MT" --eps "$EPS_MT" \
      ${TRAJ_JSON:+--traj_params_json "$TRAJ_JSON"}
  fi

  # ③ 单任务 + 文本（st_text）
  if [[ "${SKIP_ST_TEXT:-0}" != "1" ]]; then
    _te_args=()
    if [[ -n "${TEXT_ENCODER_MODEL:-}" ]]; then
      _te_args=(--text_encoder_model "${TEXT_ENCODER_MODEL}")
    fi
    run_eval st_text \
      --task "$TASK" \
      --horizon "$HORIZON" --ctx_len "$CTX_LEN" --seed "$SEED" \
      --frac "$FRAC" --sigma "$SIGMA" \
      --n_traj "$N_TRAJ_ST" --k "$K_ST" --eps "$EPS_ST" \
      --use_text_condition \
      --condition_guidance_w_text 8.0 \
      "${_te_args[@]}"
  fi

  # ④ 单任务基线（st_duo）
  if [[ "${SKIP_ST_DUO:-0}" != "1" ]]; then
    run_eval st_duo \
      --task "$TASK" \
      --horizon "$HORIZON" --ctx_len "$CTX_LEN" --seed "$SEED" \
      --frac "$FRAC" --sigma "$SIGMA" \
      --n_traj "$N_TRAJ_ST" --k "$K_ST" --eps "$EPS_ST"
  fi
}

if [[ "$NUM_SEEDS" -gt 1 ]]; then
  for ((i = 0; i < NUM_SEEDS; i++)); do
    SEED=$((START_SEED + i))
    echo ""
    echo "========== seed=$SEED (${i}/$((NUM_SEEDS - 1))) =========="
    run_all_four
  done
  unset WANDB_DISABLED || true
  echo ""
  echo "聚合 $NUM_SEEDS 个 seed 的曲线 → wandb group=$WANDB_RUN_GROUP"
  "$PYTHON" "$PROJECT/scripts/sample_viz_aggregate_wandb.py" \
    --dump_dir "$DUMP_ROOT" \
    --group "$WANDB_RUN_GROUP" \
    --num_seeds "$NUM_SEEDS" \
    --start_seed "$START_SEED" \
    >"$VISUALIZE_RESULTS/aggregate_wandb.log" 2>&1
  echo "[aggregate] 日志: $VISUALIZE_RESULTS/aggregate_wandb.log"
else
  run_all_four
  if [[ "${VISUALIZE_SINGLE_SEED_AGG:-1}" == "1" ]]; then
    unset WANDB_DISABLED || true
    echo ""
    echo "聚合 1 个 seed 的曲线 → wandb group=$WANDB_RUN_GROUP"
    "$PYTHON" "$PROJECT/scripts/sample_viz_aggregate_wandb.py" \
      --dump_dir "$DUMP_ROOT" \
      --group "$WANDB_RUN_GROUP" \
      --num_seeds 1 \
      --start_seed "$SEED" \
      >"$VISUALIZE_RESULTS/aggregate_wandb.log" 2>&1
    echo "[aggregate] 日志: $VISUALIZE_RESULTS/aggregate_wandb.log"
  fi
fi

echo ""
echo "全部完成。日志目录: $VISUALIZE_RESULTS（主日志 visualize.log，各次 evaluate：eval_<tag>_seed*.log）"
echo "在 wandb 工程 decdiff-opt 中筛选 RUN_GROUP=$WANDB_RUN_GROUP，"
echo "对比指标 sample_viz/<tag>/mean_y、sample_viz/<tag>/max_y（及 _norm）。"
if [[ "$NUM_SEEDS" -gt 1 ]]; then
  echo "（多 seed：wandb 为跨 seed 平均曲线；jsonl 在 $DUMP_ROOT）"
fi

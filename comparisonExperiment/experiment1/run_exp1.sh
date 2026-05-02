#!/usr/bin/env bash
# Chinese comments: Exp1 DUO 三步（base 训练 → gap few-shot 微调 → gap 采样），支持“已存在则跳过”。
# English doc: Idempotent 3-step runner for comparison1 experiment1 DUO pipeline.
#
# Sweep examples:
#   ./comparisonExperiment/experiment1/run_exp1.sh --seed 0,1,2 --gap 0p250,0p500
#   ./comparisonExperiment/experiment1/run_exp1.sh --seed 0-7 --gap 0.25,0.5   # decimals normalized to 0p250/0p500

set -euo pipefail

# Script path: <DUO_ROOT>/comparisonExperiment/experiment1/run_exp1.sh
DUO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${DUO_ROOT}"

PYTHON="${PYTHON:-python}"

SCRIPT="${DUO_ROOT}/comparisonExperiment/experiment1/duo_train_and_sample.py"

HORIZON="${HORIZON:-32}"
BASE_STEPS="${BASE_STEPS:-8000}"
FS_STEPS="${FS_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-2e-4}"

SAMPLE_TRAJ="${SAMPLE_TRAJ:-256}"
TOPK_POINTS="${TOPK_POINTS:-128}"

# Optional overrides (advanced): explicit ckpt paths for all runs (usually empty).
BASE_CKPT_GLOBAL="${BASE_CKPT:-}"
FS_CKPT_GLOBAL="${FS_CKPT:-}"

# Dataset overrides (usually empty; defaults are recomputed per gap unless set explicitly).
TRAIN_MERGED_PKL_GLOBAL="${TRAIN_MERGED_PKL:-}"
FEWSHOT_PKL_TEMPLATE_GLOBAL="${FEWSHOT_PKL:-}"

_resolve_ckpt() {
  # duo_train_and_sample.py saves to: <parent>/seedK/<basename(save_ckpt)>
  local save_arg="$1"
  local seed="$2"
  local p="${DUO_ROOT}/${save_arg}"
  local dir
  dir="$(dirname "${p}")"
  local base
  base="$(basename "${p}")"
  echo "${dir}/seed${seed}/${base}"
}

_norm_gap_tag() {
  # Normalize user gap input to the folder tag style used by run_exp1.py outputs, e.g. 0p250.
  # Accepts: 0.25 | 0p250 | .25 | 1.0 | 1p000
  local raw="$1"
  "${PYTHON}" - <<'PY' "${raw}"
import re
import sys

raw = sys.argv[1].strip()
if not raw:
    raise SystemExit("empty gap")

m = re.fullmatch(r"(\d+)p(\d+)", raw)
if m:
    ip, frac = m.group(1), m.group(2)
    frac = (frac + "000")[:3]
    print(f"{ip}p{frac}")
    raise SystemExit(0)

m = re.fullmatch(r"(\d+)\.(\d+)", raw)
if m:
    ip = m.group(1)
    frac = m.group(2)
    if len(frac) <= 3:
        frac = (frac + "000")[:3]
    else:
        # Too many decimals: truncate (shouldn't happen for exp1 defaults)
        frac = frac[:3]
    print(f"{ip}p{frac}")
    raise SystemExit(0)

# Fallback: try float
val = float(raw)
ip = int(val)
frac_f = val - ip
frac = int(round(frac_f * 1000))
if frac >= 1000:
    ip += 1
    frac -= 1000
print(f"{ip}p{frac:03d}")
PY
}

_split_csv_ints() {
  # Supports: "0,1,2" or "0-3" or "0-7,9"
  local raw="$1"
  local out=()
  IFS=',' read -r -a parts <<< "${raw}"
  local p a b i
  for p in "${parts[@]}"; do
    p="$(echo "${p}" | xargs)"
    [[ -z "${p}" ]] && continue
    if [[ "${p}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      a="${BASH_REMATCH[1]}"
      b="${BASH_REMATCH[2]}"
      if ((10#$a > 10#$b)); then
        echo "错误: seed range invalid: ${p}" >&2
        exit 1
      fi
      for ((i = 10#$a; i <= 10#$b; i++)); do
        out+=("${i}")
      done
    else
      if ! [[ "${p}" =~ ^[0-9]+$ ]]; then
        echo "错误: seed token invalid: ${p}" >&2
        exit 1
      fi
      out+=("${p}")
    fi
  done
  printf '%s\n' "${out[@]}"
}

_split_csv_gaps() {
  local raw="$1"
  local out=()
  IFS=',' read -r -a parts <<< "${raw}"
  local p
  for p in "${parts[@]}"; do
    p="$(echo "${p}" | xargs)"
    [[ -z "${p}" ]] && continue
    out+=("${p}")
  done
  printf '%s\n' "${out[@]}"
}

_usage() {
  cat <<'EOF'
用法:
  ./comparisonExperiment/experiment1/run_exp1.sh [--seed SEEDS] [--gap GAPS]

参数:
  --seed / -s   逗号分隔 seed，或范围：0-7（也支持与逗号混用）
  --gap  / -g   逗号分隔 gap：支持 0p250 或 0.25（会自动规范成目录名风格）

说明:
  - 不传参数时：读取环境变量 SEED / GAP_TAG（兼容旧用法）；默认 SEED=0, GAP_TAG=0p250
  - TRAIN_MERGED_PKL / FEWSHOT_PKL 仍可通过环境变量覆盖（高级）

示例:
  ./comparisonExperiment/experiment1/run_exp1.sh --seed 0,1 --gap 0p250,0p500
  ./comparisonExperiment/experiment1/run_exp1.sh --seed 0-7 --gap 0.25
EOF
}

_run_one() {
  local SEED="$1"
  local GAP_TAG="$2"

  # Canonical merged-train pkl for Step1 (gap-agnostic); keep default aligned with exp1 generator naming.
  local TRAIN_MERGED_PKL="${TRAIN_MERGED_PKL_GLOBAL}"
  if [[ -z "${TRAIN_MERGED_PKL}" ]]; then
    mapfile -t _tm < <(ls -1 "${DUO_ROOT}/generated_datasets/exp1_gap0p000"/train_merged_h"${HORIZON}"_*_lat*.pkl 2>/dev/null || true)
    TRAIN_MERGED_PKL="${_tm[0]:-}"
  fi

  local FEWSHOT_PKL="${FEWSHOT_PKL_TEMPLATE_GLOBAL}"
  if [[ -z "${FEWSHOT_PKL}" ]]; then
    mapfile -t _fs < <(ls -1 "${DUO_ROOT}/generated_datasets/exp1_gap${GAP_TAG}"/D_test_gap"${GAP_TAG}"_fewshot_h"${HORIZON}"_lat*.pkl 2>/dev/null || true)
    FEWSHOT_PKL="${_fs[0]:-}"
  fi

  # ---------- Output locations (match existing results layout under results/comparison1/exp1) ----------
  local BASE_SAVE_ARG="results/comparison1/exp1/models/duo_train_seed${SEED}.pt"
  local FS_SAVE_ARG="results/comparison1/exp1/models/duo_finetune_gap${GAP_TAG}_seed${SEED}.pt"
  local OUT_JSONL="results/comparison1/exp1/gap${GAP_TAG}/candidates/seed${SEED}/duo_fs.jsonl"

  local BASE_RESOLVED FS_RESOLVED BASE_USE FS_USE OUT_ABS
  BASE_RESOLVED="$(_resolve_ckpt "${BASE_SAVE_ARG}" "${SEED}")"
  FS_RESOLVED="$(_resolve_ckpt "${FS_SAVE_ARG}" "${SEED}")"

  if [[ -n "${BASE_CKPT_GLOBAL}" ]]; then
    BASE_USE="${BASE_CKPT_GLOBAL}"
  else
    BASE_USE="${BASE_RESOLVED}"
  fi

  if [[ -n "${FS_CKPT_GLOBAL}" ]]; then
    FS_USE="${FS_CKPT_GLOBAL}"
  else
    FS_USE="${FS_RESOLVED}"
  fi

  OUT_ABS="${DUO_ROOT}/${OUT_JSONL}"

  echo "[exp1-duo] =========="
  echo "[exp1-duo] DUO_ROOT=${DUO_ROOT}"
  echo "[exp1-duo] SEED=${SEED} GAP_TAG=${GAP_TAG}"
  echo "[exp1-duo] TRAIN_MERGED_PKL=${TRAIN_MERGED_PKL}"
  echo "[exp1-duo] FEWSHOT_PKL=${FEWSHOT_PKL}"
  echo "[exp1-duo] base_ckpt_expected=${BASE_USE}"
  echo "[exp1-duo] fs_ckpt_expected=${FS_USE}"
  echo "[exp1-duo] out_jsonl=${OUT_ABS}"

  if [[ ! -f "${SCRIPT}" ]]; then
    echo "错误: 找不到脚本: ${SCRIPT}" >&2
    exit 1
  fi

  if [[ ! -f "${TRAIN_MERGED_PKL}" ]]; then
    echo "错误: TRAIN_MERGED_PKL 不存在: ${TRAIN_MERGED_PKL}" >&2
    echo "      请先运行 comparisonExperiment/experiment1/run_exp1.py 生成数据，或 export TRAIN_MERGED_PKL=..." >&2
    exit 1
  fi

  if [[ ! -f "${FEWSHOT_PKL}" ]]; then
    echo "错误: FEWSHOT_PKL 不存在: ${FEWSHOT_PKL}" >&2
    echo "      请先运行 comparisonExperiment/experiment1/run_exp1.py 生成对应 gap 的 few-shot pkl。" >&2
    exit 1
  fi

  mkdir -p "${DUO_ROOT}/results/comparison1/exp1/tmp/seed${SEED}"

  # ---------- Step 1: base training (generic) ----------
  if [[ -f "${BASE_USE}" ]]; then
    echo "[exp1-duo] Step1 skip: base ckpt exists: ${BASE_USE}"
  else
    echo "[exp1-duo] Step1 run: base training -> ${BASE_SAVE_ARG} (resolved: ${BASE_RESOLVED})"
    ${PYTHON} "${SCRIPT}" \
      --mode train \
      --seed "${SEED}" \
      --horizon "${HORIZON}" \
      --train_steps "${BASE_STEPS}" \
      --batch_size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --train_pkl "${TRAIN_MERGED_PKL}" \
      --save_ckpt "${BASE_SAVE_ARG}" \
      --out_jsonl "results/comparison1/exp1/tmp/seed${SEED}/_unused_base_train.jsonl"
  fi

  if [[ ! -f "${BASE_USE}" ]]; then
    echo "错误: Step1 后仍找不到 base ckpt: ${BASE_USE}" >&2
    echo "      如果你手动移动了 ckpt，请 export BASE_CKPT=/abs/path/to/ckpt.pt" >&2
    exit 1
  fi

  # ---------- Step 2: gap few-shot finetune ----------
  if [[ -f "${FS_USE}" ]]; then
    echo "[exp1-duo] Step2 skip: few-shot ckpt exists: ${FS_USE}"
  else
    echo "[exp1-duo] Step2 run: few-shot finetune -> ${FS_SAVE_ARG} (resolved: ${FS_RESOLVED})"
    ${PYTHON} "${SCRIPT}" \
      --mode train \
      --seed "${SEED}" \
      --horizon "${HORIZON}" \
      --train_steps "${FS_STEPS}" \
      --batch_size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --train_pkl "${FEWSHOT_PKL}" \
      --load_ckpt "${BASE_USE}" \
      --save_ckpt "${FS_SAVE_ARG}" \
      --out_jsonl "results/comparison1/exp1/tmp/seed${SEED}/_unused_ft_gap${GAP_TAG}.jsonl"
  fi

  if [[ ! -f "${FS_USE}" ]]; then
    echo "错误: Step2 后仍找不到 few-shot ckpt: ${FS_USE}" >&2
    echo "      如果你手动移动了 ckpt，请 export FS_CKPT=/abs/path/to/ckpt.pt" >&2
    exit 1
  fi

  # ---------- Step 3: sample candidates ----------
  if [[ -f "${OUT_ABS}" ]]; then
    echo "[exp1-duo] Step3 skip: candidates exist: ${OUT_ABS}"
  else
    echo "[exp1-duo] Step3 run: sampling -> ${OUT_ABS}"
    mkdir -p "$(dirname "${OUT_ABS}")"
    ${PYTHON} "${SCRIPT}" \
      --mode sample \
      --seed "${SEED}" \
      --horizon "${HORIZON}" \
      --train_pkl "${TRAIN_MERGED_PKL}" \
      --load_ckpt "${FS_USE}" \
      --sample_traj "${SAMPLE_TRAJ}" \
      --topk_points "${TOPK_POINTS}" \
      --out_jsonl "${OUT_JSONL}"
  fi

  echo "[exp1-duo] done: seed=${SEED} gap=${GAP_TAG}"
  echo "[exp1-duo] candidates: ${OUT_ABS}"
}

SEEDS_SPEC=""
GAPS_SPEC=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      _usage
      exit 0
      ;;
    --seed | -s)
      SEEDS_SPEC="${2:-}"
      shift 2
      ;;
    --gap | -g)
      GAPS_SPEC="${2:-}"
      shift 2
      ;;
    *)
      echo "未知参数: $1" >&2
      _usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${SCRIPT}" ]]; then
  echo "错误: 找不到脚本: ${SCRIPT}" >&2
  exit 1
fi

if [[ -z "${SEEDS_SPEC}" ]]; then
  SEEDS_SPEC="${SEED:-0}"
fi
if [[ -z "${GAPS_SPEC}" ]]; then
  GAPS_SPEC="${GAP_TAG:-0p250}"
fi

mapfile -t SEEDS < <(_split_csv_ints "${SEEDS_SPEC}")

GAPS_RAW=()
while IFS= read -r line; do
  [[ -n "${line}" ]] && GAPS_RAW+=("${line}")
done < <(_split_csv_gaps "${GAPS_SPEC}")

if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "错误: seed 列表为空" >&2
  exit 1
fi
if [[ "${#GAPS_RAW[@]}" -eq 0 ]]; then
  echo "错误: gap 列表为空" >&2
  exit 1
fi

GAPS_NORM=()
for g in "${GAPS_RAW[@]}"; do
  GAPS_NORM+=("$(_norm_gap_tag "${g}")")
done

for seed in "${SEEDS[@]}"; do
  for gap in "${GAPS_NORM[@]}"; do
    _run_one "${seed}" "${gap}"
  done
done

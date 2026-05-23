#!/usr/bin/env bash
# Chinese comment: DUO v4 mt_text vs UniSO-T 对比 — 导出 DUO 候选、（可选）跑 UniSO、统一 oracle 评估、生成 LaTeX 表。
# English doc: Scene-aware Exp1 comparison aligned with quality_bundle_4 + data_exp1_sim_*_3.
#
# 用法（在 DUO 根目录）:
#   ./comparisonExperiment/experiment1/run_comparison_v4_duo_uniso.sh
#   SKIP_UNISO_TRAIN=1 ./comparisonExperiment/experiment1/run_comparison_v4_duo_uniso.sh  # 仅评估已有 ckpt/jsonl
#   RUN_UNISO_TRAIN=1 ./comparisonExperiment/experiment1/run_comparison_v4_duo_uniso.sh --shift sim_mid

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${DUO_ROOT}"
export PYTHONPATH="${DUO_ROOT}:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/home/xk/anaconda3/envs/gtg/bin/python}"
UNISO_PYTHON="${UNISO_PYTHON:-/home/xk/anaconda3/envs/uniso/bin/python}"
UNISO_ROOT="${UNISO_ROOT:-${DUO_ROOT}/../UniSO}"

BUNDLE_ROOT="${QUAL_BUNDLE_ROOT:-${DUO_ROOT}/results/quality_bundle_4}"
OUT_ROOT="${CMP_OUT_ROOT:-${DUO_ROOT}/results/comparison1/exp1_scene_v4}"
TOPK="${CMP_TOPK:-16}"
DUO_SAMPLE_BATCH="${DUO_SAMPLE_BATCH:-128}"
DUO_RESAMPLE="${DUO_RESAMPLE:-1}"
SEED="${SEED:-0}"
UNIVERSAL_CKPT_DIR="${UNIVERSAL_CKPT_DIR:-${DUO_ROOT}/results/quality_training_3/dtrain_universal_seed0/checkpoints}"

RUN_UNISO_TRAIN="${RUN_UNISO_TRAIN:-0}"
SKIP_UNISO_TRAIN="${SKIP_UNISO_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SHIFTS_SPEC="${SHIFTS_SPEC:-sim_low,sim_mid,sim_high}"

_export_duo() {
  local shift="$1"
  local phase="$2"
  local out_dir="${OUT_ROOT}/shift_${shift}_3/${phase}/candidates"
  mkdir -p "${out_dir}"
  if [[ "${DUO_RESAMPLE}" == "1" ]]; then
    return 0
  fi
  local npz="$3"
  if [[ ! -f "${npz}" ]]; then
    echo "[cmp] 警告: 缺少 DUO npz ${npz}" >&2
    return 1
  fi
  "${PYTHON}" comparisonExperiment/experiment1/export_quality_trace_jsonl.py \
    --npz "${npz}" \
    --out_jsonl "${out_dir}/duo_mt_text.jsonl" \
    --method duo_mt_text
}

_eval_cell() {
  local shift="$1"
  local phase="$2"
  local meta="$3"
  local cell_dir="${OUT_ROOT}/shift_${shift}_3/${phase}"
  local cand="${cell_dir}/candidates"
  local methods=""
  if [[ "${SKIP_EVAL}" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "${cand}/duo_mt_text.jsonl" ]]; then
    echo "[cmp] 跳过评估 ${shift}/${phase}: 无 duo_mt_text.jsonl" >&2
    return 0
  fi
  methods="duo_mt_text"
  if [[ -f "${cand}/uniso.jsonl" ]]; then
    methods="${methods},uniso"
  else
    echo "[cmp] 警告: ${shift}/${phase} 无 uniso.jsonl，仅评估 DUO" >&2
  fi
  "${PYTHON}" comparisonExperiment/experiment1/run_eval_exp1.py \
    --gap_dir "${cell_dir}" \
    --test_meta_json "${meta}" \
    --candidates_dir "${cand}" \
    --methods "${methods}" \
    --topk "${TOPK}" \
    --out_json summary.json
}

_resolve_npz() {
  local shift="$1"
  local phase="$2"
  local art
  art="$(find "${BUNDLE_ROOT}/shift_${shift}_3" -maxdepth 1 -type d -name 'artifacts_*_seed0' | head -1)"
  if [[ -z "${art}" ]]; then
    echo ""
    return
  fi
  case "${phase}" in
    zs)
      echo "${art}/quality_trace_mt_text_exp1_D_test_${shift}.meta_shift_zero_shot.npz"
      ;;
    fs10)
      echo "${art}/quality_trace_mt_text_exp1_D_test_${shift}_fewshot_tail10p.meta_shift_few_shot_tail10p.npz"
      ;;
    fs20)
      echo "${art}/quality_trace_mt_text_exp1_D_test_${shift}_fewshot_tail20p.meta_shift_few_shot_tail20p.npz"
      ;;
    fs50)
      echo "${art}/quality_trace_mt_text_exp1_D_test_${shift}_fewshot_tail50p.meta_shift_few_shot_tail50p.npz"
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shift)
      SHIFTS_SPEC="${2:-}"
      shift 2
      ;;
    --bundle)
      BUNDLE_ROOT="${2:-}"
      shift 2
      ;;
    --out)
      OUT_ROOT="${2:-}"
      shift 2
      ;;
    *)
      echo "未知参数: $1" >&2
      exit 1
      ;;
  esac
done

IFS=',' read -r -a SHIFTS <<< "${SHIFTS_SPEC}"
PHASES=(zs fs10 fs20 fs50)

echo "[cmp] BUNDLE_ROOT=${BUNDLE_ROOT}"
echo "[cmp] OUT_ROOT=${OUT_ROOT}"
echo "[cmp] RUN_UNISO_TRAIN=${RUN_UNISO_TRAIN} SKIP_UNISO_TRAIN=${SKIP_UNISO_TRAIN}"

if [[ "${DUO_RESAMPLE}" == "1" ]]; then
  echo "[cmp] DUO resample sample_batch=${DUO_SAMPLE_BATCH} (aligned with UniSO num_solutions)"
  "${PYTHON}" comparisonExperiment/experiment1/export_duo_comparison_candidates.py \
    --bundle_root "${BUNDLE_ROOT}" \
    --uniso_root "${UNISO_ROOT}" \
    --duo_root "${DUO_ROOT}" \
    --out_root "${OUT_ROOT}" \
    --universal_ckpt_dir "${UNIVERSAL_CKPT_DIR}" \
    --sample_batch "${DUO_SAMPLE_BATCH}" \
    --shifts "${SHIFTS_SPEC}"
fi

if [[ "${RUN_UNISO_TRAIN}" == "1" && "${SKIP_UNISO_TRAIN}" != "1" ]]; then
  export DUO_CMP_ROOT="${OUT_ROOT}"
  export PYTHON="${UNISO_PYTHON}"
  export PROJECT_ROOT="${UNISO_ROOT}"
  export WANDB_MODE="${WANDB_MODE:-disabled}"
  export UNISO_LOGGER="${UNISO_LOGGER:-csv}"
  (
    cd "${UNISO_ROOT}"
    ./scripts/run_exp1_uniso_scene_aware.sh --seed "${SEED}" --shift "${SHIFTS_SPEC}"
  )
fi

for shift in "${SHIFTS[@]}"; do
  shift="$(echo "${shift}" | xargs)"
  [[ -z "${shift}" ]] && continue
  META="${UNISO_ROOT}/data_exp1_${shift}_3/exp1_D_test_${shift}.meta.json"
  for phase in "${PHASES[@]}"; do
    NPZ="$(_resolve_npz "${shift}" "${phase}")"
    if [[ "${DUO_RESAMPLE}" != "1" && -n "${NPZ}" && -f "${NPZ}" ]]; then
      _export_duo "${shift}" "${phase}" "${NPZ}" || true
    fi
    _eval_cell "${shift}" "${phase}" "${META}" || true
  done
done

"${PYTHON}" comparisonExperiment/experiment1/export_duo_uniso_comparison_table.py \
  --out_root "${OUT_ROOT}" \
  --bundle_root "${BUNDLE_ROOT}" \
  --uniso_root "${UNISO_ROOT}" \
  --topk "${TOPK}"

echo "[cmp] done -> ${OUT_ROOT}/analysis_table/duo_uniso_comparison.tex"

#!/usr/bin/env bash
# 生成 evaluate 汇总表（scripts/analyze_eval_results.py，仓库根为 DUO）。
#
# sweep_w 消融（w_ablation.{tex,csv}，每 w 一列、跨 seed/run 聚合）:
#   bash run_analyze_eval.sh -w-ablation
#
# 用法:
#   bash run_analyze_eval.sh -short   # max_short.{csv,tex} + nmax.tex + max_real_task + nmax_real_task 等
#   bash run_analyze_eval.sh -real-task  # 仅 max_real_task.* / nmax_real_task.*（真实任务表）
#   bash run_analyze_eval.sh -final   # text_conditioned_result_analysis.tex
#   bash run_analyze_eval.sh -w-ablation # w_ablation.{tex,csv}（eval_sweep_w_text 下优先固定目录；见 make_sweep_w_ablation_table）
#   bash run_analyze_eval.sh          # 生成 short + final（--mode all）；另生成 w_ablation + ce_ablation（若有数据）
#
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$_SCRIPT_DIR"
PYTHON="${PYTHON:-python3}"

_mode="all"
case "${1:-}" in
  -short|--short) _mode=short ;;
  -real-task|--real-task) _mode=real_task ;;
  -final|--final) _mode=final ;;
  -w-ablation|--w-ablation) _mode=w_ablation ;;
  --mode)
    if [[ -z "${2:-}" ]]; then
      echo "Usage: $0 --mode {all|short|real_task|final|w_ablation}" >&2
      exit 1
    fi
    _mode="$2"
    ;;
  "")
    ;;
  *)
    echo "Usage: $0 [-short|-real-task|-final|-w-ablation|--mode {all|short|real_task|final|w_ablation}]" >&2
    exit 1
    ;;
esac

export PYTHONPATH="${_SCRIPT_DIR}:${PYTHONPATH:-}"
exec "$PYTHON" scripts/analyze_eval_results.py --mode "$_mode"

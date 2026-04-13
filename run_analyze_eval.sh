#!/usr/bin/env bash
# 生成 evaluate 汇总表（scripts/analyze_eval_results.py，仓库根为 GTGdfgo）。
#
# 用法:
#   bash run_analyze_eval.sh -short   # 宽表 results/analysis_table/eval_comparison.{csv,md,tex}
#   bash run_analyze_eval.sh -full    # 13 列矩阵 results/analysis_table/eval_comparison_m12.{csv,md,tex}
#   bash run_analyze_eval.sh -final   # _all 表 results/analysis_table/eval_comparison_all.tex
#   bash run_analyze_eval.sh          # 以上均生成（--mode all）；UniSO 从 results/analysis_table/uniso_result.tex 读取
#
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$_SCRIPT_DIR"
PYTHON="${PYTHON:-python3}"

_mode="all"
case "${1:-}" in
  -short|--short) _mode=short ;;
  -full|--full)   _mode=full ;;
  -final|--final) _mode=final ;;
  --mode)
    if [[ -z "${2:-}" ]]; then
      echo "Usage: $0 --mode {all|short|full|final}" >&2
      exit 1
    fi
    _mode="$2"
    ;;
  "")
    ;;
  *)
    echo "Usage: $0 [-short|-full|-final|--mode {all|short|full|final}]" >&2
    exit 1
    ;;
esac

exec "$PYTHON" scripts/analyze_eval_results.py --mode "$_mode"

#!/usr/bin/env bash
# 生成 evaluate 汇总表（scripts/analyze_eval_results.py，仓库根为 DUO）。
#
# sweep_w 消融（max_ablation.{tex,csv}，每 w 一列、跨 seed/run 聚合）:
#   bash run_analyze_eval.sh -sweep-w
#   或 SWEEP_W_MODEL_DIR=results/eval_sweep_w_text/<mt_* 或 train_eval 归档目录> bash run_analyze_eval.sh -sweep-w
#
# 用法:
#   bash run_analyze_eval.sh -short   # max_short.{csv,tex} + nmax.tex
#   bash run_analyze_eval.sh -full    # UniSO + 14 列矩阵 max_extended.{csv,tex}
#   bash run_analyze_eval.sh -final   # text_conditioned_result_analysis.tex
#   bash run_analyze_eval.sh -sweep-w # max_ablation.{tex,csv}（eval_sweep_w_text 下自动选目录，见 make_sweep_w_ablation_table）
#   bash run_analyze_eval.sh          # 以上均生成（--mode all）；text CFG 多 w 对比见 -sweep-w → max_ablation.{tex,csv}
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
  -sweep-w|--sweep-w) _mode=sweep_w ;;
  --mode)
    if [[ -z "${2:-}" ]]; then
      echo "Usage: $0 --mode {all|short|full|final|sweep_w}" >&2
      exit 1
    fi
    _mode="$2"
    ;;
  "")
    ;;
  *)
    echo "Usage: $0 [-short|-full|-final|-sweep-w|--mode {all|short|full|final|sweep_w}]" >&2
    exit 1
    ;;
esac

export PYTHONPATH="${_SCRIPT_DIR}:${PYTHONPATH:-}"
exec "$PYTHON" scripts/analyze_eval_results.py --mode "$_mode"

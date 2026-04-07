#!/usr/bin/env bash
#
# 汇总 GTGdfgo 与 GTG 的 evaluate 结果（scripts/analyze_eval_results.py 的封装）。
#
# 默认输出（无需额外参数）:
#   - Markdown:  <output>.md
#   - CSV:       <output>.csv
#   - LaTeX:     <output>.tex
# 若只要 md/csv、不要 LaTeX，请传 --no-latex，或设环境变量 NO_LATEX=1。
#
# 用法:
#   bash run_analyze_eval.sh
#   bash run_analyze_eval.sh --gtg /path/to/GTG/results -o /path/to/eval_comparison
#   bash run_analyze_eval.sh --no-latex
#   NO_LATEX=1 bash run_analyze_eval.sh
#   bash run_analyze_eval.sh --d-best-json /path/to/d_best.json
#
# 环境变量（可选，未设则与 Python 脚本默认一致）:
#   PROJECT           GTGdfgo 根目录，默认为本脚本所在目录
#   PYTHON            Python 解释器，默认 python3
#   GTGDFGO_RESULTS   传给 --gtgdfgo
#   GTG_RESULTS       传给 --gtg
#   OUTPUT            传给 -o（输出前缀，会生成 .md / .csv / .tex）
#   NO_LATEX          设为 1 时等价于追加 --no-latex（不生成 .tex）
#
# 汇总脚本为 CPU 分析，通常无需设置 GPU。
#
# 命令行参数会追加在末尾；若与环境变量重复，一般以 argparse 后出现的为准。

set -uo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python3}"
SCRIPT="$PROJECT/scripts/analyze_eval_results.py"

ARGS=()
[[ -n "${GTGDFGO_RESULTS:-}" ]] && ARGS+=(--gtgdfgo "$GTGDFGO_RESULTS")
[[ -n "${GTG_RESULTS:-}" ]] && ARGS+=(--gtg "$GTG_RESULTS")
[[ -n "${OUTPUT:-}" ]] && ARGS+=(-o "$OUTPUT")
[[ "${NO_LATEX:-0}" == "1" ]] && ARGS+=(--no-latex)

exec "$PYTHON" "$SCRIPT" "${ARGS[@]}" "$@"

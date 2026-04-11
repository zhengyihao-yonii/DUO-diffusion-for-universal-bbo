#!/usr/bin/env bash
#
# 汇总 GTGdfgo 与 GTG 的 evaluate 结果（scripts/analyze_eval_results.py）。
# 每次运行会全量更新：宽表 eval_comparison.{csv,md,tex} 与矩阵表 eval_comparison_m12.{csv,md,tex}（含 UniSO best 列与 mean rank）。
# 路径与可选 d_best 覆盖见脚本内常量（可选文件 results/d_best.json）。
#
# 用法:
#   bash run_analyze_eval.sh
#
# 环境变量（可选）:
#   PROJECT   GTGdfgo 根目录，默认为本脚本所在目录
#   PYTHON    Python 解释器，默认 python3
#
# 汇总脚本为 CPU 分析，通常无需设置 GPU。

set -uo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${PYTHON:-python3}"
SCRIPT="$PROJECT/scripts/analyze_eval_results.py"

exec "$PYTHON" "$SCRIPT"

#!/usr/bin/env bash
# Few-shot 文本条件扩散微调示例（单任务 real-world 数据 + 全任务预训练 checkpoint）。
# 前置：fewshot_data/<LunarLander|RobotPush|Rover>/similar|unsimilar/*.json；已构造轨迹；VAE 已含该任务。
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

TASK="${1:-lunar_lander}"
BASE_CKPT="${2:?用法: $0 <task> /path/to/state_STEP.pt [train_epochs]}"

TEPOCHS="${3:-30}"

export PYTHON="${PYTHON:-python}"
"$PYTHON" train.py \
  --train_tasks "$TASK" \
  --use_text_condition \
  --fewshot_text_only_finetune \
  --load_diffusion_checkpoint "$BASE_CKPT" \
  --train_epochs "$TEPOCHS"

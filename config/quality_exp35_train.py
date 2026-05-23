# -*- coding: utf-8 -*-
"""
Quality Experiment 3.5 — few-shot finetune budget (reuse v3 main quartet ckpts).

English doc: Lower LR and shorter FS steps to reduce overfitting on small D_test tails.
中文注释: 主训仍用 quality_exp1_train；本文件仅覆盖 FS 超参。
"""
from __future__ import annotations

from config.quality_exp1_train import (
    BATCH_SIZE,
    CONTEXT_LENGTH_FEWSHOT,
    CONTEXT_LENGTH_TRAIN,
    GRAD_ACCUM,
    LOG_FREQ,
    LR,
    N_TRAIN_STEPS,
    SAVE_FREQ,
)

# Main train unchanged vs v3 (not used when SKIP_MAIN_TRAIN=1).
N_TRAIN_STEPS = N_TRAIN_STEPS

# Few-shot: gentler than v3 (2e-4 × 10k steps); sweep uses FINETUNE_STEP_MILESTONES.
N_FINETUNE_STEPS: int = 2_500
FINETUNE_LR: float = 5e-5
# 中文注释: 从主训 ckpt 起各训 M 步后评估；fs10/20/50 共用同一 LR 与步数网格
FINETUNE_STEP_MILESTONES: tuple[int, ...] = (500, 1000, 1500, 2000, 2500)

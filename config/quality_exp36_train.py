# -*- coding: utf-8 -*-
"""
Quality Experiment 3.6 — same FS sweep protocol as v3.5, lower LR (reuse v3 main ckpts).

English doc: Milestones 500..2500 with eval each; only ``FINETUNE_LR`` differs from exp3.5.
中文注释: 与 v3.5 相同步数网格与 2500 总预算语义；结果目录 quality_bundle_3_6/fs_step_sweep。
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

N_TRAIN_STEPS = N_TRAIN_STEPS

# Same schedule as v3.5; LR lowered (5e-5 -> 1e-5).
N_FINETUNE_STEPS: int = 2_500
FINETUNE_LR: float = 1e-5
FINETUNE_STEP_MILESTONES: tuple[int, ...] = (500, 1000, 1500, 2000, 2500)

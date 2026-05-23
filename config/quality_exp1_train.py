# -*- coding: utf-8 -*-
"""
Quality Experiment 1 — quartet (st_duo / st_text / mt_label / mt_text) + few-shot finetune.

English doc: Same role as ``ant_config.Config`` training fields (``n_train_steps``, ``batch_size``,
``context_length``, …) for code paths that do not use ParamsProto. The shell pipeline and
``train_exp1_checkpoints`` read defaults from here; edit this file to change budgets and optimizer.
"""
from __future__ import annotations

from config import exp1_diffusion_aligned as _e1a

# ----- Main training (all four variants share the same step budget) -----
N_TRAIN_STEPS: int = 50_000
# Few-shot finetune (v3.5 sweep best: 500 steps @ 5e-5; was 10k @ 2e-4)
N_FINETUNE_STEPS: int = 500
FINETUNE_LR: float = 5e-5

BATCH_SIZE: int = 32
LR: float = 2e-4
GRAD_ACCUM: int = 2

CONTEXT_LENGTH_TRAIN: int = 32
CONTEXT_LENGTH_FEWSHOT: int = 16

# ----- Trainer IO (aligned with ``exp1_diffusion_aligned`` / ant line) -----
LOG_FREQ: int = _e1a.MAIN_LOG_FREQ
SAVE_FREQ: int = _e1a.MAIN_SAVE_FREQ

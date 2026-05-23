# -*- coding: utf-8 -*-
"""
Quality Experiment 5 — v4 FS budget + simple proxy filter (train + eval).

English doc: Same finetune as v3.5/v4 (500 steps @ 5e-5). Adds ensemble MLP proxy on
training PKL points; evaluation picks the proxy-best decoded trajectory (not min over
all samples at the last denoise step only).
"""
from __future__ import annotations

from config import quality_exp1_train as _base

# Inherit v4 training budget
N_TRAIN_STEPS: int = _base.N_TRAIN_STEPS
N_FINETUNE_STEPS: int = _base.N_FINETUNE_STEPS
FINETUNE_LR: float = _base.FINETUNE_LR
BATCH_SIZE: int = _base.BATCH_SIZE
LR: float = _base.LR
GRAD_ACCUM: int = _base.GRAD_ACCUM
CONTEXT_LENGTH_TRAIN: int = _base.CONTEXT_LENGTH_TRAIN
CONTEXT_LENGTH_FEWSHOT: int = _base.CONTEXT_LENGTH_FEWSHOT
LOG_FREQ: int = _base.LOG_FREQ
SAVE_FREQ: int = _base.SAVE_FREQ

# ----- Simple proxy (aligned with comparisonExperiment duo_train_and_sample defaults) -----
USE_PROXY_FILTER: int = 1
PROXY_N_TRAIN_STEPS: int = 1000
PROXY_LR: float = 2e-4
PROXY_HIDDEN_DIM: int = 256
PROXY_N_ENSEMBLES: int = 5

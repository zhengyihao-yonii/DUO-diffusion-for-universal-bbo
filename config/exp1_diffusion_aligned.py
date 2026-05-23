# -*- coding: utf-8 -*-
"""
Diffusion / U-Net hyperparameters aligned with ``config/ant_config.py`` (main DUO train path).

English doc: QualityExperiment and comparison exp1 trainers should use these so checkpoints
match ``scripts/train.py`` + ant defaults (except dataset / horizon / context_length, set per run).
"""
from __future__ import annotations

N_DIFFUSION_STEPS: int = 200
N_SAMPLE_TIMESTEPS: int = 200
UNET_DIM: int = 128
UNET_DIM_MULTS: tuple[int, ...] = (1, 4, 8)
UNET_CONDITION_DROPOUT: float = 0.25
LOSS_TYPE: str = "l2"
CLIP_DENOISED: bool = True
PREDICT_EPSILON: bool = True
ACTION_WEIGHT: float = 10.0
CONDITION_GUIDANCE_W: float = 1.2
CONDITION_GUIDANCE_W_TASK: float = 0.0
CONDITION_GUIDANCE_W_TEXT: float = 0.0
# ant_config: log_freq / save_freq
MAIN_LOG_FREQ: int = 50
MAIN_SAVE_FREQ: int = 5000

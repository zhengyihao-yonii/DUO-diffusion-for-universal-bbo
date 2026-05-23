# -*- coding: utf-8 -*-
"""
Diffusion / U-Net layout for QualityExperiment **trace loading** (must match saved checkpoints).

English doc: ``train_exp1_checkpoints`` builds models from ``config.exp1_diffusion_aligned``.
This module **re-exports** those same constants so ``trace_sampling`` / ``run_landscape_experiment``
never drift (historical copies of ``(1,2,4)`` + 1000 steps caused ``load_state_dict`` channel
mismatches vs ant-aligned training).

中文注释: 评估侧须与训练侧共用 ``exp1_diffusion_aligned``，勿在此单独写死 mults / 扩散步数。
"""
from __future__ import annotations

from config import exp1_diffusion_aligned as _e1a

N_TIMESTEPS: int = _e1a.N_DIFFUSION_STEPS
N_SAMPLE_TIMESTEPS: int = _e1a.N_SAMPLE_TIMESTEPS
UNET_DIM: int = _e1a.UNET_DIM
UNET_DIM_MULTS: tuple[int, ...] = _e1a.UNET_DIM_MULTS
UNET_CONDITION_DROPOUT: float = _e1a.UNET_CONDITION_DROPOUT
LOSS_TYPE: str = _e1a.LOSS_TYPE
CLIP_DENOISED: bool = _e1a.CLIP_DENOISED
PREDICT_EPSILON: bool = _e1a.PREDICT_EPSILON
ACTION_WEIGHT: float = _e1a.ACTION_WEIGHT
CONDITION_GUIDANCE_W: float = _e1a.CONDITION_GUIDANCE_W
CONDITION_GUIDANCE_W_TASK: float = _e1a.CONDITION_GUIDANCE_W_TASK

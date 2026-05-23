# -*- coding: utf-8 -*-
"""
QualityExperiment-only overrides for text-axis classifier-free guidance.

English doc: Main ant-aligned defaults keep ``condition_guidance_w_text=0``; Quality runs
use a stronger text CFG during **sampling** (and store the same value on ``GaussianDiffusion``
during training for consistency). Adjust ``CONDITION_GUIDANCE_W_TEXT`` here only.
"""
from __future__ import annotations

# 中文注释: 文本条件 CFG 强度（采样时 epsilon_task_text_cfg 使用的 wx）
CONDITION_GUIDANCE_W_TEXT: float = 8.0

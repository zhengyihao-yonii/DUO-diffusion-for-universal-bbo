# -*- coding: utf-8 -*-
"""Per-dimension observation noise profiles for exp1 native design vectors."""
from __future__ import annotations

import numpy as np


def noise_std_per_dim(
    d_x: int,
    *,
    rng: np.random.Generator,
    base_std: float = 0.03,
    n_high_noise_dims: int = 1,
    high_std: float = 0.40,
) -> np.ndarray:
    """
    English doc: Most native dims get ``base_std``; ``n_high_noise_dims`` dims get ``high_std``
    (nuisance / weakly-informative coordinates).
    """
    dx = int(d_x)
    std = np.full((dx,), float(base_std), dtype=np.float64)
    nh = int(max(0, n_high_noise_dims))
    if nh > 0 and dx > 0:
        pick = rng.choice(dx, size=min(nh, dx), replace=False)
        std[pick] = float(high_std)
    return std

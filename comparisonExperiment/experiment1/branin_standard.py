# -*- coding: utf-8 -*-
"""
Standard Branin benchmark (common analytical form and plot ranges).

English doc: f(x1, x2) with x1 in [-5, 10], x2 in [0, 15] as in most Branin papers.
Latent z in [-1, 1]^2 is affinely mapped to (x1, x2) for f(z) and landscape plots only.
Native design vectors x = A z + b use separate random A (heterogeneous d_x).
"""
from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch

# Standard Branin constants (a=1, b=5.1/(4π²), c=5/π, r=6, s=10, t=1/(8π))
BRANIN_A: float = 1.0
BRANIN_B: float = 5.1 / (4.0 * math.pi**2)
BRANIN_C: float = 5.0 / math.pi
BRANIN_R: float = 6.0
BRANIN_S: float = 10.0
BRANIN_T: float = 1.0 / (8.0 * math.pi)

BRANIN_X1_MIN: float = -5.0
BRANIN_X1_MAX: float = 10.0
BRANIN_X2_MIN: float = 0.0
BRANIN_X2_MAX: float = 15.0

# z_i ∈ [-1, 1]  ->  Branin coordinates x1, x2 on the usual plot box
X1_SCALE: float = 7.5
X1_BIAS: float = 2.5
X2_SCALE: float = 7.5
X2_BIAS: float = 7.5


def normalize_branin_domain(value: str) -> str:
    """Map CLI/meta aliases to ``legacy`` or ``standard``."""
    v = str(value).strip().lower()
    if v in ("standard", "gtg", "paper"):
        return "standard"
    if v == "legacy":
        return "legacy"
    raise ValueError(f"branin_domain must be legacy|standard, got {value!r}")


def latent_to_branin_xy(z: Union[torch.Tensor, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Map latent z to Branin plot coordinates (x1, x2); not native observation x."""
    if isinstance(z, np.ndarray):
        zt = torch.from_numpy(np.asarray(z, dtype=np.float32))
    else:
        zt = z
    x1 = zt[..., 0] * float(X1_SCALE) + float(X1_BIAS)
    x2 = zt[..., 1] * float(X2_SCALE) + float(X2_BIAS)
    return x1, x2


def branin_xy(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Evaluate standard Branin; lower is better (minimization)."""
    inner = x2 - BRANIN_B * x1**2 + BRANIN_C * x1 - BRANIN_R
    return (
        BRANIN_A * inner**2
        + BRANIN_S * (1.0 - BRANIN_T) * torch.cos(x1)
        + BRANIN_S
    )


def branin_from_latent_coords(z: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Branin f(z) with z in [-1, 1]^2 mapped to (x1, x2) on the usual Branin box."""
    x1, x2 = latent_to_branin_xy(z)
    return branin_xy(x1, x2)


def branin_grid(
    *,
    x1: tuple[float, float] = (BRANIN_X1_MIN, BRANIN_X1_MAX),
    x2: tuple[float, float] = (BRANIN_X2_MIN, BRANIN_X2_MAX),
    n: int = 140,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contour grid on Branin coordinates (x1, x2) and objective values F."""
    x1a, x1b = x1
    x2a, x2b = x2
    u = np.linspace(float(x1a), float(x1b), int(n))
    v = np.linspace(float(x2a), float(x2b), int(n))
    x1g, x2g = np.meshgrid(u, v, indexing="xy")
    xt1 = torch.from_numpy(x1g.astype(np.float32))
    xt2 = torch.from_numpy(x2g.astype(np.float32))
    f = branin_xy(xt1, xt2).detach().cpu().numpy()
    return x1g, x2g, f


def branin_gradient_xy(
    x1: np.ndarray, x2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic ∂f/∂x1, ∂f/∂x2 on Branin coordinates (optional quiver overlays)."""
    x1t = torch.from_numpy(np.asarray(x1, dtype=np.float32))
    x2t = torch.from_numpy(np.asarray(x2, dtype=np.float32))
    inner = x2t - BRANIN_B * x1t**2 + BRANIN_C * x1t - BRANIN_R
    df_dx1 = 2.0 * inner * (-2.0 * BRANIN_B * x1t + BRANIN_C) - BRANIN_S * (
        1.0 - BRANIN_T
    ) * torch.sin(x1t)
    df_dx2 = 2.0 * inner
    return df_dx1.detach().cpu().numpy(), df_dx2.detach().cpu().numpy()

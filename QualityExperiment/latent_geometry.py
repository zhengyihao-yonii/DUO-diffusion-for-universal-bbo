# -*- coding: utf-8 -*-
"""
Project observation x back to latent z and evaluate shared objectives (Branin / Ackley).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from comparisonExperiment.experiment1.task_family import (
        InstanceTransform,
        LatentObjective,
    )


def x_to_z_least_squares(
    A: np.ndarray, b: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """
    Given x ≈ z @ A.T + b with A:[d_x,d_z], recover z:[...,d_z] by linear least squares.

    English doc: Works when d_z <= d_x and A has full column rank.
    """
    a = np.asarray(A, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim == 1:
        xx = xx.reshape(1, -1)
    # x - b = z @ A.T  =>  (x-b) = z @ A.T  => z^T = A^+ (x-b)^T ... For each row:
    # x_row - b = z_row @ A.T  =>  (x_row - b) = A @ z_row in R^{d_x}  wait
    # map: x = z @ A.T + b  with z row vector [1,d_z], A [d_x, d_z]
    # x^T = A @ z^T + b  => A @ z^T = x^T - b
    rhs = (xx - bb).T  # [d_x, N]
    zt, _, rank, _ = np.linalg.lstsq(a, rhs, rcond=None)
    if int(rank) < a.shape[1]:
        # 中文注释: 秩不足时仍返回最小二乘解，供可视化用
        pass
    return np.asarray(zt.T, dtype=np.float64)  # [N, d_z]


def x_to_z_from_instance_transform(
    transform: Any, x: np.ndarray, *, as_numpy: bool = True
) -> np.ndarray:
    """Use ``InstanceTransform`` A,b tensors from task_family."""
    A = transform.A.detach().cpu().numpy()
    b = transform.b.detach().cpu().numpy()
    out = x_to_z_least_squares(A, b, x)
    return out.astype(np.float32) if as_numpy else out


def latent_objective_grid(
    objective: Any,
    *,
    z1: tuple[float, float] = (-1.0, 1.0),
    z2: tuple[float, float] = (-1.0, 1.0),
    n: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate latent objective on a 2D grid (requires objective.d_z == 2).

    Returns (Z1, Z2, F) each shaped [n,n]; F is raw objective (lower is better for Branin).
    """
    z1a, z1b = z1
    z2a, z2b = z2
    u = np.linspace(z1a, z1b, int(n))
    v = np.linspace(z2a, z2b, int(n))
    Z1, Z2 = np.meshgrid(u, v, indexing="xy")
    pts = np.stack([Z1.ravel(), Z2.ravel()], axis=-1)
    zt = torch.from_numpy(pts.astype(np.float32))
    f = objective.eval(zt).detach().cpu().numpy().reshape(Z1.shape)
    return Z1, Z2, f


def branin_grad_xy(z1: np.ndarray, z2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Analytic gradient of Branin (internal coords z in [-1,1]) for background quiver / sketch.

    English doc: Matches LatentObjective(branin).eval mapping in task_family.
    """
    import math

    x1 = z1 * 5.0 + 2.5
    x2 = z2 * 7.5 + 7.5
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    inner = x2 - b * x1**2 + c * x1 - r
    df_dx1 = (
        2.0 * inner * (-2.0 * b * x1 + c)
        - s * (1.0 - t) * math.sin(x1)
    )
    df_dx2 = 2.0 * inner
    dz1_dx1 = 1.0 / 5.0
    dz2_dx2 = 1.0 / 7.5
    g_z1 = df_dx1 * dz1_dx1
    g_z2 = df_dx2 * dz2_dx2
    return g_z1, g_z2

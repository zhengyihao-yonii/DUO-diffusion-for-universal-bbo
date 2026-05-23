# -*- coding: utf-8 -*-
"""
Greedy selection of denoise trajectories that land in spatially separated basins.

English doc: Sort by objective (lower better), then greedily keep trajectories whose
final z is at least ``min_z_sep`` Euclidean away from all already selected — favours
diverse convergence regions without clustering dependency.
"""
from __future__ import annotations

import numpy as np


def pick_diverse_basin_representatives(
    z_final: np.ndarray,
    raw_f: np.ndarray,
    *,
    max_count: int,
    min_z_sep: float,
) -> list[int]:
    """
    Return trajectory indices (batch axis) to plot.

    English doc: ``z_final`` [B,2], ``raw_f`` [B] — lower is better.
    """
    zf = np.asarray(z_final, dtype=np.float64)
    rf = np.asarray(raw_f, dtype=np.float64).reshape(-1)
    b = int(zf.shape[0])
    if b == 0 or max_count <= 0:
        return []
    order = np.argsort(rf, axis=0)
    picked: list[int] = []
    for i in order:
        bi = int(i)
        if len(picked) >= int(max_count):
            break
        z = zf[bi]
        ok = True
        for pj in picked:
            if float(np.linalg.norm(z - zf[pj])) < float(min_z_sep):
                ok = False
                break
        if ok:
            picked.append(bi)
    if not picked:
        picked = [int(order[0])]
    return picked

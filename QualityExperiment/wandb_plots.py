# -*- coding: utf-8 -*-
"""
Matplotlib landscape panels + wandb logging (Image + Table for per-trajectory selection).

English doc: Uses non-interactive Agg backend; safe for headless servers.
"""
from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


def plot_latent_panel(
    *,
    z1_grid: np.ndarray,
    z2_grid: np.ndarray,
    f_grid: np.ndarray,
    traces: dict[str, np.ndarray],
    title: str,
    elev_clip_percentile: float = 99.0,
) -> plt.Figure:
    """
    Contour-fill of objective values on (z1,z2) plus overlaid denoise trajectories.

    English doc: Same visual language as RGDiff-style figures — **color = objective level**
    (basins as regions), trajectories overlaid; no separate gradient-vector plot.

    ``traces[method]`` shape ``[S, B, 2]`` — plots each batch member as a polyline in z-space.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=120)
    fc = np.asarray(f_grid, dtype=np.float64)
    hi = np.percentile(fc[np.isfinite(fc)], float(elev_clip_percentile))
    lo = np.percentile(fc[np.isfinite(fc)], 100.0 - float(elev_clip_percentile))
    cf = ax.contourf(
        z1_grid,
        z2_grid,
        fc,
        levels=28,
        cmap="viridis",
        vmin=float(lo),
        vmax=float(hi),
    )
    fig.colorbar(
        cf,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label=r"objective $f(\mathbf{z})$ (lower is better)",
    )
    colors = ("white", "cyan", "orange", "magenta", "yellow")
    for mi, (name, zpath) in enumerate(traces.items()):
        zpath = np.asarray(zpath, dtype=np.float64)
        if zpath.ndim != 3 or zpath.shape[-1] != 2:
            raise ValueError(f"{name}: expected [S,B,2], got {zpath.shape}")
        _s, b, _two = zpath.shape
        c = colors[mi % len(colors)]
        for bi in range(b):
            xs = zpath[:, bi, 0]
            ys = zpath[:, bi, 1]
            alpha = 0.35 if b > 8 else 0.85
            lw = 0.8 if b > 8 else 1.4
            ax.plot(xs, ys, color=c, alpha=alpha, linewidth=lw)
            ax.scatter(xs[-1], ys[-1], color=c, s=12, marker="o", alpha=min(1.0, alpha + 0.2))
        ax.plot([], [], color=c, label=name, linewidth=2.0)
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def figure_to_wandb_image(fig: plt.Figure) -> Any:
    """Return ``wandb.Image`` from a matplotlib figure."""
    import wandb

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return wandb.Image(buf)


def log_trajectory_table(
    *,
    method: str,
    z_steps: np.ndarray,
    raw_f_final: np.ndarray | None,
    table_key: str = "traj_pick_table",
) -> None:
    """
    One row per batch trajectory with optional scalar ``raw_f_final`` for sorting in wandb UI.

    ``z_steps``: [S,B,2]; ``raw_f_final``: [B] Branin raw at final z (optional).
    """
    import wandb

    z_steps = np.asarray(z_steps, dtype=np.float64)
    _, b, _ = z_steps.shape
    tbl = wandb.Table(columns=["method", "traj_id", "final_z1", "final_z2", "raw_f", "thumb"])

    for bi in range(b):
        zf = z_steps[-1, bi]
        rf = float(raw_f_final[bi]) if raw_f_final is not None else float("nan")
        fig, ax = plt.subplots(figsize=(3.2, 2.8), dpi=100)
        ax.plot(z_steps[:, bi, 0], z_steps[:, bi, 1], color="C0", linewidth=1.5)
        ax.scatter(zf[0], zf[1], color="red", s=18)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{method} traj {bi}")
        tbl.add_data(method, bi, float(zf[0]), float(zf[1]), rf, figure_to_wandb_image(fig))
    wandb.log({table_key: tbl})


def log_combined_trajectory_table(
    *,
    rows: list[tuple[str, np.ndarray, np.ndarray]],
    table_key: str = "table/all_methods",
) -> None:
    """
    Single wandb Table for cross-method filtering: columns
    method, traj_id, final_z1, final_z2, raw_f, thumb.

    ``rows`` is a list of ``(method_name, z_steps[S,B,2], raw_f_final[B])`` per enabled method.
    """
    import wandb

    tbl = wandb.Table(
        columns=["method", "traj_id", "final_z1", "final_z2", "raw_f", "thumb"]
    )
    for method, z_steps, raw_f_final in rows:
        z_steps = np.asarray(z_steps, dtype=np.float64)
        rf = np.asarray(raw_f_final, dtype=np.float64).reshape(-1)
        _, b, _ = z_steps.shape
        for bi in range(b):
            zf = z_steps[-1, bi]
            fig, ax = plt.subplots(figsize=(3.2, 2.8), dpi=100)
            ax.plot(z_steps[:, bi, 0], z_steps[:, bi, 1], color="C0", linewidth=1.5)
            ax.scatter(zf[0], zf[1], color="red", s=18)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{method} #{bi}")
            tbl.add_data(
                method,
                bi,
                float(zf[0]),
                float(zf[1]),
                float(rf[bi]),
                figure_to_wandb_image(fig),
            )
    wandb.log({table_key: tbl})

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

from QualityExperiment.trajectory_selection import pick_diverse_basin_representatives


def filter_traces_for_landscape(
    traces: dict[str, np.ndarray],
    raw_f_by_method: dict[str, np.ndarray],
    *,
    max_rep_trajs: int,
    min_z_sep: float,
) -> dict[str, np.ndarray]:
    """English doc: Keep best-f trajectories per method, greedily separated in final ``z``."""
    out: dict[str, np.ndarray] = {}
    for name, zpath in traces.items():
        zpath = np.asarray(zpath, dtype=np.float64)
        if zpath.ndim != 3 or zpath.shape[-1] != 2:
            raise ValueError(f"{name}: expected [S,B,2], got {zpath.shape}")
        rf = raw_f_by_method.get(name)
        if rf is None:
            picked = list(range(min(int(zpath.shape[1]), int(max_rep_trajs))))
        else:
            picked = pick_diverse_basin_representatives(
                zpath[-1],
                np.asarray(rf, dtype=np.float64),
                max_count=int(max_rep_trajs),
                min_z_sep=float(min_z_sep),
            )
        out[name] = zpath[:, picked, :]
    return out


def plot_branin_physical_panel(
    *,
    x1_grid: np.ndarray,
    x2_grid: np.ndarray,
    f_grid: np.ndarray,
    traces_xy: dict[str, np.ndarray],
    title: str,
    method_label: str = "",
    elev_clip_percentile: float = 99.0,
) -> plt.Figure:
    """
    Branin landscape on coordinates (x1, x2) from latent z, one method per figure.

    ``traces_xy[method]`` shape ``[S, B, 2]`` — Branin (x1, x2) via ``latent_to_branin_xy``.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=120)
    fc = np.asarray(f_grid, dtype=np.float64)
    hi = np.percentile(fc[np.isfinite(fc)], float(elev_clip_percentile))
    lo = np.percentile(fc[np.isfinite(fc)], 100.0 - float(elev_clip_percentile))
    cf = ax.contourf(
        x1_grid,
        x2_grid,
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
        label=r"objective $f(x_1, x_2)$ (lower is better)",
    )
    c = "white"
    for name, xypath in traces_xy.items():
        xypath = np.asarray(xypath, dtype=np.float64)
        if xypath.ndim != 3 or xypath.shape[-1] != 2:
            raise ValueError(f"{name}: expected [S,B,2], got {xypath.shape}")
        _s, b, _two = xypath.shape
        for bi in range(b):
            xs = xypath[:, bi, 0]
            ys = xypath[:, bi, 1]
            alpha = 0.35 if b > 8 else 0.85
            lw = 0.8 if b > 8 else 1.4
            pt_sz = 10 if b > 8 else 22
            ax.plot(xs, ys, color=c, alpha=alpha, linewidth=lw, linestyle="-")
            ax.scatter(
                xs,
                ys,
                color=c,
                s=pt_sz,
                alpha=min(1.0, alpha + 0.05),
                edgecolors="none",
                zorder=3,
            )
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    lab = str(method_label).strip() or next(iter(traces_xy.keys()), "")
    ax.set_title(f"{title}\n({lab})" if lab else title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


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

    ``traces[method]`` shape ``[S, B, 2]`` — one polyline per batch index **B** (caller should
    pre-filter to representative trajectories; see ``filter_traces_for_landscape``).
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
            pt_sz = 10 if b > 8 else 22
            ax.plot(xs, ys, color=c, alpha=alpha, linewidth=lw, linestyle="-")
            ax.scatter(
                xs,
                ys,
                color=c,
                s=pt_sz,
                alpha=min(1.0, alpha + 0.05),
                edgecolors="none",
                zorder=3,
            )
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
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    # Chinese comment: wandb>=0.24 不接受裸 BytesIO，需 PIL/numpy/路径。
    pil_img = Image.open(buf).convert("RGB")
    return wandb.Image(pil_img)


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
        ax.plot(
            z_steps[:, bi, 0], z_steps[:, bi, 1], color="C0", linewidth=1.2, linestyle="-"
        )
        ax.scatter(
            z_steps[:, bi, 0],
            z_steps[:, bi, 1],
            color="C0",
            s=20,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
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
            ax.plot(
                z_steps[:, bi, 0],
                z_steps[:, bi, 1],
                color="C0",
                linewidth=1.2,
                linestyle="-",
            )
            ax.scatter(
                z_steps[:, bi, 0],
                z_steps[:, bi, 1],
                color="C0",
                s=20,
                alpha=0.9,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replot superconductor seed-0 sample_viz curves from wandb (dfgo project).

English doc: Fetch aggregated runs ``duo_viz_superconductor_ep{250..1500}_sample_viz_mean``,
plot mean / top-8 mean / max of y_norm vs diffusion time t in [1, 0], with W&B TWEMA smoothing.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MetricKind = Literal["mean", "top8", "max"]

# English doc: fixed colors across all figures (colorblind-friendly tab10 subset).
MODEL_SPECS: tuple[tuple[str, str, str], ...] = (
    ("st_duo", "st", "#1f77b4"),
    ("st_text", "st+text", "#ff7f0e"),
    ("mt_task", "mt+label", "#2ca02c"),
    ("mt_text", "mt+text", "#d62728"),
)

METRIC_SUFFIX: dict[MetricKind, str] = {
    "mean": "mean_y_norm",
    "top8": "top8_mean_norm",
    "max": "max_y_norm",
}

METRIC_TITLE: dict[MetricKind, str] = {
    "mean": "mean",
    "top8": "top8mean",
    "max": "max",
}

DEFAULT_EPOCHS: tuple[int, ...] = (250, 500, 750, 1000, 1250, 1500)


@dataclass(frozen=True)
class CurveSeries:
    """One model curve for a single epoch + metric."""

    label: str
    color: str
    t: np.ndarray
    y_raw: np.ndarray
    y_plot: np.ndarray


def _wandb_twema(
    x: np.ndarray,
    y: np.ndarray,
    *,
    smoothing_param: float,
) -> np.ndarray:
    """
    W&B time-weighted EMA (TWEMA) with debias.

    English doc: Follows https://docs.wandb.ai/models/app/features/panels/line-plot/smoothing
    with ``smoothingWeight = min(sqrt(p), 0.999)``. ``changeInX`` is normalized by mean point
    spacing so uniform series get the same effective smoothing regardless of x-axis scale
    (density-aware TWEMA; matches W&B when switching between ``sample_viz_step`` and ``t``).
    """
    if y.size == 0:
        return y
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    p = float(min(max(smoothing_param, 0.0), 1.0))
    if p <= 0.0:
        return y_arr.copy()
    smoothing_weight = min(float(np.sqrt(p)), 0.999)
    range_of_x = float(np.max(x_arr) - np.min(x_arr))
    if range_of_x <= 0.0:
        range_of_x = 1.0
    n = int(y_arr.size)
    mean_spacing = range_of_x / float(max(n - 1, 1))
    last_y = 0.0
    debias_weight = 0.0
    out = np.empty_like(y_arr, dtype=np.float64)
    for i, y_point in enumerate(y_arr):
        if i == 0:
            change_in_x = 0.0
        else:
            # 中文注释: 用 |Δx|/平均间距，使 t∈[1,0] 与 step∈[0,40] 每步 changeInX≈1
            change_in_x = abs(float(x_arr[i]) - float(x_arr[i - 1])) / mean_spacing
        w_adj = smoothing_weight ** change_in_x
        last_y = last_y * w_adj + float(y_point)
        debias_weight = debias_weight * w_adj + 1.0
        out[i] = last_y / debias_weight
    return out


def _tag_name(model_base: str, epoch: int) -> str:
    return f"{model_base}_ep{epoch}"


def _run_name(epoch: int, *, prefix: str) -> str:
    return f"{prefix}_ep{epoch}_sample_viz_mean"


def _fetch_run(api: object, project: str, run_name: str) -> object:
    import wandb

    path = f"{project}/{run_name}"
    try:
        return api.run(path)
    except Exception:
        runs = list(api.runs(project, filters={"display_name": run_name}))
        if not runs:
            runs = [r for r in api.runs(project, per_page=200) if r.name == run_name]
        if not runs:
            raise SystemExit(f"[error] wandb run not found: {path}")
        return runs[0]


def _extract_series(
    hist: pd.DataFrame,
    *,
    tag: str,
    value_suffix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_step, t in [1,0], y_norm) in denoise / sample_viz_step order."""
    y_col = f"sample_viz/{tag}/{value_suffix}"
    t_col = f"sample_viz/{tag}/t_index"
    x_col = "sample_viz_step"
    if y_col not in hist.columns or t_col not in hist.columns:
        raise SystemExit(f"[error] missing columns for tag={tag}: {y_col}")
    sub = hist[[x_col, t_col, y_col]].dropna(subset=[y_col, t_col])
    if sub.empty:
        raise SystemExit(f"[error] empty series for tag={tag}")
    sub = sub.sort_values(x_col, kind="mergesort")
    x_step = sub[x_col].to_numpy(dtype=np.float64)
    t_idx = sub[t_col].to_numpy(dtype=np.float64)
    y = sub[y_col].to_numpy(dtype=np.float64)
    t_max = float(np.max(t_idx))
    if t_max <= 0:
        raise SystemExit(f"[error] invalid t_index max for tag={tag}: {t_max}")
    t = t_idx / t_max
    return x_step, t, y


def _smooth_y(
    x: np.ndarray,
    y: np.ndarray,
    *,
    smoothing_param: float,
) -> np.ndarray:
    """English doc: TWEMA on x (``sample_viz_step`` or diffusion time ``t``)."""
    if smoothing_param <= 0.0:
        return y.copy()
    return _wandb_twema(x, y, smoothing_param=smoothing_param)


def _load_curves_for_epoch(
    api: object,
    *,
    project: str,
    run_prefix: str,
    epoch: int,
    metric: MetricKind,
    smoothing_param: float,
    smooth_x: Literal["step", "t"],
) -> list[CurveSeries]:
    run = _fetch_run(api, project, _run_name(epoch, prefix=run_prefix))
    hist = run.history(pandas=True)
    suffix = METRIC_SUFFIX[metric]
    curves: list[CurveSeries] = []
    for model_base, legend, color in MODEL_SPECS:
        tag = _tag_name(model_base, epoch)
        x_step, t, y_raw = _extract_series(hist, tag=tag, value_suffix=suffix)
        x_smooth = x_step if smooth_x == "step" else t
        y_plot = _smooth_y(x_smooth, y_raw, smoothing_param=smoothing_param)
        curves.append(
            CurveSeries(
                label=legend,
                color=color,
                t=t,
                y_raw=y_raw,
                y_plot=y_plot,
            )
        )
    return curves


def _global_ylim(
    api: object,
    *,
    project: str,
    run_prefix: str,
    epochs: tuple[int, ...],
    metric: MetricKind,
    smoothing_param: float,
    use_raw: bool,
    smooth_x: Literal["step", "t"],
) -> tuple[float, float]:
    vals: list[float] = []
    for ep in epochs:
        for curve in _load_curves_for_epoch(
            api,
            project=project,
            run_prefix=run_prefix,
            epoch=ep,
            metric=metric,
            smoothing_param=smoothing_param if not use_raw else 0.0,
            smooth_x=smooth_x,
        ):
            src = curve.y_raw if use_raw else curve.y_plot
            vals.extend(src.tolist())
    if not vals:
        return 0.0, 1.0
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    pad = max(1e-3, 0.03 * (hi - lo))
    return lo - pad, hi + pad


def _plot_one(
    curves: list[CurveSeries],
    *,
    title: str,
    out_path: Path,
    ylim: tuple[float, float],
    show_raw: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for curve in curves:
        if show_raw:
            ax.plot(
                curve.t,
                curve.y_raw,
                color=curve.color,
                linewidth=1.0,
                alpha=0.35,
            )
        ax.plot(
            curve.t,
            curve.y_plot,
            color=curve.color,
            linewidth=2.2,
            label=curve.label,
        )
    ax.set_xlim(1.0, 0.0)
    ax.set_xlabel(r"diffusion time $t$ (1 $\rightarrow$ 0)")
    ax.set_ylabel(r"$y_{\mathrm{norm}}$")
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_title(title, fontsize=13, pad=28)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replot superconductor sample_viz curves from wandb dfgo runs.",
    )
    ap.add_argument("--project", type=str, default="1585515136-/dfgo")
    ap.add_argument(
        "--run_prefix",
        type=str,
        default="duo_viz_superconductor",
        help="Run name prefix before _ep{epoch}_sample_viz_mean",
    )
    ap.add_argument(
        "--epochs",
        type=str,
        default=",".join(str(e) for e in DEFAULT_EPOCHS),
    )
    ap.add_argument("--out_dir", type=str, default="results/figures")
    ap.add_argument(
        "--smoothing",
        type=float,
        default=0.99,
        help="W&B TWEMA smoothing (0=raw only; 0.99 matches wandb slider).",
    )
    ap.add_argument(
        "--show-raw",
        dest="show_raw",
        action="store_true",
        default=False,
        help="Overlay faint raw curve (wandb 'Show Original').",
    )
    ap.add_argument(
        "--ylim-raw",
        dest="ylim_raw",
        action="store_true",
        default=False,
        help="Shared y-limits from raw values (default: from smoothed curves).",
    )
    ap.add_argument(
        "--smooth-x",
        type=str,
        choices=("step", "t"),
        default="step",
        help="X-axis used inside TWEMA: sample_viz_step (wandb default) or t.",
    )
    ap.add_argument(
        "--task",
        type=str,
        default="superconductor",
        help="Subfolder under out_dir (e.g. results/figures/superconductor_seed0).",
    )
    args = ap.parse_args()

    epochs = tuple(int(x.strip()) for x in str(args.epochs).split(",") if x.strip())
    out_root = Path(args.out_dir) / f"{args.task}_seed0"
    smoothing_param = float(min(max(args.smoothing, 0.0), 1.0))
    show_raw = bool(args.show_raw) and smoothing_param > 0.0
    ylim_raw = bool(args.ylim_raw)

    os.environ.setdefault("WANDB_BASE_URL", "https://api.wandb.ai/")
    import wandb

    api = wandb.Api(timeout=60)

    smooth_x = str(args.smooth_x)
    if smooth_x not in ("step", "t"):
        raise SystemExit(f"[error] invalid --smooth-x: {smooth_x}")
    print(f"[info] smooth_x={smooth_x} (TWEMA x-axis; plot x-axis is always t)")

    ylims: dict[MetricKind, tuple[float, float]] = {}
    for metric in ("mean", "top8", "max"):
        ylims[metric] = _global_ylim(
            api,
            project=str(args.project),
            run_prefix=str(args.run_prefix),
            epochs=epochs,
            metric=metric,
            smoothing_param=smoothing_param,
            use_raw=ylim_raw,
            smooth_x=smooth_x,
        )
        print(f"[ylim] {metric}: {ylims[metric][0]:.4f} .. {ylims[metric][1]:.4f}")

    for ep in epochs:
        for metric in ("mean", "top8", "max"):
            curves = _load_curves_for_epoch(
                api,
                project=str(args.project),
                run_prefix=str(args.run_prefix),
                epoch=ep,
                metric=metric,
                smoothing_param=smoothing_param,
                smooth_x=smooth_x,
            )
            fname = f"{METRIC_TITLE[metric]}_ep{ep}.png"
            title = f"{METRIC_TITLE[metric]}_ep{ep}"
            _plot_one(
                curves,
                title=title,
                out_path=out_root / fname,
                ylim=ylims[metric],
                show_raw=show_raw,
            )

    print(f"[done] wrote {len(epochs) * 3} figures under {out_root.resolve()}")


if __name__ == "__main__":
    main()

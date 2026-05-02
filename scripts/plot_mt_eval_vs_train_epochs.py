#!/usr/bin/env python3
"""
Aggregate eval_summary_e*epochs.json under a sweep MODEL_ROOT and plot metrics vs train epoch.

Per task: for each checkpoint epoch, aggregate across seeds with mean ± std (configurable).
Optional: one curve averaging the metric across tasks per seed, then mean ± std across seeds.

Optionally upload figures to Weights & Biases (wandb).

Example:
  cd /data/xk/zyh_dfgo/DUO && python3 scripts/plot_mt_eval_vs_train_epochs.py \\
    --model-root results/epoch1500/mt_911054c35daad7e0_textcond_mttextonly_ce0.005_tsbias0.5_lr0.0002 \\
    --metric ntop8_mean --tasks ant,dkitty \\
    --std-across-seeds stdev \\
    --macro-across-tasks \\
    --wandb-run-name mt_ce005_ckpt_curve
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import Any, DefaultDict

_METRICS_ALL = (
    "nmax",
    "max",
    "mean",
    "median",
    "nmedian",
    "nmean",
    "top8_mean",
    "ntop8_mean",
)


def _gather_jsons(model_root: Path) -> list[Path]:
    # Support both new and legacy naming.
    # New:    eval_summary_w8p0_1500epochs.json
    # Legacy: eval_summary_e1500epochs.json
    out = list(model_root.glob("seed*/eval_summary_w*_?*epochs.json"))
    out += list(model_root.glob("seed*/eval_summary_w*_*.json"))  # be permissive
    out += list(model_root.glob("seed*/eval_summary_e*epochs.json"))
    # Deduplicate and sort for stability.
    return sorted({p.resolve() for p in out})


def _epoch_from_name(p: Path) -> int | None:
    m = re.search(r"eval_summary_e(\d+)epochs\.json$", p.name)
    if m:
        return int(m.group(1))
    m2 = re.search(r"eval_summary_w[^_]+_(\d+)epochs\.json$", p.name)
    return int(m2.group(1)) if m2 else None


def _load_rows(model_root: Path) -> list[tuple[int, int, str, dict[str, Any]]]:
    """rows: (epoch_label, seed, task, task_metrics)"""
    rows: list[tuple[int, int, str, dict[str, Any]]] = []
    for jp in _gather_jsons(model_root):
        ep = _epoch_from_name(jp)
        if ep is None:
            continue
        m = re.search(r"seed(\d+)", jp.parent.name)
        if not m:
            continue
        seed = int(m.group(1))
        data = json.loads(jp.read_text(encoding="utf-8"))
        tasks = data.get("tasks") or {}
        for task, metrics in tasks.items():
            rows.append((ep, seed, str(task), dict(metrics)))
    return rows


def _std_across_seeds(vals: list[float], kind: str) -> float:
    if len(vals) < 2:
        return 0.0
    if kind == "pstdev":
        return float(pstdev(vals))
    if kind == "stdev":
        return float(stdev(vals))
    raise ValueError(f"unknown std kind: {kind!r}")


def _aggregate(
    rows: list[tuple[int, int, str, dict[str, Any]]],
    *,
    metric: str,
    tasks: list[str],
    std_kind: str,
) -> dict[str, dict[int, tuple[float, float]]]:
    """
    task -> epoch -> (mean_metric, std_over_seeds).
    """
    by_t_e_s: DefaultDict[tuple[str, int], list[float]] = defaultdict(list)
    for ep, _seed, task, metrics in rows:
        if task not in tasks:
            continue
        if metric not in metrics:
            continue
        v = float(metrics[metric])
        if not math.isfinite(v):
            continue
        by_t_e_s[(task, ep)].append(v)
    out: dict[str, dict[int, tuple[float, float]]] = {}
    for task in tasks:
        out[task] = {}
        epochs = sorted({e for (t, e) in by_t_e_s if t == task})
        for ep in epochs:
            vals = by_t_e_s[(task, ep)]
            if not vals:
                continue
            mu = float(mean(vals))
            sd = _std_across_seeds(vals, std_kind)
            out[task][ep] = (mu, sd)
    return out


def _aggregate_macro_across_tasks(
    rows: list[tuple[int, int, str, dict[str, Any]]],
    *,
    metric: str,
    tasks: list[str],
    std_kind: str,
) -> dict[int, tuple[float, float]]:
    """
    epoch -> (mean across seeds of macro_mean, std across seeds).

    For each (epoch, seed): macro = arithmetic mean of metric over ``tasks`` (skip missing / non-finite).
    """
    by_ep_seed: DefaultDict[tuple[int, int], list[float]] = defaultdict(list)
    for ep, seed, task, metrics in rows:
        if task not in tasks:
            continue
        if metric not in metrics:
            continue
        v = float(metrics[metric])
        if not math.isfinite(v):
            continue
        by_ep_seed[(ep, seed)].append(v)

    out: dict[int, tuple[float, float]] = {}
    for ep in sorted({e for (e, _s) in by_ep_seed}):
        seed_vals: list[float] = []
        for seed in sorted({s for (e, s) in by_ep_seed if e == ep}):
            vals = by_ep_seed[(ep, seed)]
            if len(vals) != len(tasks):
                # 仅当该 seed 上所有选定任务都有该 metric 时才纳入 macro，避免混用不完整点
                continue
            seed_vals.append(float(mean(vals)))
        if not seed_vals:
            continue
        out[ep] = (float(mean(seed_vals)), _std_across_seeds(seed_vals, std_kind))
    return out


def _plot(
    agg: dict[str, dict[int, tuple[float, float]]],
    *,
    out_dir: Path,
    metric: str,
    title: str,
    std_kind: str,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    _sd_label = "±1 pstdev (across seeds)" if std_kind == "pstdev" else "±1 sample stdev (across seeds)"
    for task, series in agg.items():
        if not series:
            continue
        xs = sorted(series)
        mus = [series[x][0] for x in xs]
        sds = [series[x][1] for x in xs]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, mus, marker="o", label=f"{metric} mean")
        y_lo = [m - s for m, s in zip(mus, sds)]
        y_hi = [m + s for m, s in zip(mus, sds)]
        ax.fill_between(xs, y_lo, y_hi, alpha=0.25, label=_sd_label)
        ax.set_xlabel("checkpoint train epoch (from filename)")
        ax.set_ylabel(metric)
        ax.set_title(f"{title} — {task}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        outp = out_dir / f"curve_{metric}_{task}.png"
        fig.tight_layout()
        fig.savefig(outp, dpi=150)
        plt.close(fig)
        written.append(outp)
    return written


def _plot_macro(
    series: dict[int, tuple[float, float]],
    *,
    out_dir: Path,
    metric: str,
    title: str,
    std_kind: str,
) -> Path | None:
    if not series:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    xs = sorted(series)
    mus = [series[x][0] for x in xs]
    sds = [series[x][1] for x in xs]
    _sd_label = "±1 pstdev (across seeds)" if std_kind == "pstdev" else "±1 sample stdev (across seeds)"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, mus, marker="o", label=f"macro mean({metric}) over tasks")
    y_lo = [m - s for m, s in zip(mus, sds)]
    y_hi = [m + s for m, s in zip(mus, sds)]
    ax.fill_between(xs, y_lo, y_hi, alpha=0.25, label=_sd_label)
    ax.set_xlabel("checkpoint train epoch (from filename)")
    ax.set_ylabel(f"macro {metric}")
    ax.set_title(f"{title} — macro across tasks (mean per seed, then across seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    outp = out_dir / f"curve_{metric}_macro_across_tasks.png"
    fig.tight_layout()
    fig.savefig(outp, dpi=150)
    plt.close(fig)
    return outp


def _maybe_wandb(paths: list[Path], *, project: str, run_name: str) -> None:
    try:
        import wandb
    except ImportError:
        print("[plot] wandb not installed; skip upload", file=sys.stderr)
        return
    run = wandb.init(project=project, name=run_name, job_type="analysis")
    for p in paths:
        run.log({f"figure/{p.stem}": wandb.Image(str(p))})
    run.finish()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument(
        "--metric",
        type=str,
        default="nmax",
        choices=_METRICS_ALL,
        help="Per-task metric from eval_summary JSON (ntop8_mean: normalized top-8 oracle mean).",
    )
    ap.add_argument(
        "--std-across-seeds",
        type=str,
        default="pstdev",
        choices=("pstdev", "stdev"),
        help="pstdev=statistics.pstdev (N divisor); stdev=sample stdev ddof=1 (needs ≥2 seeds).",
    )
    ap.add_argument(
        "--macro-across-tasks",
        action="store_true",
        help="Also plot one curve: mean(metric) over --tasks per seed, then mean±std across seeds.",
    )
    ap.add_argument(
        "--tasks",
        type=str,
        default="ant,dkitty,gtopx2,gtopx3,gtopx4,gtopx6,superconductor,tfbind10,tfbind8",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="PNG output (default: model-root/plots_ckpt_sweep)")
    ap.add_argument("--wandb-project", type=str, default="decdiff-opt")
    ap.add_argument("--wandb-run-name", type=str, default="")
    args = ap.parse_args()

    model_root = args.model_root.resolve()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    rows = _load_rows(model_root)
    if not rows:
        raise SystemExit(f"No eval_summary_e*epochs.json under {model_root}/seed*/")

    agg = _aggregate(
        rows,
        metric=args.metric,
        tasks=tasks,
        std_kind=args.std_across_seeds,
    )
    out_dir = args.out_dir or (model_root / "plots_ckpt_sweep")
    title = model_root.name
    paths = _plot(
        agg,
        out_dir=out_dir,
        metric=args.metric,
        title=title,
        std_kind=args.std_across_seeds,
    )
    print(f"[plot] wrote {len(paths)} per-task PNG(s) under {out_dir}")
    if args.macro_across_tasks:
        macro = _aggregate_macro_across_tasks(
            rows,
            metric=args.metric,
            tasks=tasks,
            std_kind=args.std_across_seeds,
        )
        mp = _plot_macro(
            macro,
            out_dir=out_dir,
            metric=args.metric,
            title=title,
            std_kind=args.std_across_seeds,
        )
        if mp is not None:
            paths.append(mp)
            print(f"[plot] wrote macro PNG: {mp}")
        else:
            print(
                "[plot] macro plot skipped (no epoch had all tasks×seeds with this metric; "
                "re-run eval with current evaluate.py or check JSON keys).",
                file=sys.stderr,
            )
    if args.wandb_run_name.strip():
        _maybe_wandb(paths, project=args.wandb_project, run_name=args.wandb_run_name.strip())


if __name__ == "__main__":
    main()

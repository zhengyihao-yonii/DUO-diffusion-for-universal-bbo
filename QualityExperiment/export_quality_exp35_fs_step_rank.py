#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rank few-shot finetune step milestones by eval objective (mt_text focus).

English doc: Reads ``fs_step_sweep/<shift_tag>/step_<M>/eval/*.npz``; outputs CSV rank + LaTeX grid.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from QualityExperiment.export_quality_latex_table import (
    _TRACE_RE,
    _ingest_one_npz,
    _parse_slug,
    _phase_tex,
    _uniso_dir_for_stem,
)

_FS_PHASES: tuple[str, ...] = (
    "shift_few_shot_tail10p",
    "shift_few_shot_tail20p",
    "shift_few_shot_tail50p",
)


def _escape_tex(s: str) -> str:
    return s.replace("_", r"\_")


def _phase_short(phase: str) -> str:
    m = re.match(r"shift_few_shot_tail(\d+)p", phase)
    return f"fs{m.group(1)}" if m else phase


def _shift_from_tid(tid: str) -> str:
    m = re.match(r"D_test_(sim_low|sim_mid|sim_high)", tid)
    return m.group(1) if m else tid


def _scan_dir(
    eval_dir: Path,
    *,
    uniso_parent: Path,
    train_uniso: Path,
) -> dict[tuple[str, str], dict[str, float]]:
    bucket: dict[tuple[str, str], dict[str, float]] = {}
    for npz in sorted(eval_dir.glob("quality_trace_*.npz")):
        m = _TRACE_RE.match(npz.name)
        if not m:
            continue
        stem, _phase = _parse_slug(m.group(2))
        u = _uniso_dir_for_stem(stem, uniso_parent=uniso_parent, train_uniso=train_uniso)
        _ingest_one_npz(npz, bucket, u)
    return bucket


def _collect(
    sweep_root: Path,
    *,
    uniso_parent: Path,
    train_uniso: Path,
) -> tuple[dict[tuple[int, str, str, str], float], dict[tuple[str, str], float]]:
    cells: dict[tuple[int, str, str, str], float] = {}
    zs_cells: dict[tuple[str, str], float] = {}

    for shift_dir in sorted(p for p in sweep_root.iterdir() if p.is_dir() and p.name.startswith("sim_")):
        zs_dir = shift_dir / "eval_zs"
        if zs_dir.is_dir():
            for (tid, phase), row in _scan_dir(zs_dir, uniso_parent=uniso_parent, train_uniso=train_uniso).items():
                if phase != "shift_zero_shot":
                    continue
                sh = _shift_from_tid(tid)
                for meth, val in row.items():
                    zs_cells[(sh, meth)] = min(val, zs_cells.get((sh, meth), val))

        for step_dir in sorted(shift_dir.glob("step_*")):
            m = re.match(r"^step_(\d+)$", step_dir.name)
            if not m:
                continue
            step = int(m.group(1))
            ev = step_dir / "eval"
            if not ev.is_dir():
                continue
            for (tid, phase), row in _scan_dir(ev, uniso_parent=uniso_parent, train_uniso=train_uniso).items():
                if phase not in _FS_PHASES:
                    continue
                sh = _shift_from_tid(tid)
                for meth, val in row.items():
                    k = (step, sh, phase, meth)
                    cells[k] = min(val, cells.get(k, float("inf")))

    return cells, zs_cells


def export_tables(
    cells: dict[tuple[int, str, str, str], float],
    zs_cells: dict[tuple[str, str], float],
    out_dir: Path,
    *,
    focus_method: str = "mt_text",
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = sorted({k[0] for k in cells})
    shifts = sorted({k[1] for k in cells})
    phases = list(_FS_PHASES)

    rank_rows: list[dict[str, object]] = []
    step_scores: dict[int, list[int]] = defaultdict(list)

    for sh in shifts:
        for ph in phases:
            pairs = [(st, cells.get((st, sh, ph, focus_method), float("nan"))) for st in steps]
            pairs = [(st, v) for st, v in pairs if not np.isnan(v)]
            if not pairs:
                continue
            pairs.sort(key=lambda x: x[1])
            zs_ref = zs_cells.get((sh, focus_method), float("nan"))
            for rank, (st, v) in enumerate(pairs, start=1):
                rank_rows.append(
                    {
                        "shift": sh,
                        "phase": _phase_short(ph),
                        "finetune_steps": st,
                        "rank": rank,
                        focus_method: v,
                        "zs_baseline": zs_ref,
                        "delta_vs_zs": (v - zs_ref) if not np.isnan(zs_ref) else "",
                    }
                )
                step_scores[st].append(rank)

    csv_path = out_dir / f"fs_step_rank_{focus_method}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["shift", "phase", "finetune_steps", "rank", focus_method, "zs_baseline", "delta_vs_zs"],
        )
        w.writeheader()
        w.writerows(rank_rows)
    print(f"[save] {csv_path}")

    mean_rank = sorted(
        ((st, float(np.mean(step_scores[st]))) for st in steps if step_scores.get(st)),
        key=lambda x: x[1],
    )
    best_step = mean_rank[0][0] if mean_rank else None
    summary_path = out_dir / f"fs_step_mean_rank_{focus_method}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["finetune_steps", "mean_rank", "recommended"])
        for st, mr in mean_rank:
            w.writerow([st, f"{mr:.3f}", "yes" if st == best_step else ""])
    print(f"[save] {summary_path}")

    tex_path = out_dir / f"fs_step_grid_{focus_method}.tex"
    lines = [
        r"% Auto-generated by export_quality_exp35_fs_step_rank.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{FS finetune steps vs best $f$ ({focus_method}; lower is better).}}",
        rf"\label{{tab:fs-step-grid-{focus_method}}}",
        r"\begin{tabular}{ll" + "r" * len(steps) + "}",
        r"\hline",
        "Shift & Phase & " + " & ".join(str(s) for s in steps) + r" \\",
        r"\hline",
    ]
    for sh in shifts:
        for i, ph in enumerate(phases):
            vals = [cells.get((st, sh, ph, focus_method), float("nan")) for st in steps]
            vmin = np.nanmin(np.asarray(vals, dtype=np.float64))
            col_cells: list[str] = []
            for v in vals:
                if np.isnan(v):
                    col_cells.append("---")
                elif v <= vmin + 1e-9:
                    col_cells.append(rf"\textbf{{{v:.4f}}}")
                else:
                    col_cells.append(f"{v:.4f}")
            sh_disp = _escape_tex(sh) if i == 0 else ""
            lines.append(f"{sh_disp} & {_phase_tex(ph)} & " + " & ".join(col_cells) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] {tex_path}")

    rec = out_dir / "recommended_finetune_steps.txt"
    rec.write_text(f"focus_method={focus_method}\nbest_mean_rank_step={best_step}\n", encoding="utf-8")
    print(f"[save] {rec}")
    return int(best_step) if best_step is not None else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--uniso_parent", type=str, default="")
    ap.add_argument("--train_uniso", type=str, default="")
    ap.add_argument("--focus_method", type=str, default="mt_text")
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root).resolve()
    out_dir = Path(args.out_dir or sweep_root / "analysis_table").resolve()
    duo = _PROJECT_ROOT
    uniso_parent = Path(args.uniso_parent or duo.parent / "UniSO").resolve()
    train_uniso = Path(
        args.train_uniso or duo / "results/quality_training_3/dtrain_universal_seed0/eval_train_domain"
    ).resolve()

    cells, zs_cells = _collect(sweep_root, uniso_parent=uniso_parent, train_uniso=train_uniso)
    if not cells:
        raise SystemExit(f"no sweep eval npz under {sweep_root}")
    export_tables(cells, zs_cells, out_dir, focus_method=str(args.focus_method))


if __name__ == "__main__":
    main()

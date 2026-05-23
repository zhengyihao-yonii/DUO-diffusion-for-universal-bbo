# -*- coding: utf-8 -*-
"""Frozen protocol for DUO v4 mt_text vs UniSO-T scene-aware Exp1 comparison."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SHIFTS: tuple[str, ...] = ("sim_low", "sim_mid", "sim_high")
PHASES: tuple[str, ...] = ("zs", "fs10", "fs20", "fs50")
PKL_SUFFIX: str = "_3"
SEED: int = 0
TOPK: int = 16
# Align with UniSO ``searcher.ga`` ``num_solutions`` (Quality trace export uses 32 by default).
N_CANDIDATES_DEFAULT: int = 128


@dataclass(frozen=True)
class ComparisonCell:
    """One eval cell: shift × phase (zs or fs10/20/50)."""

    shift: str
    phase: str
    test_task: str
    test_meta_json: Path
    duo_npz: Path
    uniso_data_dir: Path
    uniso_ckpt: Path | None = None


def _phase_to_tail(phase: str) -> str | None:
    if phase == "zs":
        return None
    if phase == "fs10":
        return "tail10p"
    if phase == "fs20":
        return "tail20p"
    if phase == "fs50":
        return "tail50p"
    raise ValueError(f"unknown phase: {phase}")


def _duo_npz_name(shift: str, phase: str) -> str:
    task = f"exp1_D_test_{shift}"
    tail = _phase_to_tail(phase)
    if tail is None:
        return f"quality_trace_mt_text_{task}.meta_shift_zero_shot.npz"
    return (
        f"quality_trace_mt_text_{task}_fewshot_{tail}"
        f".meta_shift_few_shot_{tail}.npz"
    )


def _artifacts_dir(bundle_root: Path, shift: str) -> Path:
    shift_dir = bundle_root / f"shift_{shift}{PKL_SUFFIX}"
    matches = sorted(shift_dir.glob(f"artifacts_*_seed{SEED}"))
    if not matches:
        raise FileNotFoundError(f"no artifacts dir under {shift_dir}")
    return matches[0]


def iter_cells(
    *,
    bundle_root: Path,
    uniso_root: Path,
    shifts: Sequence[str] = SHIFTS,
    phases: Sequence[str] = PHASES,
) -> Iterable[ComparisonCell]:
    """Yield comparison cells with resolved paths."""
    for shift in shifts:
        art = _artifacts_dir(bundle_root, shift)
        data_dir = uniso_root / f"data_exp1_{shift}{PKL_SUFFIX}"
        test_task = f"exp1_D_test_{shift}"
        meta = data_dir / f"{test_task}.meta.json"
        for phase in phases:
            npz = art / _duo_npz_name(shift, phase)
            yield ComparisonCell(
                shift=shift,
                phase=phase,
                test_task=test_task,
                test_meta_json=meta,
                duo_npz=npz,
                uniso_data_dir=data_dir,
            )


def comparison_out_dir(out_root: Path, shift: str, phase: str) -> Path:
    return out_root / f"shift_{shift}{PKL_SUFFIX}" / phase


def uniso_run_root(uniso_root: Path, shift: str, seed: int = SEED) -> Path:
    return uniso_root / "logs" / "exp1_scene" / f"{shift}{PKL_SUFFIX}" / f"seed{seed}"


def uniso_ckpt_for_phase(uniso_root: Path, shift: str, phase: str, seed: int = SEED) -> Path:
    """Checkpoint used for test search at this phase."""
    root = uniso_run_root(uniso_root, shift, seed)
    if phase == "zs":
        return root / "01_base" / "checkpoints" / "last.ckpt"
    tail = _phase_to_tail(phase)
    assert tail is not None
    return root / f"02_fewshot_{tail}" / "checkpoints" / "last.ckpt"

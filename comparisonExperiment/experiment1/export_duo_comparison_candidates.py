# -*- coding: utf-8 -*-
"""
Resample DUO mt_text candidates for comparison (default 128, aligned with UniSO GA).

English doc: Quality ``quality_trace_*.npz`` uses ``sample_batch=32``; this script
re-samples from the same v4 checkpoints / PKLs / phase settings as ``run_quality_suite``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from comparisonExperiment.experiment1.comparison_v4_protocol import (
    N_CANDIDATES_DEFAULT,
    PHASES,
    SEED,
    SHIFTS,
    _phase_to_tail,
    iter_cells,
)
from comparisonExperiment.experiment1.export_quality_trace_jsonl import _to_jsonl_rows
from QualityExperiment.metadata_embed import resolve_shift_text_embedding
from QualityExperiment.run_landscape_experiment import (
    _TRACE_DIFF_OVERRIDES,
    _context_length_for_ckpt,
    _make_projector,
)
from QualityExperiment.trace_sampling import sample_latent_trace


def _find_pkl(pkl_dir: Path, task_id: str, *, tail: str | None) -> Path:
    """Match ``run_quality_suite._find_pkl_for_task`` naming (no ``exp1_`` prefix on PKL stem)."""
    if tail:
        hits = sorted(pkl_dir.glob(f"{task_id}_fewshot_{tail}_*.pkl"))
        if not hits:
            hits = sorted(
                p for p in pkl_dir.glob(f"{task_id}_fewshot*.pkl") if tail in p.stem
            )
    else:
        hits = sorted(pkl_dir.glob(f"{task_id}_h*.pkl"))
    if not hits:
        raise FileNotFoundError(f"no pkl for {task_id} tail={tail} under {pkl_dir}")
    return hits[0]


def _fs_ckpt(base: Path, tail: str) -> Path:
    stem = base.stem
    if stem.endswith("_fs"):
        stem = f"{stem}_{tail}"
    else:
        stem = f"{stem}_fs_{tail}"
    p = base.parent / f"{stem}.pt"
    if not p.is_file():
        p = base.parent / f"ckpt_mt_text_fs_{tail}.pt"
    return p


def _resolve_ckpt(
    *,
    bundle_root: Path,
    shift: str,
    phase: str,
    universal_ckpt_dir: Path,
) -> Path:
    if phase == "zs":
        p = universal_ckpt_dir / "ckpt_mt_text.pt"
        if not p.is_file():
            raise FileNotFoundError(str(p))
        return p
    tail = _phase_to_tail(phase)
    assert tail is not None
    fs_dir = bundle_root / f"shift_{shift}_3" / "fs_checkpoints"
    p = fs_dir / f"ckpt_mt_text_fs_{tail}.pt"
    if not p.is_file():
        p = _fs_ckpt(universal_ckpt_dir / "ckpt_mt_text.pt", tail)
    if not p.is_file():
        raise FileNotFoundError(f"no fs ckpt for {shift}/{phase} under {fs_dir}")
    return p


def _mt_text_overrides(mt_num_tasks: int, text_dim: int) -> dict[str, Any]:
    mo = dict(_TRACE_DIFF_OVERRIDES)
    mo.update(
        {
            "task_condition": True,
            "text_condition": True,
            "num_tasks": int(mt_num_tasks),
            "text_embed_input_dim": int(text_dim),
        }
    )
    return mo


def resample_duo_mt_text(
    *,
    meta_json: Path,
    train_pkl: Path,
    ckpt: Path,
    out_jsonl: Path,
    sample_batch: int,
    horizon: int,
    context_length_train: int,
    context_length_fewshot: int,
    phase: str,
    text_encoder: str,
    device: str,
    prefix_seed: int,
) -> int:
    """Sample ``sample_batch`` trajectories; export native ``x_last`` rows to jsonl."""
    held_vec = resolve_shift_text_embedding(
        meta_json,
        explicit_npy=None,
        encoder_model=str(text_encoder),
    )
    proj = _make_projector(meta_json, train_pkl, device=device)
    @dataclass(frozen=True)
    class _CtxCfg:
        context_length_train: int
        context_length_fewshot: int

    ctx_cfg = _CtxCfg(
        context_length_train=int(context_length_train),
        context_length_fewshot=int(context_length_fewshot),
    )

    tr = sample_latent_trace(
        train_pkl=train_pkl,
        ckpt_path=ckpt,
        horizon=int(horizon),
        context_length=_context_length_for_ckpt(ckpt, ctx_cfg),
        sample_batch=int(sample_batch),
        device=str(device),
        project_z=proj,
        model_overrides=_mt_text_overrides(
            mt_num_tasks=5,
            text_dim=int(held_vec.shape[-1]),
        ),
        include_task_idx=True,
        task_idx=0,
        text_embed_override=held_vec,
        prefix_seed=int(prefix_seed),
    )
    x = np.asarray(tr.raw_x_last, dtype=np.float64)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = _to_jsonl_rows(x, method="duo_mt_text")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return int(x.shape[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle_root", type=str, default="results/quality_bundle_4")
    ap.add_argument("--uniso_root", type=str, default="../UniSO")
    ap.add_argument("--duo_root", type=str, default=".")
    ap.add_argument(
        "--universal_ckpt_dir",
        type=str,
        default="results/quality_training_3/dtrain_universal_seed0/checkpoints",
    )
    ap.add_argument("--out_root", type=str, default="results/comparison1/exp1_scene_v4")
    ap.add_argument("--sample_batch", type=int, default=N_CANDIDATES_DEFAULT)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--context_length_train", type=int, default=32)
    ap.add_argument("--context_length_fewshot", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--text_encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--shifts", type=str, default=",".join(SHIFTS))
    ap.add_argument("--phases", type=str, default=",".join(PHASES))
    args = ap.parse_args()

    duo_root = Path(args.duo_root).resolve()
    bundle_root = (duo_root / args.bundle_root).resolve()
    uniso_root = (duo_root / args.uniso_root).resolve()
    out_root = (duo_root / args.out_root).resolve()
    univ_ckpt = (duo_root / args.universal_ckpt_dir).resolve()

    shifts = [s.strip() for s in str(args.shifts).split(",") if s.strip()]
    phases = [p.strip() for p in str(args.phases).split(",") if p.strip()]

    for cell in iter_cells(
        bundle_root=bundle_root,
        uniso_root=uniso_root,
        shifts=shifts,
        phases=phases,
    ):
        pkl_dir = duo_root / "generated_datasets" / f"exp1_{cell.shift}_3"
        tail = _phase_to_tail(cell.phase)
        task_id = f"D_test_{cell.shift}"
        pkl = _find_pkl(pkl_dir, task_id, tail=tail)
        ckpt = _resolve_ckpt(
            bundle_root=bundle_root,
            shift=cell.shift,
            phase=cell.phase,
            universal_ckpt_dir=univ_ckpt,
        )
        out_jsonl = (
            out_root / f"shift_{cell.shift}_3" / cell.phase / "candidates" / "duo_mt_text.jsonl"
        )
        seed_tag = hashlib.md5(f"{cell.shift}:{cell.phase}:duo_mt_text".encode()).hexdigest()
        prefix_seed = int(seed_tag[:8], 16) % (2**31)
        n = resample_duo_mt_text(
            meta_json=cell.test_meta_json,
            train_pkl=pkl,
            ckpt=ckpt,
            out_jsonl=out_jsonl,
            sample_batch=int(args.sample_batch),
            horizon=int(args.horizon),
            context_length_train=int(args.context_length_train),
            context_length_fewshot=int(args.context_length_fewshot),
            phase=cell.phase,
            text_encoder=str(args.text_encoder),
            device=str(args.device),
            prefix_seed=prefix_seed,
        )
        print(f"[duo-resample] {cell.shift}/{cell.phase} n={n} ckpt={ckpt.name} -> {out_jsonl}")


if __name__ == "__main__":
    main()

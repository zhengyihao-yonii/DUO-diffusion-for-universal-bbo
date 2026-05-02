#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch QualityExperiment: train-domain trajectories + shifted-domain ZS/FS (wandb).

English doc:
  * **Train domain**: for each ``exp1_D_train_*.meta.json``, run four visualize modes
    (``st_duo``, ``st_text``, ``mt_label``, ``mt_text``) on that task's PKL with the correct
    multitask ``task_idx``.
  * **Shift zero-shot**: for ``exp1_D_test_*.meta.json``, evaluate multitask checkpoints on the
    held-out instance; single-task checkpoints are skipped (no universal backbone).
  * **Shift few-shot**: optional separate checkpoints fine-tuned on ``*_fewshot*.pkl`` (small-data retrain).

中文注释: 与 ``visualize.sh`` 四条分支对齐；wandb Table 中可按 ``raw_f`` 排序挑选代表性轨迹。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from QualityExperiment.run_landscape_experiment import (
    LandscapeRunConfig,
    run_landscape_experiment_core,
)


def _parse_d_train_index(task_id: str) -> int | None:
    m = re.match(r"^D_train_(\d+)$", task_id)
    if not m:
        return None
    return int(m.group(1)) - 1


def _task_id_from_meta_path(p: Path) -> str:
    raw = json.loads(p.read_text(encoding="utf-8"))
    tid = raw.get("task_id")
    if tid is None:
        raise ValueError(f"meta missing task_id: {p}")
    return str(tid)


def _find_pkl_for_task(pkl_dir: Path, task_id: str, *, fewshot: bool) -> Path | None:
    """Pick first matching PKL (exp1 naming from ``run_exp1.py``)."""
    if fewshot:
        hits = sorted(pkl_dir.glob(f"{task_id}_fewshot*.pkl"))
    else:
        hits = sorted(pkl_dir.glob(f"{task_id}_h*.pkl"))
    return hits[0] if hits else None


def _discover_metas(uniso_dir: Path, glob_pat: str) -> list[Path]:
    return sorted(uniso_dir.glob(glob_pat))


def _resolve_meta_json(uniso_dir: Path, task_id: str, *, fewshot: bool) -> Path:
    """Prefer ``exp1_<task>_fewshot.meta.json`` for FS phase when present."""
    if fewshot:
        fp = uniso_dir / f"exp1_{task_id}_fewshot.meta.json"
        if fp.is_file():
            return fp
    return uniso_dir / f"exp1_{task_id}.meta.json"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Orchestrate QualityExperiment wandb runs on exp1 synthetic families."
    )
    ap.add_argument(
        "--uniso_data_dir",
        type=str,
        default=str(_PROJECT_ROOT.parent / "UniSO" / "data"),
        help="Directory containing exp1_*.meta.json (from run_exp1.py).",
    )
    ap.add_argument(
        "--pkl_dir",
        type=str,
        required=True,
        help="generated_datasets/exp1_gap0p500 (contains D_train_*_h*_*.pkl).",
    )
    ap.add_argument(
        "--phases",
        type=str,
        default="train_domain,shift_zero_shot",
        help="Comma-separated: train_domain | shift_zero_shot | shift_few_shot",
    )
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--sample_batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--wandb_project", type=str, default="duo-quality-suite")
    ap.add_argument("--wandb_group", type=str, default="quality-exp1")
    ap.add_argument("--task_text_embeds_npy", type=str, default="")
    ap.add_argument("--mt_num_tasks", type=int, default=5)
    ap.add_argument("--no_traj_table", action="store_true")
    ap.add_argument(
        "--held_out_text_embed_npy",
        type=str,
        default="",
        help="Optional single [E] vector shared override; default encodes each meta's metadata_text.",
    )
    ap.add_argument(
        "--text_encoder_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    ap.add_argument(
        "--local_out_dir",
        type=str,
        default="",
        help="Directory for PNG + NPZ artifacts (recommended for paper figures).",
    )
    ap.add_argument(
        "--include_st_on_shift_zs",
        action="store_true",
        help="If set, also run st_duo / st_text on held-out tasks (default: skip; no true ZS for ST).",
    )
    ap.add_argument(
        "--zs_task_idx",
        type=int,
        default=0,
        help="Task index passed to multitask models on held-out instance (proxy; default 0).",
    )
    ap.add_argument("--ckpt_st_duo", type=str, default="")
    ap.add_argument("--ckpt_st_text", type=str, default="")
    ap.add_argument("--ckpt_mt_label", type=str, default="")
    ap.add_argument("--ckpt_mt_text", type=str, default="")
    ap.add_argument("--ckpt_st_duo_fs", type=str, default="")
    ap.add_argument("--ckpt_st_text_fs", type=str, default="")
    ap.add_argument("--ckpt_mt_label_fs", type=str, default="")
    ap.add_argument("--ckpt_mt_text_fs", type=str, default="")
    args = ap.parse_args()

    phases = {p.strip() for p in str(args.phases).split(",") if p.strip()}
    uniso_dir = Path(args.uniso_data_dir).resolve()
    pkl_dir = Path(args.pkl_dir).resolve()
    if not pkl_dir.is_dir():
        raise SystemExit(f"pkl_dir not found: {pkl_dir}")

    zs_skip_st = not bool(args.include_st_on_shift_zs)

    base_kw = dict(
        horizon=int(args.horizon),
        sample_batch=int(args.sample_batch),
        stride=int(args.stride),
        device=str(args.device),
        wandb_project=str(args.wandb_project),
        wandb_group=str(args.wandb_group),
        ckpt_st_duo=str(args.ckpt_st_duo),
        ckpt_st_text=str(args.ckpt_st_text),
        ckpt_mt_label=str(args.ckpt_mt_label),
        ckpt_mt_text=str(args.ckpt_mt_text),
        task_text_embeds_npy=str(args.task_text_embeds_npy),
        mt_num_tasks=int(args.mt_num_tasks),
        no_traj_table=bool(args.no_traj_table),
        held_out_text_embed_npy=str(args.held_out_text_embed_npy),
        text_encoder_model=str(args.text_encoder_model),
        local_out_dir=str(args.local_out_dir),
    )

    # ----- Phase A: one wandb run per training task instance -----
    if "train_domain" in phases:
        metas = _discover_metas(uniso_dir, "exp1_D_train_*.meta.json")
        if not metas:
            print(f"[warn] no exp1_D_train_*.meta.json under {uniso_dir}")
        for meta in metas:
            tid = _task_id_from_meta_path(meta)
            pkl = _find_pkl_for_task(pkl_dir, tid, fewshot=False)
            if pkl is None:
                print(f"[skip] no pkl for {tid} in {pkl_dir}")
                continue
            tidx = _parse_d_train_index(tid)
            if tidx is None or tidx < 0:
                print(f"[skip] cannot parse train index from {tid}")
                continue
            run_name = f"{tid}__train_domain"
            print(f"[run] {run_name} task_idx={tidx}")
            cfg = LandscapeRunConfig(
                meta_json=meta,
                train_pkl=pkl,
                wandb_run_name=run_name,
                task_idx=int(tidx),
                phase_label="train_domain",
                task_id_label=tid,
                skip_methods=frozenset(),
                **base_kw,
            )
            run_landscape_experiment_core(cfg)

    # ----- Phase B: held-out synthetic instance (multitask ZS; ST skipped) -----
    if "shift_zero_shot" in phases:
        metas = _discover_metas(uniso_dir, "exp1_D_test_*.meta.json")
        if not metas:
            print(f"[warn] no exp1_D_test_*.meta.json under {uniso_dir}")
        sk = frozenset(("st_duo", "st_text")) if zs_skip_st else frozenset()
        for meta in metas:
            tid = _task_id_from_meta_path(meta)
            pkl = _find_pkl_for_task(pkl_dir, tid, fewshot=False)
            if pkl is None:
                print(f"[skip] no pkl for {tid}")
                continue
            run_name = f"{tid}__shift_zs"
            print(f"[run] {run_name} zs_task_idx={int(args.zs_task_idx)} skip_st={zs_skip_st}")
            cfg = LandscapeRunConfig(
                meta_json=meta,
                train_pkl=pkl,
                wandb_run_name=run_name,
                task_idx=int(args.zs_task_idx),
                phase_label="shift_zero_shot",
                task_id_label=tid,
                skip_methods=sk,
                **base_kw,
            )
            run_landscape_experiment_core(cfg)

    # ----- Phase C: few-shot PKLs + optional fine-tuned ckpts -----
    if "shift_few_shot" in phases:
        metas = _discover_metas(uniso_dir, "exp1_D_test_*.meta.json")
        fs_kw = {
            **base_kw,
            "ckpt_st_duo": str(args.ckpt_st_duo_fs or args.ckpt_st_duo),
            "ckpt_st_text": str(args.ckpt_st_text_fs or args.ckpt_st_text),
            "ckpt_mt_label": str(args.ckpt_mt_label_fs or args.ckpt_mt_label),
            "ckpt_mt_text": str(args.ckpt_mt_text_fs or args.ckpt_mt_text),
        }
        for meta in metas:
            tid = _task_id_from_meta_path(meta)
            pkl = _find_pkl_for_task(pkl_dir, tid, fewshot=True)
            if pkl is None:
                print(f"[skip] no fewshot pkl for {tid} (*_fewshot*.pkl)")
                continue
            meta_use = _resolve_meta_json(uniso_dir, tid, fewshot=True)
            run_name = f"{tid}__shift_few_shot"
            print(f"[run] {run_name} pkl={pkl.name} meta={meta_use.name}")
            cfg = LandscapeRunConfig(
                meta_json=meta_use,
                train_pkl=pkl,
                wandb_run_name=run_name,
                task_idx=int(args.zs_task_idx),
                phase_label="shift_few_shot",
                task_id_label=tid,
                skip_methods=frozenset(),
                **fs_kw,
            )
            run_landscape_experiment_core(cfg)


if __name__ == "__main__":
    main()

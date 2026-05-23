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


def _find_pkl_for_task(
    pkl_dir: Path,
    task_id: str,
    *,
    fewshot: bool,
    tail_tag: str | None = None,
) -> Path | None:
    """Pick first matching PKL (exp1 naming from ``run_exp1.py``)."""
    if fewshot:
        if tail_tag:
            hits = sorted(pkl_dir.glob(f"{task_id}_fewshot_{tail_tag}_*.pkl"))
            if hits:
                return hits[0]
        hits = sorted(pkl_dir.glob(f"{task_id}_fewshot*.pkl"))
    else:
        hits = sorted(pkl_dir.glob(f"{task_id}_h*.pkl"))
    return hits[0] if hits else None


def _fs_ckpt_path(base_ckpt: str, tail_tag: str | None) -> str:
    """English doc: ``ckpt_*_fs.pt`` → ``ckpt_*_fs_tail10p.pt``; fallback to base if tail ckpt missing."""
    if not str(base_ckpt).strip():
        return str(base_ckpt)
    p = Path(base_ckpt)
    if tail_tag is None:
        return str(base_ckpt)
    cand = p.with_name(f"{p.stem}_{tail_tag}{p.suffix}")
    if cand.is_file():
        return str(cand)
    if p.is_file():
        print(f"[warn] fs ckpt missing {cand.name}; using {p.name}")
        return str(p)
    return str(cand)


def _fs_tail_from_phase(phase: str) -> str | None:
    """Return ``tail10p`` / … or None for legacy ``shift_few_shot``."""
    if phase == "shift_few_shot":
        return None
    prefix = "shift_few_shot_"
    if phase.startswith(prefix):
        suf = phase[len(prefix) :]
        if suf.startswith("tail") and suf.endswith("p"):
            return suf
    return None


def _discover_metas(uniso_dir: Path, glob_pat: str) -> list[Path]:
    return sorted(uniso_dir.glob(glob_pat))


def _d_test_base_metas(uniso_dir: Path) -> list[Path]:
    """English doc: ``exp1_D_test_*.meta.json`` excluding few-shot sidecars."""
    return sorted(
        p
        for p in uniso_dir.glob("exp1_D_test_*.meta.json")
        if "fewshot" not in p.stem
    )


def _resolve_meta_json(
    uniso_dir: Path,
    task_id: str,
    *,
    fewshot: bool,
    tail_tag: str | None = None,
) -> Path:
    """Prefer ``exp1_<task>_fewshot_<tail>.meta.json`` for FS phase when present."""
    if fewshot:
        if tail_tag:
            fp = uniso_dir / f"exp1_{task_id}_fewshot_{tail_tag}.meta.json"
            if fp.is_file():
                return fp
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
        default=(
            "train_domain,shift_zero_shot,"
            "shift_few_shot_tail10p,shift_few_shot_tail20p,shift_few_shot_tail50p"
        ),
        help="Comma-separated phases incl. shift_few_shot_tail{10,20,50}p or legacy shift_few_shot",
    )
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument(
        "--context_length_train",
        type=int,
        default=32,
        help="Prefix ctx for main checkpoints (must match training).",
    )
    ap.add_argument(
        "--context_length_fewshot",
        type=int,
        default=16,
        help="Prefix ctx for *_fs_* checkpoints (must match finetune).",
    )
    ap.add_argument("--sample_batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--wandb_project", type=str, default="duo-quality-suite")
    ap.add_argument("--wandb_group", type=str, default="quality-exp1")
    ap.add_argument("--task_text_embeds_npy", type=str, default="")
    ap.add_argument("--mt_num_tasks", type=int, default=4)
    ap.add_argument("--no_traj_table", action="store_true")
    ap.add_argument(
        "--no_landscape_figure",
        action="store_true",
        help="Skip latent landscape PNG + wandb image (NPZ metrics artifacts still written).",
    )
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
    ap.add_argument(
        "--no_proxy_filter",
        action="store_true",
        help="Eval: disable proxy trajectory selection (legacy last-step min).",
    )
    args = ap.parse_args()

    phases = {p.strip() for p in str(args.phases).split(",") if p.strip()}
    uniso_dir = Path(args.uniso_data_dir).resolve()
    pkl_dir = Path(args.pkl_dir).resolve()
    if not pkl_dir.is_dir():
        raise SystemExit(f"pkl_dir not found: {pkl_dir}")

    zs_skip_st = not bool(args.include_st_on_shift_zs)

    base_kw = dict(
        horizon=int(args.horizon),
        context_length_train=int(args.context_length_train),
        context_length_fewshot=int(args.context_length_fewshot),
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
        no_landscape_figure=bool(args.no_landscape_figure),
        held_out_text_embed_npy=str(args.held_out_text_embed_npy),
        text_encoder_model=str(args.text_encoder_model),
        local_out_dir=str(args.local_out_dir),
        use_proxy_filter=not bool(args.no_proxy_filter),
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
        metas = _d_test_base_metas(uniso_dir)
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

    # ----- Phase C: few-shot PKLs + fine-tuned ckpts (optional tail-specific) -----
    fs_phases = sorted(p for p in phases if str(p).startswith("shift_few_shot"))
    if fs_phases:
        metas = _d_test_base_metas(uniso_dir)
        for phase_fs in fs_phases:
            tail_tag = _fs_tail_from_phase(str(phase_fs))
            fs_kw = {
                **base_kw,
                "ckpt_st_duo": _fs_ckpt_path(str(args.ckpt_st_duo_fs or args.ckpt_st_duo), tail_tag),
                "ckpt_st_text": _fs_ckpt_path(str(args.ckpt_st_text_fs or args.ckpt_st_text), tail_tag),
                "ckpt_mt_label": _fs_ckpt_path(str(args.ckpt_mt_label_fs or args.ckpt_mt_label), tail_tag),
                "ckpt_mt_text": _fs_ckpt_path(str(args.ckpt_mt_text_fs or args.ckpt_mt_text), tail_tag),
            }
            for meta in metas:
                tid = _task_id_from_meta_path(meta)
                pkl = _find_pkl_for_task(pkl_dir, tid, fewshot=True, tail_tag=tail_tag)
                if pkl is None:
                    print(f"[skip] no fewshot pkl for {tid} tail={tail_tag} (*_fewshot*.pkl)")
                    continue
                meta_use = _resolve_meta_json(uniso_dir, tid, fewshot=True, tail_tag=tail_tag)
                run_name = f"{tid}__{phase_fs}"
                print(f"[run] {run_name} pkl={pkl.name} meta={meta_use.name}")
                cfg = LandscapeRunConfig(
                    meta_json=meta_use,
                    train_pkl=pkl,
                    wandb_run_name=run_name,
                    task_idx=int(args.zs_task_idx),
                    phase_label=str(phase_fs),
                    task_id_label=tid,
                    skip_methods=frozenset(),
                    **fs_kw,
                )
                run_landscape_experiment_core(cfg)


if __name__ == "__main__":
    main()

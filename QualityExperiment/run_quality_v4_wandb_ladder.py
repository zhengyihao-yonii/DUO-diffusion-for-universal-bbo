#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W&B ladder traces for Quality v4: one run per D_train / D_test task, four method figures each.

English doc: Uses ``run_landscape_experiment_core`` with ``split_landscape_per_method=True``.
Phases: train_domain (5× D_train), shift_zero_shot + shift_few_shot_tail10p per test shift.

中文注释: 每个任务一个 wandb run；四模型各一张 Branin 阶梯轨迹图。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from QualityExperiment.run_landscape_experiment import (  # noqa: E402
    LandscapeRunConfig,
    run_landscape_experiment_core,
)
from QualityExperiment.run_quality_suite import (  # noqa: E402
    _d_test_base_metas,
    _discover_metas,
    _find_pkl_for_task,
    _fs_ckpt_path,
    _parse_d_train_index,
    _task_id_from_meta_path,
)

_TEST_SHIFTS: tuple[str, ...] = ("sim_low", "sim_mid", "sim_high")
_SHIFT_TAG_RE = re.compile(r"^shift_(sim_low|sim_mid|sim_high)_3$")


def _shift_dirs(bundle_root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for p in sorted(bundle_root.iterdir()):
        if not p.is_dir():
            continue
        m = _SHIFT_TAG_RE.match(p.name)
        if m:
            out.append((m.group(1), p))
    return out


def _run_ladder(
    *,
    cfg: LandscapeRunConfig,
    label: str,
) -> None:
    print(f"[ladder] {label} -> wandb run {cfg.wandb_run_name}", flush=True)
    run_landscape_experiment_core(cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Quality v4 W&B ladder plots per task.")
    ap.add_argument("--bundle_root", type=str, required=True)
    ap.add_argument("--train_root", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--context_length_train", type=int, default=32)
    ap.add_argument("--context_length_fewshot", type=int, default=16)
    ap.add_argument("--mt_num_tasks", type=int, default=5)
    ap.add_argument("--text_encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--wandb_project", type=str, default="duo-quality-suite")
    ap.add_argument("--wandb_group", type=str, default="quality_v4_ladder_traces")
    ap.add_argument("--max_rep_trajs", type=int, default=4)
    ap.add_argument("--sample_batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--zs_task_idx", type=int, default=0)
    ap.add_argument("--skip_fs_ladder", action="store_true")
    args = ap.parse_args()

    bundle_root = Path(args.bundle_root).resolve()
    train_root = Path(
        args.train_root or bundle_root.parent / "quality_training_3"
    ).resolve()
    univ = train_root / f"dtrain_universal_seed{int(args.seed)}"
    main_ckpt = univ / "checkpoints"
    task_emb = univ / "task_text_embeds_TxE.npy"
    if not task_emb.is_file():
        raise SystemExit(f"missing task embeddings: {task_emb}")

    duo_root = _PROJECT_ROOT
    canon_pkl = duo_root / "generated_datasets" / "exp1_sim_low_3"
    canon_uniso = duo_root.parent / "UniSO" / "data_exp1_sim_low_3"
    eval_train = univ / "eval_train_domain"

    base_kw = dict(
        horizon=int(args.horizon),
        context_length_train=int(args.context_length_train),
        context_length_fewshot=int(args.context_length_fewshot),
        sample_batch=int(args.sample_batch),
        stride=int(args.stride),
        device=str(args.device),
        wandb_project=str(args.wandb_project),
        wandb_group=str(args.wandb_group),
        task_text_embeds_npy=str(task_emb),
        mt_num_tasks=int(args.mt_num_tasks),
        no_traj_table=False,
        no_landscape_figure=False,
        split_landscape_per_method=True,
        max_rep_trajs=int(args.max_rep_trajs),
        plot_all_trajs=False,
        text_encoder_model=str(args.text_encoder),
        log_combined_table=False,
    )

    # ----- D_train × train_domain (canonical sim_low PKL / main ckpts) -----
    if canon_uniso.is_dir() and canon_pkl.is_dir():
        for meta in _discover_metas(canon_uniso, "exp1_D_train_*.meta.json"):
            tid = _task_id_from_meta_path(meta)
            pkl = _find_pkl_for_task(canon_pkl, tid, fewshot=False)
            tidx = _parse_d_train_index(tid)
            if pkl is None or tidx is None:
                continue
            out_dir = eval_train / f"ladder_{tid}"
            _run_ladder(
                cfg=LandscapeRunConfig(
                    meta_json=meta,
                    train_pkl=pkl,
                    wandb_run_name=f"{tid}__train_domain",
                    task_idx=int(tidx),
                    phase_label="train_domain",
                    task_id_label=tid,
                    skip_methods=frozenset(),
                    local_out_dir=str(out_dir),
                    ckpt_st_duo=str(main_ckpt / "ckpt_st_duo.pt"),
                    ckpt_st_text=str(main_ckpt / "ckpt_st_text.pt"),
                    ckpt_mt_label=str(main_ckpt / "ckpt_mt_label.pt"),
                    ckpt_mt_text=str(main_ckpt / "ckpt_mt_text.pt"),
                    **base_kw,
                ),
                label=f"train {tid}",
            )

    # ----- D_test per shift: ZS + FS tail10p -----
    for shift, shift_dir in _shift_dirs(bundle_root):
        pkl_dir = duo_root / "generated_datasets" / f"exp1_{shift}_3"
        uniso_dir = duo_root.parent / "UniSO" / f"data_exp1_{shift}_3"
        fs_dir = shift_dir / "fs_checkpoints"
        art_dir = shift_dir / f"artifacts_exp1_{shift}_3_seed{int(args.seed)}"
        if not uniso_dir.is_dir() or not pkl_dir.is_dir():
            print(f"[warn] skip shift={shift}: missing uniso or pkl")
            continue

        main_kw = {
            "ckpt_st_duo": str(main_ckpt / "ckpt_st_duo.pt"),
            "ckpt_st_text": str(main_ckpt / "ckpt_st_text.pt"),
            "ckpt_mt_label": str(main_ckpt / "ckpt_mt_label.pt"),
            "ckpt_mt_text": str(main_ckpt / "ckpt_mt_text.pt"),
        }
        fs_kw = {
            "ckpt_st_duo": _fs_ckpt_path(str(fs_dir / "ckpt_st_duo_fs.pt"), "tail10p"),
            "ckpt_st_text": _fs_ckpt_path(str(fs_dir / "ckpt_st_text_fs.pt"), "tail10p"),
            "ckpt_mt_label": _fs_ckpt_path(str(fs_dir / "ckpt_mt_label_fs.pt"), "tail10p"),
            "ckpt_mt_text": _fs_ckpt_path(str(fs_dir / "ckpt_mt_text_fs.pt"), "tail10p"),
        }

        for meta in _d_test_base_metas(uniso_dir):
            tid = _task_id_from_meta_path(meta)
            pkl_zs = _find_pkl_for_task(pkl_dir, tid, fewshot=False)
            if pkl_zs is not None:
                _run_ladder(
                    cfg=LandscapeRunConfig(
                        meta_json=meta,
                        train_pkl=pkl_zs,
                        wandb_run_name=f"{tid}__{shift}__shift_zs",
                        task_idx=int(args.zs_task_idx),
                        phase_label="shift_zero_shot",
                        task_id_label=tid,
                        skip_methods=frozenset(("st_duo", "st_text")),
                        local_out_dir=str(art_dir / f"ladder_{tid}_zs"),
                        **main_kw,
                        **base_kw,
                    ),
                    label=f"{tid} zs ({shift})",
                )

            if not bool(args.skip_fs_ladder):
                pkl_fs = _find_pkl_for_task(pkl_dir, tid, fewshot=True, tail_tag="tail10p")
                if pkl_fs is not None:
                    _run_ladder(
                        cfg=LandscapeRunConfig(
                            meta_json=meta,
                            train_pkl=pkl_fs,
                            wandb_run_name=f"{tid}__{shift}__fs_tail10p",
                            task_idx=int(args.zs_task_idx),
                            phase_label="shift_few_shot_tail10p",
                            task_id_label=tid,
                            skip_methods=frozenset(),
                            local_out_dir=str(art_dir / f"ladder_{tid}_fs10"),
                            **fs_kw,
                            **base_kw,
                        ),
                        label=f"{tid} fs10 ({shift})",
                    )

    print(f"[done] wandb group={args.wandb_group}")


if __name__ == "__main__":
    main()

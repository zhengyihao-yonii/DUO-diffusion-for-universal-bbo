#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-shot finetune step sweep for Quality exp3.5.

English doc: Same ``finetune_lr`` for tail10p/20p/50p. For each milestone M, finetune from
main ckpts for M steps, evaluate fs10/20/50, export rank tables (mt_text focus).

中文注释: 每个 M 从主训 ckpt 独立微调（非续训累加）；统一学习率。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import importlib

_DEFAULT_CFG = "config.quality_exp35_train"

# 每个 milestone：4 variants × 3 tail → 12 ckpt / 12 eval npz；ZS 仅 mt_label/mt_text → 2
_EXPECTED_FS_CKPTS = 12
_EXPECTED_FS_EVAL_NPZ = 12
_EXPECTED_ZS_EVAL_NPZ = 2


def _pkl_tag(shift: str, suffix: str) -> str:
    return f"exp1_{shift.strip()}{suffix}"


def _fs_ckpts_done(fs_dir: Path) -> bool:
    if not fs_dir.is_dir():
        return False
    return len(list(fs_dir.glob("ckpt_*_fs_*.pt"))) >= _EXPECTED_FS_CKPTS


def _fs_eval_done(eval_dir: Path) -> bool:
    if not eval_dir.is_dir():
        return False
    return len(list(eval_dir.glob("quality_trace_*.npz"))) >= _EXPECTED_FS_EVAL_NPZ


def _zs_eval_done(zs_eval: Path) -> bool:
    if not zs_eval.is_dir():
        return False
    return len(list(zs_eval.glob("quality_trace_*.npz"))) >= _EXPECTED_ZS_EVAL_NPZ


def _run(cmd: list[str], *, log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd), flush=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("\n=== " + " ".join(cmd) + " ===\n")
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
            env=os.environ.copy(),
        )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): see {log_path}")


def _finetune_milestone(
    *,
    python: str,
    duo_root: Path,
    canon_pkl: Path,
    pkl_dir_ft: Path,
    main_ckpt_dir: Path,
    fs_out_dir: Path,
    task_emb: Path,
    milestone: int,
    finetune_lr: float,
    horizon: int,
    ctx_fs: int,
    n_tasks: int,
    device: str,
    log_path: Path,
    shift_tag: str,
    wandb_project: str,
    run_label: str,
    force: bool,
) -> None:
    fs_out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        "-m",
        "QualityExperiment.train_exp1_checkpoints",
        "--pkl_dir",
        str(canon_pkl),
        "--pkl_dir_finetune",
        str(pkl_dir_ft),
        "--task_text_embeds_npy",
        str(task_emb),
        "--out_dir",
        str(main_ckpt_dir),
        "--finetune_out_dir",
        str(fs_out_dir),
        "--horizon",
        str(horizon),
        "--context_length_fewshot",
        str(ctx_fs),
        "--mt_num_tasks",
        str(n_tasks),
        "--device",
        device,
        "--skip_main",
        "--finetune_steps",
        str(int(milestone)),
        "--finetune_lr",
        str(float(finetune_lr)),
        "--fs_tail_tags",
        "tail10p,tail20p,tail50p",
    ]
    if force:
        cmd.append("--force_finetune")
    if wandb_project:
        cmd.extend(
            [
                "--wandb_project",
                wandb_project,
                "--wandb_group",
                f"quality_{run_label}_fs_sweep_{shift_tag}_step{milestone}",
            ]
        )
    _run(cmd, log_path=log_path, cwd=duo_root)


def _eval_suite(
    *,
    python: str,
    duo_root: Path,
    pkl_dir: Path,
    uniso_dir: Path,
    main_ckpt_dir: Path,
    fs_ckpt_dir: Path,
    eval_out: Path,
    task_emb: Path,
    horizon: int,
    ctx_train: int,
    ctx_fs: int,
    n_tasks: int,
    text_encoder: str,
    device: str,
    log_path: Path,
    shift_tag: str,
    wandb_project: str,
    run_label: str,
    phases: str,
) -> None:
    eval_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        "-m",
        "QualityExperiment.run_quality_suite",
        "--uniso_data_dir",
        str(uniso_dir),
        "--pkl_dir",
        str(pkl_dir),
        "--horizon",
        str(horizon),
        "--context_length_train",
        str(ctx_train),
        "--context_length_fewshot",
        str(ctx_fs),
        "--phases",
        phases,
        "--task_text_embeds_npy",
        str(task_emb),
        "--mt_num_tasks",
        str(n_tasks),
        "--text_encoder_model",
        text_encoder,
        "--local_out_dir",
        str(eval_out),
        "--ckpt_st_duo",
        str(main_ckpt_dir / "ckpt_st_duo.pt"),
        "--ckpt_st_text",
        str(main_ckpt_dir / "ckpt_st_text.pt"),
        "--ckpt_mt_label",
        str(main_ckpt_dir / "ckpt_mt_label.pt"),
        "--ckpt_mt_text",
        str(main_ckpt_dir / "ckpt_mt_text.pt"),
        "--ckpt_st_duo_fs",
        str(fs_ckpt_dir / "ckpt_st_duo_fs.pt"),
        "--ckpt_st_text_fs",
        str(fs_ckpt_dir / "ckpt_st_text_fs.pt"),
        "--ckpt_mt_label_fs",
        str(fs_ckpt_dir / "ckpt_mt_label_fs.pt"),
        "--ckpt_mt_text_fs",
        str(fs_ckpt_dir / "ckpt_mt_text_fs.pt"),
        "--device",
        device,
        "--no_landscape_figure",
    ]
    if wandb_project:
        cmd.extend(
            ["--wandb_project", wandb_project, "--wandb_group", f"quality_{run_label}_eval_{shift_tag}"]
        )
    _run(cmd, log_path=log_path, cwd=duo_root)


def main() -> None:
    ap = argparse.ArgumentParser(description="Quality exp3.5 FS step sweep + rank table.")
    ap.add_argument("--duo_root", type=str, default=str(_PROJECT_ROOT))
    ap.add_argument("--python", type=str, default=os.environ.get("PYTHON", sys.executable))
    ap.add_argument("--sweep_root", type=str, default="")
    ap.add_argument("--train_root", type=str, default="")
    ap.add_argument("--canon_shift", type=str, default="sim_low")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shifts", type=str, default="sim_low,sim_mid,sim_high")
    ap.add_argument("--milestones", type=str, default="")
    ap.add_argument(
        "--config_module",
        type=str,
        default=os.environ.get("QUAL_TRAIN_CONFIG", _DEFAULT_CFG),
        help="Train config module for default LR and FINETUNE_STEP_MILESTONES.",
    )
    ap.add_argument(
        "--run_label",
        type=str,
        default="exp35",
        help="W&B group prefix tag, e.g. exp35 or exp36.",
    )
    ap.add_argument("--finetune_lr", type=float, default=-1.0)
    ap.add_argument("--suffix", type=str, default="_3")
    ap.add_argument("--train_d_x", type=str, default="5,6,7,9,10")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--force_finetune", action="store_true")
    ap.add_argument("--skip_finetune", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--skip_zs", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="duo-quality-suite")
    args = ap.parse_args()

    _cfg = importlib.import_module(str(args.config_module))
    _ft_lr = float(args.finetune_lr)
    if _ft_lr <= 0.0:
        _ft_lr = float(_cfg.FINETUNE_LR)
    _run_label = str(args.run_label).strip() or "exp35"

    duo_root = Path(args.duo_root).resolve()
    suffix = str(args.suffix)
    sweep_root = Path(args.sweep_root or duo_root / "results/quality_bundle_3_5/fs_step_sweep").resolve()
    train_root = Path(args.train_root or duo_root / "results/quality_training_3").resolve()
    univ = train_root / f"dtrain_universal_seed{int(args.seed)}"
    main_ckpt = univ / "checkpoints"
    task_emb = univ / "task_text_embeds_TxE.npy"
    if not main_ckpt.is_dir():
        raise SystemExit(f"missing main checkpoints: {main_ckpt}")

    _milestones = getattr(_cfg, "FINETUNE_STEP_MILESTONES", (500, 1000, 1500, 2000, 2500))
    ms = [int(x.strip()) for x in (args.milestones or ",".join(map(str, _milestones))).split(",") if x.strip()]
    shifts = [s.strip() for s in str(args.shifts).split(",") if s.strip()]
    canon_pkl = duo_root / "generated_datasets" / _pkl_tag(str(args.canon_shift), suffix)
    n_tasks = len(str(args.train_d_x).split(","))
    horizon = int(os.environ.get("HORIZON", "64"))
    ctx_train = int(_cfg.CONTEXT_LENGTH_TRAIN)
    ctx_fs = int(_cfg.CONTEXT_LENGTH_FEWSHOT)
    text_encoder = os.environ.get("TEXT_ENCODER", "sentence-transformers/all-MiniLM-L6-v2")
    uniso_root = Path(os.environ.get("UNISO_ROOT", str(duo_root.parent / "UniSO"))).resolve()

    sweep_root.mkdir(parents=True, exist_ok=True)
    (sweep_root / "sweep_meta.txt").write_text(
        f"config_module={args.config_module}\nrun_label={_run_label}\n"
        f"finetune_lr={_ft_lr}\nmilestones={ms}\nshifts={shifts}\ncanon_pkl={canon_pkl}\n",
        encoding="utf-8",
    )

    for shift in shifts:
        tag = _pkl_tag(shift, suffix)
        shift_tag = tag.replace("exp1_", "")
        pkl_dir = duo_root / "generated_datasets" / tag
        uniso_dir = uniso_root / f"data_exp1_{shift_tag}"
        shift_dir = sweep_root / shift_tag

        if not args.skip_eval and not args.skip_zs:
            zs_eval = shift_dir / "eval_zs"
            if not _zs_eval_done(zs_eval):
                _eval_suite(
                    python=str(args.python),
                    duo_root=duo_root,
                    pkl_dir=pkl_dir,
                    uniso_dir=uniso_dir,
                    main_ckpt_dir=main_ckpt,
                    fs_ckpt_dir=main_ckpt,
                    eval_out=zs_eval,
                    task_emb=task_emb,
                    horizon=horizon,
                    ctx_train=ctx_train,
                    ctx_fs=ctx_fs,
                    n_tasks=n_tasks,
                    text_encoder=text_encoder,
                    device=str(args.device),
                    log_path=shift_dir / "logs" / "eval_zs.log",
                    shift_tag=shift_tag,
                    wandb_project=str(args.wandb_project).strip(),
                    run_label=_run_label,
                    phases="shift_zero_shot",
                )

        for m in ms:
            step_dir = shift_dir / f"step_{m}"
            fs_dir = step_dir / "fs_checkpoints"
            if not args.skip_finetune and (
                bool(args.force_finetune) or not _fs_ckpts_done(fs_dir)
            ):
                _finetune_milestone(
                    python=str(args.python),
                    duo_root=duo_root,
                    canon_pkl=canon_pkl,
                    pkl_dir_ft=pkl_dir,
                    main_ckpt_dir=main_ckpt,
                    fs_out_dir=fs_dir,
                    task_emb=task_emb,
                    milestone=m,
                    finetune_lr=_ft_lr,
                    horizon=horizon,
                    ctx_fs=ctx_fs,
                    n_tasks=n_tasks,
                    device=str(args.device),
                    log_path=step_dir / "logs" / "finetune.log",
                    shift_tag=shift_tag,
                    wandb_project=str(args.wandb_project).strip(),
                    run_label=_run_label,
                    force=bool(args.force_finetune),
                )
            if not args.skip_eval:
                ev_dir = step_dir / "eval"
                if not _fs_eval_done(ev_dir):
                    _eval_suite(
                        python=str(args.python),
                        duo_root=duo_root,
                        pkl_dir=pkl_dir,
                        uniso_dir=uniso_dir,
                        main_ckpt_dir=main_ckpt,
                        fs_ckpt_dir=fs_dir,
                        eval_out=ev_dir,
                        task_emb=task_emb,
                        horizon=horizon,
                        ctx_train=ctx_train,
                        ctx_fs=ctx_fs,
                        n_tasks=n_tasks,
                        text_encoder=text_encoder,
                        device=str(args.device),
                        log_path=step_dir / "logs" / "eval.log",
                        shift_tag=f"{shift_tag}_step{m}",
                        wandb_project=str(args.wandb_project).strip(),
                        run_label=_run_label,
                        phases="shift_few_shot_tail10p,shift_few_shot_tail20p,shift_few_shot_tail50p",
                    )

    rank_py = duo_root / "QualityExperiment" / "export_quality_exp35_fs_step_rank.py"
    _run(
        [
            str(args.python),
            str(rank_py),
            "--sweep_root",
            str(sweep_root),
            "--out_dir",
            str(sweep_root / "analysis_table"),
        ],
        log_path=sweep_root / "logs" / "export_rank.log",
        cwd=duo_root,
    )
    _done_stamp = sweep_root / "SWEEP_COMPLETE.txt"
    _done_stamp.write_text(
        f"status=complete\nsweep_root={sweep_root}\nconfig={args.config_module}\n"
        f"run_label={_run_label}\nfinetune_lr={_ft_lr}\nmilestones={ms}\n",
        encoding="utf-8",
    )
    print(f"[done] sweep_root={sweep_root}")
    print(f"[done] stamp={_done_stamp}")


if __name__ == "__main__":
    main()

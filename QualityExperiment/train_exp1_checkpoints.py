#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train four DUO checkpoints (st_duo / st_text / mt_label / mt_text) for QualityExperiment on exp1 PKLs,
then optional few-shot finetune on the test-task few-shot PKL (proxy task index for multitask heads).

English doc: Checkpoints match ``trace_sampling.build_diffusion`` layouts used by ``run_quality_suite``.
中文注释: 输出格式与 ``duo_train_and_sample`` 一致（ema 整模 state_dict）。
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from diffuser.datasets.sequence import PointRegretDataset
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.models.temporal import TemporalUnet
from diffuser.utils.training import Trainer


Variant = Literal["st_duo", "st_text", "mt_label", "mt_text"]


def _parse_train_index(stem: str) -> int | None:
    m = re.match(r"^D_train_(\d+)_", stem)
    return int(m.group(1)) if m else None


def _load_pkl5(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, (list, tuple)) or len(obj) < 5:
        raise ValueError(f"expected 5-tuple pkl, got {path}")
    pts, vals, pr, rtg, ts = obj[:5]
    return pts, vals, pr, rtg, ts


def _merge_multitask_train_pkls(train_pkls: list[Path]) -> Path:
    """Concatenate single-task PKLs into a 7-tuple multitask PKL (task id 0..K-1)."""
    if not train_pkls:
        raise SystemExit("no train pkls to merge")
    chunks_pts: list[torch.Tensor] = []
    chunks_vals: list[torch.Tensor] = []
    chunks_pr: list[torch.Tensor] = []
    chunks_rtg: list[torch.Tensor] = []
    chunks_ts: list[torch.Tensor] = []
    chunks_tid: list[torch.Tensor] = []
    tasks_list: list[str] = []
    for k, p in enumerate(train_pkls):
        pts, vals, pr, rtg, ts = _load_pkl5(p)
        n = int(pts.shape[0])
        chunks_pts.append(pts)
        chunks_vals.append(vals)
        chunks_pr.append(pr)
        chunks_rtg.append(rtg)
        chunks_ts.append(ts)
        chunks_tid.append(torch.full((n, 1), k, dtype=torch.long))
        tasks_list.append(f"D_train_{k + 1}")
    merged = [
        torch.cat(chunks_pts, dim=0),
        torch.cat(chunks_vals, dim=0),
        torch.cat(chunks_pr, dim=0),
        torch.cat(chunks_rtg, dim=0),
        torch.cat(chunks_ts, dim=0),
        torch.cat(chunks_tid, dim=0),
        tasks_list,
    ]
    out = train_pkls[0].parent / "_quality_mt_train_merged.pkl"
    with out.open("wb") as f:
        pickle.dump(merged, f)
    print(f"[merge] wrote {out} n_traj={merged[0].shape[0]} K={len(tasks_list)}")
    return out


def _fewshot_proxy_multitask(
    fewshot_pkl: Path, *, num_tasks: int, proxy_task_idx: int
) -> Path:
    """Wrap few-shot 5-tuple as 7-tuple; all trajectories use ``proxy_task_idx`` (shift ZS proxy)."""
    pts, vals, pr, rtg, ts = _load_pkl5(fewshot_pkl)
    n = int(pts.shape[0])
    tid = torch.full((n, 1), int(proxy_task_idx), dtype=torch.long)
    tasks_list = [f"D_train_{i + 1}" for i in range(int(num_tasks))]
    obj = [pts, vals, pr, rtg, ts, tid, tasks_list]
    out = fewshot_pkl.parent / f"_quality_fs_proxy_{fewshot_pkl.stem}.pkl"
    with out.open("wb") as f:
        pickle.dump(obj, f)
    print(f"[fs] wrote {out} proxy_task_idx={proxy_task_idx}")
    return out


def _find_train_merged(pkl_dir: Path) -> Path | None:
    hits = sorted(pkl_dir.glob("train_merged_h*_n*_lat*.pkl"))
    if not hits:
        hits = sorted(pkl_dir.glob("train_merged_h*_n*_dx*.pkl"))
    return hits[0] if hits else None


def _discover_d_train_pkls(pkl_dir: Path) -> list[Path]:
    cands: list[tuple[int, Path]] = []
    for p in pkl_dir.glob("D_train_*_h*.pkl"):
        if "fewshot" in p.name:
            continue
        if "_quality_" in p.name:
            continue
        idx = _parse_train_index(p.name)
        if idx is None:
            continue
        cands.append((idx, p))
    cands.sort(key=lambda x: x[0])
    return [p for _, p in cands]


def _find_fewshot_pkl(pkl_dir: Path) -> Path | None:
    hits = sorted(pkl_dir.glob("D_test_*_fewshot*lat*.pkl"))
    if not hits:
        hits = sorted(pkl_dir.glob("D_test_*_fewshot*.pkl"))
    return hits[0] if hits else None


def _build_diffusion(
    dataset: PointRegretDataset,
    *,
    horizon: int,
    variant: Variant,
    text_dim: int,
    num_tasks: int,
) -> GaussianDiffusion:
    transition_dim = int(dataset.observation_dim + dataset.action_dim)
    if variant == "st_duo":
        tc, nt, tx, ted = False, 1, False, 384
    elif variant == "st_text":
        tc, nt, tx, ted = False, 1, True, int(text_dim)
    elif variant == "mt_label":
        tc, nt, tx, ted = True, int(num_tasks), False, 384
    else:
        tc, nt, tx, ted = True, int(num_tasks), True, int(text_dim)
    model = TemporalUnet(
        horizon=int(horizon),
        transition_dim=transition_dim,
        cond_dim=0,
        dim=128,
        dim_mults=(1, 2, 4),
        returns_condition=False,
        task_condition=bool(tc),
        num_tasks=int(nt),
        condition_dropout=0.0,
        text_condition=bool(tx),
        text_embed_input_dim=int(ted),
    )
    return GaussianDiffusion(
        model=model,
        horizon=int(horizon),
        observation_dim=int(dataset.observation_dim),
        action_dim=int(dataset.action_dim),
        n_timesteps=1000,
        n_sample_timesteps=200,
        loss_type="l1",
        clip_denoised=False,
        predict_epsilon=True,
        returns_condition=False,
    )


def _dataset_for_variant(
    pkl_path: Path,
    *,
    horizon: int,
    variant: Variant,
    task_embeds: np.ndarray | None,
    num_tasks: int,
) -> PointRegretDataset:
    if variant in ("st_duo", "st_text"):
        te = None
        if variant == "st_text":
            if task_embeds is None:
                raise SystemExit("st_text requires task_text_embeds npy")
            # 中文注释: 单任务 merged 上用全体任务嵌入均值，避免只贴 D_train_1 文本。
            te = np.mean(np.asarray(task_embeds, dtype=np.float32), axis=0, keepdims=True)
        return PointRegretDataset(
            horizon=int(horizon),
            data_path=str(pkl_path),
            context_length=0,
            regret=False,
            include_returns=False,
            task_name=None,
            task_text_embeds=te,
            include_task_idx=False,
        )
    te = None
    if variant == "mt_text":
        if task_embeds is None:
            raise SystemExit("mt_text requires task_text_embeds npy")
        te = np.asarray(task_embeds, dtype=np.float32)
    return PointRegretDataset(
        horizon=int(horizon),
        data_path=str(pkl_path),
        context_length=0,
        regret=False,
        include_returns=False,
        task_name=None,
        task_text_embeds=te,
        include_task_idx=True,
    )


def _train_one(
    *,
    variant: Variant,
    train_pkl: Path,
    task_embeds: np.ndarray | None,
    num_tasks: int,
    horizon: int,
    train_steps: int,
    batch_size: int,
    lr: float,
    grad_accum: int,
    device: str,
    out_ckpt: Path,
    load_ckpt: Path | None,
) -> None:
    ds = _dataset_for_variant(
        train_pkl, horizon=horizon, variant=variant, task_embeds=task_embeds, num_tasks=num_tasks
    )
    text_dim = int(task_embeds.shape[1]) if task_embeds is not None else 384
    diffusion = _build_diffusion(
        ds, horizon=horizon, variant=variant, text_dim=text_dim, num_tasks=num_tasks
    )
    diffusion = diffusion.to(torch.device(device))
    trainer = Trainer(
        diffusion_model=diffusion,
        proxy_model=None,
        dataset=ds,
        proxy_dataset=None,
        renderer=None,
        ema_decay=0.995,
        train_batch_size=int(batch_size),
        train_lr=float(lr),
        proxy_train_lr=float(lr),
        gradient_accumulate_every=int(grad_accum),
        log_freq=max(10, int(train_steps) // 10),
        sample_freq=0,
        save_freq=int(train_steps) + 1,
        proxy_save_freq=10**9,
        train_device=str(device),
        save_checkpoints=False,
    )
    if load_ckpt is not None:
        trainer.load_from_path(str(load_ckpt))
    else:
        trainer.step = 0
    setattr(trainer, "_total_train_steps", int(train_steps))
    trainer.train(int(train_steps))
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(trainer.step),
            "model": trainer.model.state_dict(),
            "ema": trainer.ema_model.state_dict(),
        },
        str(out_ckpt),
    )
    print(f"[train] {variant} -> {out_ckpt}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train exp1 QualityExperiment DUO quartet (+ optional FS).")
    ap.add_argument("--pkl_dir", type=str, required=True, help="e.g. generated_datasets/exp1_gap0p500")
    ap.add_argument(
        "--task_text_embeds_npy",
        type=str,
        default="",
        help="Required for st_text / mt_text (same as run_quality_suite).",
    )
    ap.add_argument("--out_dir", type=str, required=True, help="Directory for *.pt checkpoints.")
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument(
        "--mt_num_tasks",
        type=int,
        default=5,
        help="Number of D_train_* instances merged for mt_* (match run_exp1 --n_train_tasks).",
    )
    ap.add_argument(
        "--train_steps",
        type=int,
        default=0,
        help="If >0, use this many Trainer steps; else train_rounds * steps_per_round.",
    )
    ap.add_argument(
        "--finetune_steps",
        type=int,
        default=0,
        help="If >0, use this many finetune steps; else finetune_rounds * steps_per_round.",
    )
    ap.add_argument(
        "--train_rounds",
        type=int,
        default=100,
        help="Main training 'rounds'; steps = rounds * steps_per_round when train_steps not set.",
    )
    ap.add_argument(
        "--finetune_rounds",
        type=int,
        default=50,
        help="Few-shot finetune rounds; steps = rounds * steps_per_round when finetune_steps not set.",
    )
    ap.add_argument(
        "--steps_per_round",
        type=int,
        default=100,
        help="Gradient steps per round (轮); total steps = rounds * this value.",
    )
    ap.add_argument("--skip_finetune", action="store_true")
    ap.add_argument("--zs_task_idx", type=int, default=0, help="Proxy task index for FS multitask PKL.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    spr = int(args.steps_per_round)
    train_steps = int(args.train_steps) if int(args.train_steps) > 0 else int(args.train_rounds) * spr
    finetune_steps = int(args.finetune_steps) if int(args.finetune_steps) > 0 else int(args.finetune_rounds) * spr
    if int(args.train_steps) > 0:
        print(f"[cfg] train_steps={train_steps} (explicit --train_steps)")
    else:
        print(
            f"[cfg] train_steps={train_steps} "
            f"({int(args.train_rounds)} rounds * {spr} steps_per_round)"
        )
    if int(args.finetune_steps) > 0:
        print(f"[cfg] finetune_steps={finetune_steps} (explicit --finetune_steps)")
    else:
        print(
            f"[cfg] finetune_steps={finetune_steps} "
            f"({int(args.finetune_rounds)} rounds * {spr} steps_per_round)"
        )

    pkl_dir = Path(args.pkl_dir).resolve()
    if not pkl_dir.is_dir():
        raise SystemExit(f"missing pkl_dir: {pkl_dir}")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    task_embeds: np.ndarray | None = None
    if str(args.task_text_embeds_npy).strip():
        task_embeds = np.load(str(args.task_text_embeds_npy))
        if task_embeds.ndim != 2:
            raise SystemExit("task_text_embeds_npy must be [T, E]")

    merged_st = _find_train_merged(pkl_dir)
    if merged_st is None:
        raise SystemExit(f"no train_merged latent/dx pkl under {pkl_dir}; run run_exp1.py for this gap.")

    d_train_pkls = _discover_d_train_pkls(pkl_dir)
    if len(d_train_pkls) < int(args.mt_num_tasks):
        raise SystemExit(f"expected >= {args.mt_num_tasks} D_train_* pkls, found {len(d_train_pkls)}")
    d_train_pkls = d_train_pkls[: int(args.mt_num_tasks)]
    mt_pkl = _merge_multitask_train_pkls(d_train_pkls)

    horizon = int(args.horizon)
    K = int(args.mt_num_tasks)
    variants: tuple[Variant, ...] = ("st_duo", "st_text", "mt_label", "mt_text")
    for v in variants:
        if v in ("st_duo", "st_text"):
            data_p = merged_st
        else:
            data_p = mt_pkl
        _train_one(
            variant=v,
            train_pkl=data_p,
            task_embeds=task_embeds,
            num_tasks=K,
            horizon=horizon,
            train_steps=int(train_steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            grad_accum=int(args.grad_accum),
            device=str(args.device),
            out_ckpt=out_dir / f"ckpt_{v}.pt",
            load_ckpt=None,
        )

    if bool(args.skip_finetune):
        print("[fs] skip_finetune set; done.")
        return

    fs_src = _find_fewshot_pkl(pkl_dir)
    if fs_src is None:
        print(f"[warn] no fewshot pkl under {pkl_dir}; skip finetune.")
        return
    fs_mt = _fewshot_proxy_multitask(fs_src, num_tasks=K, proxy_task_idx=int(args.zs_task_idx))
    fs_st = fs_src

    for v in variants:
        base = out_dir / f"ckpt_{v}.pt"
        if v in ("st_duo", "st_text"):
            data_p = fs_st
        else:
            data_p = fs_mt
        _train_one(
            variant=v,
            train_pkl=data_p,
            task_embeds=task_embeds,
            num_tasks=K,
            horizon=horizon,
            train_steps=int(finetune_steps),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            grad_accum=int(args.grad_accum),
            device=str(args.device),
            out_ckpt=out_dir / f"ckpt_{v}_fs.pt",
            load_ckpt=base,
        )


if __name__ == "__main__":
    main()

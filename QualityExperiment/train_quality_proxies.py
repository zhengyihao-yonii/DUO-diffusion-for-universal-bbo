#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Quality proxies without touching diffusion checkpoints.

English doc: (1) Per-task proxies for D_train eval (same PKL as train_domain sampling).
(2) Colocated proxies for main / few-shot ckpts (merged or fewshot PKL).
中文注释: 仅训 proxy；不重训扩散主训/微调。
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from QualityExperiment.quality_proxy import (
    proxy_path_for_ckpt,
    proxy_path_for_train_domain,
    train_and_save_proxy,
)
from QualityExperiment.train_exp1_checkpoints import (
    Variant,
    _discover_d_train_pkls,
    _dataset_for_variant,
    _fewshot_proxy_multitask,
    _find_fewshot_pkl_for_tail,
    _find_legacy_fewshot_pkl,
    _find_train_merged,
    _merge_multitask_train_pkls,
    _parse_ckpt_variant,
    _proxy_cfg,
    _resolve_quality_cfg,
)

_FS_TAIL_TAGS: tuple[str, ...] = ("tail10p", "tail20p", "tail50p")


def _parse_train_index(stem: str) -> int | None:
    m = re.match(r"^D_train_(\d+)_", stem)
    return int(m.group(1)) if m else None


def _task_id_from_pkl(pkl: Path) -> str:
    m = re.match(r"^(D_train_\d+)_", pkl.name)
    if m:
        return m.group(1)
    idx = _parse_train_index(pkl.name)
    if idx is not None:
        return f"D_train_{idx}"
    raise ValueError(f"cannot parse D_train task id from {pkl.name}")


def _train_domain_proxies(
    *,
    pkl_dir: Path,
    ckpt_dir: Path,
    variants: tuple[Variant, ...],
    task_embeds: Any,
    num_tasks: int,
    horizon: int,
    context_length: int,
    device: str,
    proxy_steps: int,
    proxy_lr: float,
    proxy_hidden: int,
    proxy_ensembles: int,
    force_proxy: bool,
) -> None:
    d_train = _discover_d_train_pkls(pkl_dir)
    if not d_train:
        raise SystemExit(f"no D_train_* pkls under {pkl_dir}")
    for pkl in d_train:
        tid = _task_id_from_pkl(pkl)
        for v in variants:
            ckpt = ckpt_dir / f"ckpt_{v}.pt"
            if not ckpt.is_file():
                print(f"[skip] train_domain proxy {tid}/{v}: missing {ckpt}")
                continue
            out_p = proxy_path_for_train_domain(ckpt, tid)
            if out_p.is_file() and not force_proxy:
                print(f"[skip] train_domain proxy exists {out_p.name}")
                continue
            ds = _dataset_for_variant(
                pkl,
                horizon=horizon,
                variant=v,
                task_embeds=task_embeds,
                num_tasks=num_tasks,
                context_length=context_length,
            )
            train_and_save_proxy(
                train_pkl=pkl,
                dataset=ds,
                device=device,
                save_path=out_p,
                n_steps=proxy_steps,
                proxy_lr=proxy_lr,
                hidden_dim=proxy_hidden,
                n_ensembles=proxy_ensembles,
            )


def _colocated_proxies_in_dir(
    *,
    ckpt_dir: Path,
    pkl_dir: Path,
    pkl_dir_finetune: Path | None,
    variants: tuple[Variant, ...],
    task_embeds: Any,
    num_tasks: int,
    horizon: int,
    ctx_main: int,
    ctx_fs: int,
    zs_task_idx: int,
    device: str,
    proxy_steps: int,
    proxy_lr: float,
    proxy_hidden: int,
    proxy_ensembles: int,
    force_proxy: bool,
    main_only: bool,
) -> None:
    merged_st = _find_train_merged(pkl_dir)
    if merged_st is None:
        raise SystemExit(f"no train_merged under {pkl_dir}")
    d_train = _discover_d_train_pkls(pkl_dir)
    mt_pkl = _merge_multitask_train_pkls(d_train[: int(num_tasks)])
    pkl_ft = pkl_dir_finetune if pkl_dir_finetune is not None else pkl_dir

    for ckpt in sorted(ckpt_dir.glob("ckpt_*.pt")):
        parsed = _parse_ckpt_variant(ckpt)
        if parsed is None:
            continue
        v, tail = parsed
        proxy_out = proxy_path_for_ckpt(ckpt)
        if proxy_out.is_file() and not force_proxy:
            print(f"[skip] colocated proxy exists {proxy_out.name}")
            continue
        if tail == "__main__":
            if main_only is False and ckpt_dir != Path(ckpt).resolve():
                pass
            data_p = merged_st if v in ("st_duo", "st_text") else mt_pkl
            ctx = ctx_main
        else:
            if main_only:
                continue
            fs_src = (
                _find_fewshot_pkl_for_tail(pkl_ft, str(tail))
                if tail is not None
                else _find_legacy_fewshot_pkl(pkl_ft)
            )
            if fs_src is None:
                print(f"[warn] no fewshot pkl for {ckpt.name}")
                continue
            data_p = (
                fs_src
                if v in ("st_duo", "st_text")
                else _fewshot_proxy_multitask(
                    fs_src, num_tasks=int(num_tasks), proxy_task_idx=int(zs_task_idx)
                )
            )
            ctx = ctx_fs
        ds = _dataset_for_variant(
            data_p,
            horizon=horizon,
            variant=v,
            task_embeds=task_embeds,
            num_tasks=num_tasks,
            context_length=ctx,
        )
        train_and_save_proxy(
            train_pkl=data_p,
            dataset=ds,
            device=device,
            save_path=proxy_out,
            n_steps=proxy_steps,
            proxy_lr=proxy_lr,
            hidden_dim=proxy_hidden,
            n_ensembles=proxy_ensembles,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Quality proxies only (no diffusion).")
    ap.add_argument("--pkl_dir", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--pkl_dir_finetune", type=str, default="")
    ap.add_argument("--fs_ckpt_dir", type=str, default="", help="Few-shot ckpt dir (optional).")
    ap.add_argument("--task_text_embeds_npy", type=str, default="")
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--context_length_train", type=int, default=32)
    ap.add_argument("--context_length_fewshot", type=int, default=16)
    ap.add_argument("--mt_num_tasks", type=int, default=4)
    ap.add_argument("--zs_task_idx", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--train_domain", action="store_true", help="Per-task D_train proxies.")
    ap.add_argument("--colocated_main", action="store_true", help="proxy_* next to main ckpt_*.pt.")
    ap.add_argument("--colocated_fs", action="store_true", help="proxy_* for fs ckpts in --fs_ckpt_dir.")
    ap.add_argument("--force_proxy", action="store_true")
    args = ap.parse_args()

    import numpy as np

    cfg = _resolve_quality_cfg()
    use_proxy, proxy_steps, proxy_lr, proxy_hid, proxy_ens = _proxy_cfg(cfg)
    if not use_proxy:
        print("[skip] USE_PROXY_FILTER=0 in config")
        return

    task_embeds = None
    if str(args.task_text_embeds_npy).strip():
        task_embeds = np.load(str(args.task_text_embeds_npy))

    variants: tuple[Variant, ...] = ("st_duo", "st_text", "mt_label", "mt_text")
    pkl_dir = Path(args.pkl_dir).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    pkl_ft = Path(str(args.pkl_dir_finetune).strip()).resolve() if str(args.pkl_dir_finetune).strip() else None

    if bool(args.train_domain):
        _train_domain_proxies(
            pkl_dir=pkl_dir,
            ckpt_dir=ckpt_dir,
            variants=variants,
            task_embeds=task_embeds,
            num_tasks=int(args.mt_num_tasks),
            horizon=int(args.horizon),
            context_length=int(args.context_length_train),
            device=str(args.device),
            proxy_steps=proxy_steps,
            proxy_lr=proxy_lr,
            proxy_hidden=proxy_hid,
            proxy_ensembles=proxy_ens,
            force_proxy=bool(args.force_proxy),
        )

    if bool(args.colocated_main):
        _colocated_proxies_in_dir(
            ckpt_dir=ckpt_dir,
            pkl_dir=pkl_dir,
            pkl_dir_finetune=pkl_ft,
            variants=variants,
            task_embeds=task_embeds,
            num_tasks=int(args.mt_num_tasks),
            horizon=int(args.horizon),
            ctx_main=int(args.context_length_train),
            ctx_fs=int(args.context_length_fewshot),
            zs_task_idx=int(args.zs_task_idx),
            device=str(args.device),
            proxy_steps=proxy_steps,
            proxy_lr=proxy_lr,
            proxy_hidden=proxy_hid,
            proxy_ensembles=proxy_ens,
            force_proxy=bool(args.force_proxy),
            main_only=True,
        )

    if bool(args.colocated_fs):
        fs_dir = Path(str(args.fs_ckpt_dir).strip()).resolve()
        if not fs_dir.is_dir():
            raise SystemExit(f"fs_ckpt_dir missing: {fs_dir}")
        if pkl_ft is None:
            raise SystemExit("colocated_fs requires --pkl_dir_finetune")
        _colocated_proxies_in_dir(
            ckpt_dir=fs_dir,
            pkl_dir=pkl_dir,
            pkl_dir_finetune=pkl_ft,
            variants=variants,
            task_embeds=task_embeds,
            num_tasks=int(args.mt_num_tasks),
            horizon=int(args.horizon),
            ctx_main=int(args.context_length_train),
            ctx_fs=int(args.context_length_fewshot),
            zs_task_idx=int(args.zs_task_idx),
            device=str(args.device),
            proxy_steps=proxy_steps,
            proxy_lr=proxy_lr,
            proxy_hidden=proxy_hid,
            proxy_ensembles=proxy_ens,
            force_proxy=bool(args.force_proxy),
            main_only=False,
        )


if __name__ == "__main__":
    main()

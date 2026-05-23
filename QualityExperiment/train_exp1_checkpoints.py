#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train four DUO checkpoints (st_duo / st_text / mt_label / mt_text) for QualityExperiment on exp1 PKLs,
then optional few-shot finetune on the test-task few-shot PKL (proxy task index for multitask heads).

English doc: **Main training** fits all four variants on **D_train**-domain data only: ``st_*`` use the
latent ``train_merged`` PKL (pool built from ``D_train_*`` in ``run_exp1``); ``mt_*`` use a merged
multitask PKL over ``D_train_1..K``. **Finetune** adapts on ``D_test_*_fewshot_*`` from ``--pkl_dir_finetune``
when set, otherwise from ``--pkl_dir``. Use ``--finetune_out_dir`` to write FS ckpts separately from main
``--out_dir``; ``--skip_main`` / ``--skip_finetune`` split phases for a universal-then-per-gap pipeline.
Checkpoints match ``trace_sampling.build_diffusion`` layouts used by ``run_quality_suite``.
Optional W&B: pass ``--wandb_project`` (and ``--wandb_group``); ``diffuser.utils.training.Trainer`` logs
loss scalars when a run is active (``_safe_wandb_log``).
中文注释: **主训练**仅在 **D_train** 域数据上训练四个变体；微调数据来自 ``--pkl_dir_finetune``（若设）否则 ``--pkl_dir``。
FS 权重可写入 ``--finetune_out_dir``。输出为 ema 整模 state_dict；训练曲线见 train_metrics/*.jsonl 或 W&B。
"""
from __future__ import annotations

import argparse
import importlib
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import exp1_diffusion_aligned as _e1a
from config import quality_exp1_train as _qtr
from QualityExperiment.quality_proxy import proxy_path_for_ckpt, train_and_save_proxy
from QualityExperiment.quality_text_condition_cfg import CONDITION_GUIDANCE_W_TEXT as _Q_W_TEXT
from diffuser.datasets.sequence import PointRegretDataset
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.models.temporal import TemporalUnet
from diffuser.utils.training import Trainer


Variant = Literal["st_duo", "st_text", "mt_label", "mt_text"]

_FS_TAIL_TAGS: tuple[str, ...] = ("tail10p", "tail20p", "tail50p")
_VARIANTS_ORDER: tuple[Variant, ...] = ("mt_text", "mt_label", "st_text", "st_duo")


def _resolve_quality_cfg() -> Any:
    """English doc: ``QUAL_TRAIN_CONFIG`` env (e.g. config.quality_exp5_train) overrides defaults."""
    name = os.environ.get("QUAL_TRAIN_CONFIG", "").strip()
    if not name:
        return _qtr
    return importlib.import_module(name)


def _proxy_cfg(cfg: Any) -> tuple[bool, int, float, int, int]:
    use = int(getattr(cfg, "USE_PROXY_FILTER", 0)) == 1
    return (
        use,
        int(getattr(cfg, "PROXY_N_TRAIN_STEPS", 1000)),
        float(getattr(cfg, "PROXY_LR", 2e-4)),
        int(getattr(cfg, "PROXY_HIDDEN_DIM", 256)),
        int(getattr(cfg, "PROXY_N_ENSEMBLES", 5)),
    )


def _parse_ckpt_variant(ckpt_path: Path) -> tuple[Variant, str | None] | None:
    """Return (variant, fs_tail_tag|None). ``None`` tail = legacy ``_fs``; main ckpt tail is sentinel ``__main__``."""
    stem = ckpt_path.stem
    if not stem.startswith("ckpt_"):
        return None
    rest = stem[5:]
    for v in _VARIANTS_ORDER:
        if not rest.startswith(v):
            continue
        suffix = rest[len(v) :]
        if suffix == "":
            return v, "__main__"
        if suffix.startswith("_fs"):
            tail = suffix[4:]
            if tail.startswith("_"):
                tail = tail[1:]
            return v, (tail if tail else None)
        return None
    return None


def _train_proxy_for_ckpt(
    *,
    variant: Variant,
    train_pkl: Path,
    ckpt_path: Path,
    task_embeds: np.ndarray | None,
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
    proxy_out = proxy_path_for_ckpt(ckpt_path)
    if proxy_out.is_file() and not force_proxy:
        print(f"[skip] proxy exists {proxy_out}")
        return
    if not ckpt_path.is_file():
        print(f"[skip] proxy: missing ckpt {ckpt_path}")
        return
    ds = _dataset_for_variant(
        train_pkl,
        horizon=horizon,
        variant=variant,
        task_embeds=task_embeds,
        num_tasks=num_tasks,
        context_length=int(context_length),
    )
    train_and_save_proxy(
        train_pkl=train_pkl,
        dataset=ds,
        device=str(device),
        save_path=proxy_out,
        n_steps=int(proxy_steps),
        proxy_lr=float(proxy_lr),
        hidden_dim=int(proxy_hidden),
        n_ensembles=int(proxy_ensembles),
    )


def _read_ckpt_step(path: Path) -> int | None:
    """English doc: Optimizer step counter stored in Quality quartet checkpoints."""
    if not path.is_file():
        return None
    try:
        obj = torch.load(str(path), map_location="cpu")
        if isinstance(obj, dict) and "step" in obj:
            return int(obj["step"])
    except Exception:
        return None
    return None


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


def _find_fewshot_pkl_for_tail(pkl_dir: Path, tail_tag: str) -> Path | None:
    hits = sorted(pkl_dir.glob(f"D_test_*_fewshot_{tail_tag}_*.pkl"))
    return hits[0] if hits else None


def _find_legacy_fewshot_pkl(pkl_dir: Path) -> Path | None:
    """English doc: Pre–tail-tag naming: ``*_fewshot_h*_lat*.pkl`` without ``fewshot_tail``."""
    for p in sorted(pkl_dir.glob("D_test_*_fewshot_h*_lat*.pkl")):
        if "fewshot_tail" in p.name:
            continue
        return p
    for p in sorted(pkl_dir.glob("D_test_*_fewshot*.pkl")):
        if "fewshot_tail" in p.name:
            continue
        return p
    return None


def _build_diffusion(
    dataset: PointRegretDataset,
    *,
    horizon: int,
    variant: Variant,
    text_dim: int,
    num_tasks: int,
) -> GaussianDiffusion:
    """English doc: UNet + diffusion match ``config/ant_config.py`` (main train line)."""
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
        dim=int(_e1a.UNET_DIM),
        dim_mults=_e1a.UNET_DIM_MULTS,
        returns_condition=False,
        task_condition=bool(tc),
        num_tasks=int(nt),
        condition_dropout=float(_e1a.UNET_CONDITION_DROPOUT),
        text_condition=bool(tx),
        text_embed_input_dim=int(ted),
    )
    return GaussianDiffusion(
        model=model,
        horizon=int(horizon),
        observation_dim=int(dataset.observation_dim),
        action_dim=int(dataset.action_dim),
        n_timesteps=int(_e1a.N_DIFFUSION_STEPS),
        n_sample_timesteps=int(_e1a.N_SAMPLE_TIMESTEPS),
        loss_type=str(_e1a.LOSS_TYPE),
        clip_denoised=bool(_e1a.CLIP_DENOISED),
        predict_epsilon=bool(_e1a.PREDICT_EPSILON),
        action_weight=float(_e1a.ACTION_WEIGHT),
        returns_condition=False,
        condition_guidance_w=float(_e1a.CONDITION_GUIDANCE_W),
        condition_guidance_w_task=float(_e1a.CONDITION_GUIDANCE_W_TASK),
        condition_guidance_w_text=float(_Q_W_TEXT),
    )


def _dataset_for_variant(
    pkl_path: Path,
    *,
    horizon: int,
    variant: Variant,
    task_embeds: np.ndarray | None,
    num_tasks: int,
    context_length: int,
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
            context_length=int(context_length),
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
        context_length=int(context_length),
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
    context_length: int,
    train_steps: int,
    batch_size: int,
    lr: float,
    grad_accum: int,
    device: str,
    out_ckpt: Path,
    load_ckpt: Path | None,
    run_tag: str = "",
    log_freq: int,
    save_freq: int,
    wandb_project: str = "",
    wandb_group: str = "",
    wandb_run_name: str = "",
) -> None:
    prev_v = os.environ.get("QUAL_TRAIN_VARIANT")
    os.environ["QUAL_TRAIN_VARIANT"] = str(run_tag or variant)
    wb_started = False
    try:
        _wp = str(wandb_project).strip()
        if _wp:
            import wandb as wb

            from diffuser.utils.training import configure_wandb_step_axes
            from diffuser.utils.wandb_auth import init_wandb_run

            # 中文注释: 先 login 再 init（entity 覆盖共享机 ~/.netrc）
            _nm = str(wandb_run_name).strip() or f"train_{variant}__{run_tag}"
            try:
                _init_to = float(os.environ.get("WANDB_INIT_TIMEOUT", "120"))
                init_wandb_run(
                    _wp,
                    group=str(wandb_group).strip() or None,
                    name=_nm,
                    reinit=True,
                    settings=wb.Settings(init_timeout=_init_to),
                )
                configure_wandb_step_axes(include_proxy_axis=False)
                wb.config.update(
                    {
                        "variant": str(variant),
                        "run_tag": str(run_tag),
                        "train_pkl": str(train_pkl),
                        "train_steps": int(train_steps),
                        "horizon": int(horizon),
                        "context_length": int(context_length),
                    }
                )
                wb_started = True
            except Exception as exc:
                print(f"[warn] wandb.init skipped: {exc}", flush=True)

        ds = _dataset_for_variant(
            train_pkl,
            horizon=horizon,
            variant=variant,
            task_embeds=task_embeds,
            num_tasks=num_tasks,
            context_length=int(context_length),
        )
        text_dim = int(task_embeds.shape[1]) if task_embeds is not None else 384
        diffusion = _build_diffusion(
            ds, horizon=horizon, variant=variant, text_dim=text_dim, num_tasks=num_tasks
        )
        diffusion = diffusion.to(torch.device(device))
        _log_f = max(1, int(log_freq))
        _save_f = max(1, min(int(save_freq), int(train_steps)))
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
            log_freq=_log_f,
            sample_freq=0,
            save_freq=_save_f,
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
    finally:
        if wb_started:
            try:
                import wandb as wb

                wb.finish()
            except Exception:
                pass
        if prev_v is None:
            os.environ.pop("QUAL_TRAIN_VARIANT", None)
        else:
            os.environ["QUAL_TRAIN_VARIANT"] = prev_v


def main() -> None:
    ap = argparse.ArgumentParser(description="Train exp1 QualityExperiment DUO quartet (+ optional FS).")
    ap.add_argument("--pkl_dir", type=str, required=True, help="e.g. generated_datasets/exp1_gap0p500")
    ap.add_argument(
        "--pkl_dir_finetune",
        type=str,
        default="",
        help=(
            "If set: read D_test few-shot PKLs from this directory only (per eval gap). "
            "Main training data still comes from --pkl_dir (canonical D_train PKLs)."
        ),
    )
    ap.add_argument(
        "--task_text_embeds_npy",
        type=str,
        default="",
        help="Required for st_text / mt_text (same as run_quality_suite).",
    )
    ap.add_argument("--out_dir", type=str, required=True, help="Directory for *.pt checkpoints.")
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument(
        "--context_length_train",
        type=int,
        default=int(_qtr.CONTEXT_LENGTH_TRAIN),
        help="Prefix conditioning steps for main (merged / multitask) training.",
    )
    ap.add_argument(
        "--context_length_fewshot",
        type=int,
        default=int(_qtr.CONTEXT_LENGTH_FEWSHOT),
        help="Prefix conditioning steps for few-shot finetune PKLs.",
    )
    ap.add_argument(
        "--mt_num_tasks",
        type=int,
        default=4,
        help="Number of D_train_* instances merged for mt_* (match run_exp1 --n_train_tasks).",
    )
    ap.add_argument(
        "--train_steps",
        type=int,
        default=int(_qtr.N_TRAIN_STEPS),
        help="Main training optimizer steps per variant (default: config.quality_exp1_train.N_TRAIN_STEPS).",
    )
    ap.add_argument(
        "--finetune_steps",
        type=int,
        default=int(_qtr.N_FINETUNE_STEPS),
        help="Few-shot finetune steps per variant (default: config.quality_exp1_train.N_FINETUNE_STEPS).",
    )
    ap.add_argument("--skip_finetune", action="store_true")
    ap.add_argument(
        "--skip_main",
        action="store_true",
        help="Skip D_train main quartet training (only few-shot finetune; requires existing ckpt_*.pt in --out_dir).",
    )
    ap.add_argument(
        "--finetune_out_dir",
        type=str,
        default="",
        help="If set: write few-shot finetuned *.pt here; load main ckpt_*.pt from --out_dir.",
    )
    ap.add_argument("--zs_task_idx", type=int, default=0, help="Proxy task index for FS multitask PKL.")
    ap.add_argument("--batch_size", type=int, default=int(_qtr.BATCH_SIZE))
    ap.add_argument("--lr", type=float, default=float(_qtr.LR))
    ap.add_argument(
        "--finetune_lr",
        type=float,
        default=float(getattr(_qtr, "FINETUNE_LR", _qtr.LR)),
        help="Few-shot finetune Adam LR (default: FINETUNE_LR or LR from quality config).",
    )
    ap.add_argument("--grad_accum", type=int, default=int(_qtr.GRAD_ACCUM))
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--train_step_target",
        type=int,
        default=0,
        help=(
            "If >0: target total Trainer.step for every variant (st_* and mt_* share the same budget). "
            "Resumes from existing ckpt when present. If 0: use --train_steps from config/CLI; "
            "skip variant if ckpt exists unless --force_train."
        ),
    )
    ap.add_argument(
        "--force_train",
        action="store_true",
        help="When --train_step_target is 0, still run main training even if ckpt exists (overwrite).",
    )
    ap.add_argument(
        "--finetune_add_steps",
        type=int,
        default=0,
        help="If >0: load existing fs ckpt when present and run this many extra steps (additive finetune).",
    )
    ap.add_argument(
        "--force_finetune",
        action="store_true",
        help="Re-run finetune from main ckpt even if fs ckpt already exists (unless finetune_add_steps>0 loads fs).",
    )
    ap.add_argument(
        "--finetune_preserve_variants",
        type=str,
        default="",
        help="Comma-separated variants never force-overwritten (e.g. mt_text when reusing external fs ckpts).",
    )
    ap.add_argument(
        "--fs_tail_tags",
        type=str,
        default="",
        help="Comma-separated few-shot tails to finetune (e.g. tail10p,tail20p). Empty = all found.",
    )
    ap.add_argument(
        "--log_freq",
        type=int,
        default=int(_qtr.LOG_FREQ),
        help="Trainer log interval (steps); default from config.quality_exp1_train.LOG_FREQ.",
    )
    ap.add_argument(
        "--save_freq",
        type=int,
        default=int(_qtr.SAVE_FREQ),
        help="Trainer.save interval (steps); default from config.quality_exp1_train.SAVE_FREQ.",
    )
    ap.add_argument(
        "--wandb_project",
        type=str,
        default="",
        help="If non-empty: init wandb and log Trainer loss metrics (same hooks as diffuser Trainer).",
    )
    ap.add_argument(
        "--wandb_group",
        type=str,
        default="",
        help="W&B group for training runs (e.g. quality_exp1_train_gap_0p250_seed0).",
    )
    ap.add_argument(
        "--proxy_only",
        action="store_true",
        help="Only train/save proxy_*.pt for existing ckpt_*.pt (skip diffusion train).",
    )
    ap.add_argument(
        "--skip_proxy",
        action="store_true",
        help="Do not train proxy even when config USE_PROXY_FILTER=1.",
    )
    ap.add_argument(
        "--force_proxy",
        action="store_true",
        help="Re-train proxy even if proxy_*.pt already exists.",
    )
    args = ap.parse_args()
    _cfg = _resolve_quality_cfg()
    _use_proxy, _proxy_steps, _proxy_lr, _proxy_hid, _proxy_ens = _proxy_cfg(_cfg)
    if bool(args.skip_proxy):
        _use_proxy = False

    os.environ.setdefault("DUO_LOG_PER_T_LOSS", "1")
    os.environ.setdefault("DUO_LOG_PER_T_LOSS_BINS", "20")

    train_steps = int(args.train_steps)
    finetune_steps = int(args.finetune_steps)
    finetune_lr = float(args.finetune_lr)
    print(
        f"[cfg] train_steps={train_steps} lr={float(args.lr):.2e} "
        f"finetune_steps={finetune_steps} finetune_lr={finetune_lr:.2e} "
        f"(config.quality_exp1_train / FINETUNE_*)",
        flush=True,
    )
    print(f"[cfg] log_freq={int(args.log_freq)} save_freq={int(args.save_freq)}")
    print(
        f"[cfg] proxy_filter={int(_use_proxy)} proxy_steps={_proxy_steps} "
        f"proxy_lr={_proxy_lr:.2e} (module={_cfg.__name__})",
        flush=True,
    )

    pkl_dir = Path(args.pkl_dir).resolve()
    if not pkl_dir.is_dir():
        raise SystemExit(f"missing pkl_dir: {pkl_dir}")
    pkl_dir_ft = Path(str(args.pkl_dir_finetune).strip()).resolve() if str(args.pkl_dir_finetune).strip() else pkl_dir
    if not pkl_dir_ft.is_dir():
        raise SystemExit(f"missing pkl_dir_finetune: {pkl_dir_ft}")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    finetune_out = Path(str(args.finetune_out_dir).strip()).resolve() if str(args.finetune_out_dir).strip() else out_dir
    if str(args.finetune_out_dir).strip():
        finetune_out.mkdir(parents=True, exist_ok=True)

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

    pkl_label = str(pkl_dir.name)
    pkl_label_ft = str(pkl_dir_ft.name)
    _wb_proj = str(args.wandb_project).strip()
    _wb_grp = str(args.wandb_group).strip()

    horizon = int(args.horizon)
    ctx_main = int(args.context_length_train)
    ctx_fs = int(args.context_length_fewshot)
    if ctx_main > horizon or ctx_fs > horizon:
        raise SystemExit("context_length_* must be <= horizon")
    K = int(args.mt_num_tasks)
    variants: tuple[Variant, ...] = ("st_duo", "st_text", "mt_label", "mt_text")
    tgt = int(args.train_step_target)

    def _maybe_proxy(
        *,
        variant: Variant,
        data_p: Path,
        ckpt: Path,
        ctx_len: int,
    ) -> None:
        if not _use_proxy:
            return
        _train_proxy_for_ckpt(
            variant=variant,
            train_pkl=data_p,
            ckpt_path=ckpt,
            task_embeds=task_embeds,
            num_tasks=K,
            horizon=horizon,
            context_length=int(ctx_len),
            device=str(args.device),
            proxy_steps=_proxy_steps,
            proxy_lr=_proxy_lr,
            proxy_hidden=_proxy_hid,
            proxy_ensembles=_proxy_ens,
            force_proxy=bool(args.force_proxy),
        )

    if bool(args.proxy_only):
        for ckpt in sorted(out_dir.glob("ckpt_*.pt")):
            parsed = _parse_ckpt_variant(ckpt)
            if parsed is None:
                continue
            v, tail = parsed
            if tail == "__main__":
                data_p = merged_st if v in ("st_duo", "st_text") else mt_pkl
                _maybe_proxy(variant=v, data_p=data_p, ckpt=ckpt, ctx_len=ctx_main)
        for ckpt in sorted(finetune_out.glob("ckpt_*.pt")):
            parsed = _parse_ckpt_variant(ckpt)
            if parsed is None or parsed[1] == "__main__":
                continue
            v, tail = parsed
            fs_src = (
                _find_fewshot_pkl_for_tail(pkl_dir_ft, str(tail))
                if tail is not None
                else _find_legacy_fewshot_pkl(pkl_dir_ft)
            )
            if fs_src is None:
                print(f"[warn] proxy_only: no fewshot pkl for {ckpt.name}")
                continue
            fs_mt = _fewshot_proxy_multitask(
                fs_src, num_tasks=K, proxy_task_idx=int(args.zs_task_idx)
            )
            data_p = fs_src if v in ("st_duo", "st_text") else fs_mt
            _maybe_proxy(variant=v, data_p=data_p, ckpt=ckpt, ctx_len=ctx_fs)
        print("[done] proxy_only pass")
        return

    if not bool(args.skip_main):
        for v in variants:
            if v in ("st_duo", "st_text"):
                data_p = merged_st
            else:
                data_p = mt_pkl
            steps_base = int(train_steps)
            out_ckpt = out_dir / f"ckpt_{v}.pt"
            cur = _read_ckpt_step(out_ckpt)
            if tgt > 0:
                goal = int(tgt)
                need = max(0, int(goal) - int(cur or 0))
                if need <= 0:
                    print(f"[skip] {v} main: step>={goal} ({out_ckpt})")
                    continue
                load_p: Path | None = out_ckpt if cur is not None else None
                steps_run = int(need)
            else:
                if cur is not None and not bool(args.force_train):
                    print(f"[skip] {v} main: ckpt exists ({out_ckpt}); use --force_train or --train_step_target")
                    continue
                load_p = None
                steps_run = int(steps_base)
            _train_one(
                variant=v,
                train_pkl=data_p,
                task_embeds=task_embeds,
                num_tasks=K,
                horizon=horizon,
                context_length=ctx_main,
                train_steps=int(steps_run),
                batch_size=int(args.batch_size),
                lr=float(args.lr),
                grad_accum=int(args.grad_accum),
                device=str(args.device),
                out_ckpt=out_ckpt,
                load_ckpt=load_p,
                run_tag=f"{v}_main",
                log_freq=int(args.log_freq),
                save_freq=int(args.save_freq),
                wandb_project=_wb_proj,
                wandb_group=_wb_grp,
                wandb_run_name=f"train_{v}_main__{pkl_label}",
            )
            _maybe_proxy(
                variant=v,
                data_p=data_p,
                ckpt=out_ckpt,
                ctx_len=ctx_main,
            )
    else:
        print("[skip] main quartet (--skip_main)")
        if _use_proxy:
            for v in variants:
                out_ckpt = out_dir / f"ckpt_{v}.pt"
                if not out_ckpt.is_file():
                    continue
                data_p = merged_st if v in ("st_duo", "st_text") else mt_pkl
                _maybe_proxy(variant=v, data_p=data_p, ckpt=out_ckpt, ctx_len=ctx_main)

    if bool(args.skip_finetune):
        print("[fs] skip_finetune set; done.")
        return

    for v in variants:
        if not (out_dir / f"ckpt_{v}.pt").is_file():
            raise SystemExit(f"missing main ckpt for finetune: {out_dir}/ckpt_{v}.pt")

    tails_found = [t for t in _FS_TAIL_TAGS if _find_fewshot_pkl_for_tail(pkl_dir_ft, t)]
    _ft_filter = [x.strip() for x in str(args.fs_tail_tags).split(",") if x.strip()]
    if _ft_filter:
        tails_found = [t for t in _ft_filter if t in tails_found]
        if len(tails_found) != len(_ft_filter):
            missing = sorted(set(_ft_filter) - set(tails_found))
            raise SystemExit(f"fs_tail_tags not found under {pkl_dir_ft}: {missing}")
    if tails_found:
        tail_iter: tuple[str | None, ...] = tuple(tails_found)
    else:
        legacy = _find_legacy_fewshot_pkl(pkl_dir_ft)
        if legacy is None:
            print(f"[warn] no fewshot pkl under {pkl_dir_ft}; skip finetune.")
            return
        tail_iter = (None,)

    for tail_tag in tail_iter:
        fs_src = (
            _find_fewshot_pkl_for_tail(pkl_dir_ft, str(tail_tag))
            if tail_tag is not None
            else _find_legacy_fewshot_pkl(pkl_dir_ft)
        )
        if fs_src is None:
            continue
        fs_mt = _fewshot_proxy_multitask(fs_src, num_tasks=K, proxy_task_idx=int(args.zs_task_idx))
        fs_st = fs_src
        fs_suffix = f"_fs_{tail_tag}" if tail_tag else "_fs"
        print(f"[fs] finetune tail={tail_tag or 'legacy'} pkl={fs_src.name} (fewshot_dir={pkl_dir_ft.name})")

        for v in variants:
            base = out_dir / f"ckpt_{v}.pt"
            out_fs = finetune_out / f"ckpt_{v}{fs_suffix}.pt"
            if v in ("st_duo", "st_text"):
                data_p = fs_st
            else:
                data_p = fs_mt
            add_fs = int(args.finetune_add_steps)
            cur_fs = _read_ckpt_step(out_fs)
            _preserve = {x.strip() for x in str(args.finetune_preserve_variants).split(",") if x.strip()}
            _force_v = bool(args.force_finetune) and v not in _preserve
            if cur_fs is not None and add_fs <= 0 and not _force_v:
                print(f"[skip] {v}{fs_suffix}: fs ckpt exists ({out_fs})")
                continue
            load_fs: Path | None
            steps_fs: int
            if add_fs > 0:
                load_fs = out_fs if cur_fs is not None else base
                steps_fs = int(add_fs)
            else:
                load_fs = base
                steps_fs = int(finetune_steps)
            _wandb_fs_name = f"train_{v}{fs_suffix}__{pkl_label}__ftdata_{pkl_label_ft}"
            _train_one(
                variant=v,
                train_pkl=data_p,
                task_embeds=task_embeds,
                num_tasks=K,
                horizon=horizon,
                context_length=ctx_fs,
                train_steps=int(steps_fs),
                batch_size=int(args.batch_size),
                lr=float(finetune_lr),
                grad_accum=int(args.grad_accum),
                device=str(args.device),
                out_ckpt=out_fs,
                load_ckpt=load_fs,
                run_tag=f"{v}{fs_suffix}",
                log_freq=int(args.log_freq),
                save_freq=int(args.save_freq),
                wandb_project=_wb_proj,
                wandb_group=_wb_grp,
                wandb_run_name=_wandb_fs_name,
            )
            _maybe_proxy(
                variant=v,
                data_p=data_p,
                ckpt=out_fs,
                ctx_len=ctx_fs,
            )


if __name__ == "__main__":
    main()

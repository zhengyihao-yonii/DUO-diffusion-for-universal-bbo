# -*- coding: utf-8 -*-
"""
Load DUO-style checkpoints on synthetic PKLs and record denoise-step latent traces.

English doc: Mirrors ``comparisonExperiment/experiment1/duo_train_and_sample.py`` model
construction; extend ``model_overrides`` when your checkpoint used text/task conditioning.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from QualityExperiment.quality_text_condition_cfg import CONDITION_GUIDANCE_W_TEXT as _Q_W_TEXT
from QualityExperiment.quality_trace_arch import (
    ACTION_WEIGHT,
    CLIP_DENOISED,
    CONDITION_GUIDANCE_W,
    CONDITION_GUIDANCE_W_TASK,
    LOSS_TYPE,
    N_SAMPLE_TIMESTEPS,
    N_TIMESTEPS,
    PREDICT_EPSILON,
    UNET_CONDITION_DROPOUT,
    UNET_DIM,
    UNET_DIM_MULTS,
)
from diffuser.datasets.sequence import PointRegretDataset
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.models.temporal import TemporalUnet


def _dim_mults_tuple(v: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(v, (list, tuple)) and v:
        return tuple(int(x) for x in v)
    return tuple(int(x) for x in default)


@dataclass(frozen=True)
class TraceResult:
    """Denoising latent trajectory: ``z_steps[s, b, :]`` in R^{d_z}."""

    z_steps: np.ndarray
    timestep_indices: list[int]
    raw_x_last: np.ndarray  # final step, flattened [B*H, d_x] unnormalized
    proxy_best_idx: int = 0
    proxy_best_f: float = float("nan")
    legacy_min_f: float = float("nan")
    proxy_scores: np.ndarray | None = None


def _dataset_from_pkl(
    train_pkl: Path,
    *,
    horizon: int,
    context_length: int,
    task_text_embeds: np.ndarray | None,
    include_task_idx: bool,
    task_idx: int,
) -> PointRegretDataset:
    # Chinese comment: PointRegretDataset 不接受 task_idx；多任务 7 元组需用 task_name 对齐 tasks_list。
    with open(train_pkl, "rb") as f:
        data_obj = pickle.load(f)
    n = len(data_obj) if isinstance(data_obj, (list, tuple)) else 0
    task_name: str | None = None
    if n == 7 and include_task_idx:
        task_name = f"D_train_{int(task_idx) + 1}"

    return PointRegretDataset(
        horizon=int(horizon),
        data_path=str(train_pkl),
        context_length=int(context_length),
        regret=False,
        include_returns=False,
        task_name=task_name,
        task_text_embeds=task_text_embeds,
        include_task_idx=bool(include_task_idx),
    )


def build_diffusion(
    dataset: PointRegretDataset,
    *,
    horizon: int,
    dim: int | None = None,
    dim_mults: tuple[int, ...] | None = None,
    condition_dropout: float | None = None,
    n_timesteps: int | None = None,
    n_sample_timesteps: int | None = None,
    task_condition: bool = False,
    num_tasks: int = 1,
    text_condition: bool = False,
    text_embed_input_dim: int = 384,
    loss_type: str | None = None,
    clip_denoised: bool | None = None,
    action_weight: float | None = None,
    condition_guidance_w: float | None = None,
    condition_guidance_w_task: float | None = None,
    condition_guidance_w_text: float | None = None,
) -> GaussianDiffusion:
    """English doc: Defaults match ``config.exp1_diffusion_aligned`` via ``quality_trace_arch`` + text CFG from ``quality_text_condition_cfg``."""
    _dim = int(UNET_DIM if dim is None else dim)
    if dim_mults is None:
        _mults = UNET_DIM_MULTS
    else:
        _mults = _dim_mults_tuple(dim_mults, UNET_DIM_MULTS)
    _cd = float(UNET_CONDITION_DROPOUT if condition_dropout is None else condition_dropout)
    _nt = int(N_TIMESTEPS if n_timesteps is None else n_timesteps)
    _ns = int(N_SAMPLE_TIMESTEPS if n_sample_timesteps is None else n_sample_timesteps)
    _lt = str(LOSS_TYPE if loss_type is None else loss_type)
    _clip = bool(CLIP_DENOISED if clip_denoised is None else clip_denoised)
    _aw = float(ACTION_WEIGHT if action_weight is None else action_weight)
    _cg = float(CONDITION_GUIDANCE_W if condition_guidance_w is None else condition_guidance_w)
    _cgt = float(
        CONDITION_GUIDANCE_W_TASK if condition_guidance_w_task is None else condition_guidance_w_task
    )
    _cgx = float(
        _Q_W_TEXT if condition_guidance_w_text is None else condition_guidance_w_text
    )
    transition_dim = int(dataset.observation_dim + dataset.action_dim)
    model = TemporalUnet(
        horizon=int(horizon),
        transition_dim=transition_dim,
        cond_dim=0,
        dim=_dim,
        dim_mults=_mults,
        returns_condition=False,
        task_condition=bool(task_condition),
        num_tasks=int(num_tasks),
        condition_dropout=_cd,
        text_condition=bool(text_condition),
        text_embed_input_dim=int(text_embed_input_dim),
    )
    diffusion = GaussianDiffusion(
        model=model,
        horizon=int(horizon),
        observation_dim=int(dataset.observation_dim),
        action_dim=int(dataset.action_dim),
        n_timesteps=_nt,
        n_sample_timesteps=_ns,
        loss_type=_lt,
        clip_denoised=_clip,
        predict_epsilon=bool(PREDICT_EPSILON),
        action_weight=_aw,
        returns_condition=False,
        condition_guidance_w=_cg,
        condition_guidance_w_task=_cgt,
        condition_guidance_w_text=_cgx,
    )
    return diffusion


def load_ema_weights(diffusion: GaussianDiffusion, ckpt_path: Path, *, device: torch.device) -> None:
    """
    Load full ``GaussianDiffusion`` weights from ``duo_train_and_sample`` checkpoints.

    English doc: Those checkpoints store ``{\"ema\": diffusion.state_dict()}`` where ``ema``
    is the whole diffusion module (not only the inner UNet).
    """
    payload = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(payload, dict):
        state = payload.get("ema")
        if state is None:
            state = payload.get("model")
        if state is None:
            raise KeyError(f"checkpoint {ckpt_path} has no 'ema' or 'model'")
        diffusion.load_state_dict(state, strict=False)
    else:
        diffusion.load_state_dict(payload, strict=False)
    diffusion.to(device)


def _unnormalize_query_x(
    x: torch.Tensor,
    dataset: PointRegretDataset,
    *,
    obs_dim: int,
) -> torch.Tensor:
    """x: [B,H,transition] — use last horizon index as query design (exp1 convention)."""
    # Chinese comment: SafeLimitsNormalizer 的 mins/maxs 在 CPU；采样张量在 CUDA 时需对齐设备。
    x_slice = x[:, -1, :obs_dim].cpu()
    return dataset.normalizer.unnormalize(x_slice)


def sample_latent_trace(
    *,
    train_pkl: Path,
    ckpt_path: Path,
    horizon: int,
    context_length: int,
    sample_batch: int,
    device: str,
    project_z: Callable[[np.ndarray], np.ndarray],
    step_callback_stride: int = 2,
    task_text_embeds: np.ndarray | None = None,
    include_task_idx: bool = False,
    task_idx: int = 0,
    model_overrides: dict[str, Any] | None = None,
    text_embed_override: np.ndarray | None = None,
    prefix_seed: int = 0,
    proxy_ckpt: Path | None = None,
    objective: Any | None = None,
) -> TraceResult:
    """
    Run conditional_sample with optional prefix conditioning (matches train ctx_len) and record z.

    English doc: When ``context_length > 0``, prefixes are taken from random dataset windows
    (same PKL as training). ``project_z`` maps unnormalized x rows [N,d_x] -> [N,d_z].
    If ``text_embed_override`` is set and the model uses ``text_condition``, it replaces the
    row ``task_text_embeds[task_idx]`` (held-out / shift evaluation).
    """
    mo = dict(model_overrides or {})
    ds = _dataset_from_pkl(
        train_pkl,
        horizon=horizon,
        context_length=int(context_length),
        task_text_embeds=task_text_embeds,
        include_task_idx=include_task_idx,
        task_idx=task_idx,
    )
    _dm = mo.get("dim_mults", None)
    diffusion = build_diffusion(
        ds,
        horizon=horizon,
        task_condition=bool(mo.get("task_condition", False)),
        num_tasks=int(mo.get("num_tasks", 1)),
        text_condition=bool(mo.get("text_condition", False)),
        text_embed_input_dim=int(mo.get("text_embed_input_dim", 384)),
        dim=mo.get("dim") if mo.get("dim") is not None else None,
        dim_mults=_dim_mults_tuple(_dm, UNET_DIM_MULTS) if _dm is not None else None,
        condition_dropout=mo.get("condition_dropout"),
        n_timesteps=mo.get("n_timesteps") if mo.get("n_timesteps") is not None else None,
        n_sample_timesteps=mo.get("n_sample_timesteps")
        if mo.get("n_sample_timesteps") is not None
        else None,
        loss_type=mo.get("loss_type"),
        clip_denoised=mo.get("clip_denoised"),
        action_weight=mo.get("action_weight"),
        condition_guidance_w=mo.get("condition_guidance_w"),
        condition_guidance_w_task=mo.get("condition_guidance_w_task"),
        condition_guidance_w_text=mo.get("condition_guidance_w_text"),
    )
    dev = torch.device(device)
    load_ema_weights(diffusion, Path(ckpt_path), device=dev)
    diffusion.eval()

    obs_dim = int(ds.observation_dim)
    z_acc: list[np.ndarray] = []
    t_idx_acc: list[int] = []

    def _cb(
        timestep_index: int,
        step_ordinal: int,
        total_steps: int,
        x: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            u = _unnormalize_query_x(x, ds, obs_dim=obs_dim).detach().cpu().numpy()
        z = project_z(u)
        z_acc.append(z.astype(np.float32))
        t_idx_acc.append(int(timestep_index))

    cctx = int(max(0, min(int(context_length), int(horizon))))
    cond: dict[str, Any] = {}
    if cctx <= 0:
        cond["ctx_len"] = torch.zeros((int(sample_batch),), dtype=torch.long, device=dev)
    else:
        rng = np.random.default_rng(int(prefix_seed))
        n_idx = len(ds)
        pick = rng.integers(0, n_idx, size=int(sample_batch), endpoint=False)
        rows: list[torch.Tensor] = []
        for ii in pick:
            item = ds[int(ii)]
            rows.append(item.trajectories)
        pre_stack = torch.stack(rows, dim=0).to(dev)
        cond["ctx_len"] = torch.full(
            (int(sample_batch),), cctx, dtype=torch.long, device=dev
        )
        for t in range(cctx):
            cond[int(t)] = pre_stack[:, t, :].to(dev)
    if include_task_idx:
        cond["task_idx"] = torch.full(
            (int(sample_batch),), int(task_idx), dtype=torch.long, device=dev
        )
    if mo.get("text_condition"):
        if text_embed_override is not None:
            te = torch.from_numpy(
                np.asarray(text_embed_override, dtype=np.float32).reshape(-1)
            ).to(dev)
        elif task_text_embeds is not None:
            te = torch.from_numpy(
                np.asarray(task_text_embeds[int(task_idx)], dtype=np.float32)
            ).to(dev)
        else:
            te = None
        if te is not None:
            cond["text_embed"] = te.unsqueeze(0).expand(int(sample_batch), -1)

    with torch.no_grad():
        sample_out = diffusion.conditional_sample(
            cond,
            horizon=int(horizon),
            verbose=False,
            step_callback=_cb,
            step_callback_stride=int(step_callback_stride),
        )
    sample = sample_out[0] if isinstance(sample_out, tuple) else sample_out
    x_last = _unnormalize_query_x(sample, ds, obs_dim=obs_dim).detach().cpu().numpy()
    z_steps = np.stack(z_acc, axis=0)
    z_fin = z_steps[-1].astype(np.float64)
    proxy_best_idx = 0
    proxy_best_f = float("nan")
    legacy_min_f = float("nan")
    proxy_scores_out: np.ndarray | None = None
    if objective is not None:
        zt = torch.from_numpy(z_fin.astype(np.float32))
        of = objective.eval(zt).detach().cpu().numpy().reshape(-1)
        legacy_min_f = float(np.min(of))
        proxy_best_idx = int(np.argmin(of))
        proxy_best_f = float(of[proxy_best_idx])
    if proxy_ckpt is not None and Path(proxy_ckpt).is_file():
        from QualityExperiment.quality_proxy import (
            load_proxy,
            score_x_rows,
            select_proxy_best_oracle_f,
        )

        proxy, _ = load_proxy(Path(proxy_ckpt), device=str(device))
        ps = score_x_rows(dataset=ds, proxy=proxy, x_rows=x_last, device=str(device))
        proxy_scores_out = ps
        if objective is not None:
            zt = torch.from_numpy(z_fin.astype(np.float32))
            of = objective.eval(zt).detach().cpu().numpy().reshape(-1)
            proxy_best_idx, proxy_best_f, legacy_min_f = select_proxy_best_oracle_f(
                proxy_scores=ps,
                oracle_f=of,
            )
    return TraceResult(
        z_steps=z_steps,
        timestep_indices=t_idx_acc,
        raw_x_last=x_last,
        proxy_best_idx=int(proxy_best_idx),
        proxy_best_f=float(proxy_best_f),
        legacy_min_f=float(legacy_min_f),
        proxy_scores=proxy_scores_out,
    )


def load_ab_from_exp1_meta(meta_json: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``A,b`` from ``exp1_<task>.meta.json`` written by ``run_exp1.py``."""
    import json

    raw = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    a = np.asarray(raw["A"], dtype=np.float64)
    b = np.asarray(raw["b"], dtype=np.float64)
    return a, b

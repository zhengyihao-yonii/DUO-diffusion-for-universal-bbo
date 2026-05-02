# -*- coding: utf-8 -*-
"""
Load DUO-style checkpoints on synthetic PKLs and record denoise-step latent traces.

English doc: Mirrors ``comparisonExperiment/experiment1/duo_train_and_sample.py`` model
construction; extend ``model_overrides`` when your checkpoint used text/task conditioning.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from diffuser.datasets.sequence import PointRegretDataset
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.models.temporal import TemporalUnet


@dataclass(frozen=True)
class TraceResult:
    """Denoising latent trajectory: ``z_steps[s, b, :]`` in R^{d_z}."""

    z_steps: np.ndarray
    timestep_indices: list[int]
    raw_x_last: np.ndarray  # final step, flattened [B*H, d_x] unnormalized


def _dataset_from_pkl(
    train_pkl: Path,
    *,
    horizon: int,
    task_text_embeds: np.ndarray | None,
    include_task_idx: bool,
    task_idx: int,
) -> PointRegretDataset:
    return PointRegretDataset(
        horizon=int(horizon),
        data_path=str(train_pkl),
        context_length=0,
        regret=False,
        include_returns=False,
        task_name=None,
        task_text_embeds=task_text_embeds,
        include_task_idx=bool(include_task_idx),
        task_idx=int(task_idx),
    )


def build_diffusion(
    dataset: PointRegretDataset,
    *,
    horizon: int,
    dim: int = 128,
    dim_mults: tuple[int, ...] = (1, 2, 4),
    n_timesteps: int = 1000,
    n_sample_timesteps: int = 200,
    task_condition: bool = False,
    num_tasks: int = 1,
    text_condition: bool = False,
    text_embed_input_dim: int = 384,
) -> GaussianDiffusion:
    transition_dim = int(dataset.observation_dim + dataset.action_dim)
    model = TemporalUnet(
        horizon=int(horizon),
        transition_dim=transition_dim,
        cond_dim=0,
        dim=int(dim),
        dim_mults=dim_mults,
        returns_condition=False,
        task_condition=bool(task_condition),
        num_tasks=int(num_tasks),
        condition_dropout=0.0,
        text_condition=bool(text_condition),
        text_embed_input_dim=int(text_embed_input_dim),
    )
    diffusion = GaussianDiffusion(
        model=model,
        horizon=int(horizon),
        observation_dim=int(dataset.observation_dim),
        action_dim=int(dataset.action_dim),
        n_timesteps=int(n_timesteps),
        n_sample_timesteps=int(n_sample_timesteps),
        loss_type="l1",
        clip_denoised=False,
        predict_epsilon=True,
        returns_condition=False,
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
    x_slice = x[:, -1, :obs_dim]
    return dataset.normalizer.unnormalize(x_slice)


def sample_latent_trace(
    *,
    train_pkl: Path,
    ckpt_path: Path,
    horizon: int,
    sample_batch: int,
    device: str,
    project_z: Callable[[np.ndarray], np.ndarray],
    step_callback_stride: int = 2,
    task_text_embeds: np.ndarray | None = None,
    include_task_idx: bool = False,
    task_idx: int = 0,
    model_overrides: dict[str, Any] | None = None,
    text_embed_override: np.ndarray | None = None,
) -> TraceResult:
    """
    Run conditional_sample with ctx_len=0 (same as exp1 duo_train_and_sample) and record z.

    English doc: ``project_z`` maps unnormalized x rows [N,d_x] -> [N,d_z] (see latent_geometry).
    If ``text_embed_override`` is set and the model uses ``text_condition``, it replaces the
    row ``task_text_embeds[task_idx]`` (held-out / shift evaluation).
    """
    mo = dict(model_overrides or {})
    ds = _dataset_from_pkl(
        train_pkl,
        horizon=horizon,
        task_text_embeds=task_text_embeds,
        include_task_idx=include_task_idx,
        task_idx=task_idx,
    )
    diffusion = build_diffusion(
        ds,
        horizon=horizon,
        task_condition=bool(mo.get("task_condition", False)),
        num_tasks=int(mo.get("num_tasks", 1)),
        text_condition=bool(mo.get("text_condition", False)),
        text_embed_input_dim=int(mo.get("text_embed_input_dim", 384)),
        n_sample_timesteps=int(mo.get("n_sample_timesteps", 200)),
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

    cond: dict[str, Any] = {
        "ctx_len": torch.zeros((int(sample_batch),), dtype=torch.long, device=dev)
    }
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
    return TraceResult(
        z_steps=z_steps,
        timestep_indices=t_idx_acc,
        raw_x_last=x_last,
    )


def load_ab_from_exp1_meta(meta_json: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``A,b`` from ``exp1_<task>.meta.json`` written by ``run_exp1.py``."""
    import json

    raw = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    a = np.asarray(raw["A"], dtype=np.float64)
    b = np.asarray(raw["b"], dtype=np.float64)
    return a, b

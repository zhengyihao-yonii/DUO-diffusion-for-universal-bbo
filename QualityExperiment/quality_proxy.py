# -*- coding: utf-8 -*-
"""Simple MLP proxy for Quality exp (x -> y), aligned with main DUO / comparison1."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data as data

from diffuser.models.temporal import Proxy
from diffuser.utils.training import Trainer


class _ProxyPointDataset(data.Dataset):
    """English doc: flattened (x, y) from PointRegretDataset for proxy training."""

    def __init__(self, points: torch.Tensor, values: torch.Tensor, normalizer: Any) -> None:
        self._points = points
        self._values = values
        self._norm = normalizer
        self.data_x = points
        self.data_y = values

    def __len__(self) -> int:
        return int(self._points.shape[0])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        xi = self._norm.normalize(self._points[i])
        yi = self._values[i]
        if yi.ndim == 0:
            yi = yi.view(1)
        return xi, yi


def proxy_path_for_ckpt(ckpt: Path) -> Path:
    """``ckpt_mt_text.pt`` -> ``proxy_mt_text.pt``; ``ckpt_mt_text_fs_tail10p.pt`` -> same pattern."""
    name = ckpt.name
    if name.startswith("ckpt_"):
        return ckpt.with_name("proxy_" + name[5:])
    return ckpt.with_name("proxy_" + name)


def proxy_path_for_train_domain(ckpt: Path, task_id: str) -> Path:
    """
    Per-task proxy for D_train eval (aligned with main-experiment per-task proxy pools).

    ``ckpt_st_duo.pt`` + ``D_train_2`` -> ``proxy_st_duo__D_train_2.pt``.
    """
    tid = str(task_id).strip()
    if not tid:
        return proxy_path_for_ckpt(ckpt)
    base = proxy_path_for_ckpt(ckpt).stem
    return ckpt.with_name(f"{base}__{tid}.pt")


def resolve_proxy_path_for_eval(
    ckpt: Path,
    *,
    phase_label: str,
    task_id: str = "",
) -> Path:
    """English doc: train_domain uses per-task proxy; shift/fs use ckpt-colocated proxy."""
    pl = str(phase_label).strip().lower()
    tid = str(task_id).strip()
    if pl == "train_domain" and tid:
        per_task = proxy_path_for_train_domain(ckpt, tid)
        if per_task.is_file():
            return per_task
    return proxy_path_for_ckpt(ckpt)


def build_proxy_dataset_from_points(
    dataset: Any,
) -> _ProxyPointDataset:
    """English doc: use ``dataset.points`` / ``values`` (same as comparison1)."""
    x_flat = dataset.points.reshape(-1, int(dataset.observation_dim))
    y_flat = dataset.values.reshape(-1, 1)
    return _ProxyPointDataset(x_flat, y_flat, dataset.normalizer)


def train_and_save_proxy(
    *,
    train_pkl: Path,
    dataset: Any,
    device: str,
    save_path: Path,
    n_steps: int,
    proxy_lr: float,
    hidden_dim: int,
    n_ensembles: int,
    batch_size: int = 256,
) -> Path:
    """Train ensemble proxy on PKL point pool and save ``state_dict``."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    proxy_ds = build_proxy_dataset_from_points(dataset)
    obs_dim = int(dataset.observation_dim)
    proxy_model = Proxy(
        input_dim=obs_dim,
        hidden_dim=int(hidden_dim),
        output_dim=1,
        n_ensembles=int(n_ensembles),
    )
    dev = torch.device(str(device))
    proxy_model = proxy_model.to(dev)
    trainer = Trainer(
        diffusion_model=proxy_model,
        proxy_model=proxy_model,
        dataset=dataset,
        proxy_dataset=proxy_ds,
        renderer=None,
        ema_decay=0.995,
        train_batch_size=int(batch_size),
        train_lr=float(proxy_lr),
        proxy_train_lr=float(proxy_lr),
        gradient_accumulate_every=1,
        log_freq=max(100, int(n_steps) // 10),
        sample_freq=0,
        save_freq=int(n_steps) + 1,
        proxy_save_freq=int(n_steps) + 1,
        train_device=str(device),
        save_checkpoints=False,
    )
    setattr(trainer, "_total_proxy_steps", int(n_steps))
    trainer.train_proxy(int(n_steps))
    torch.save(
        {
            "state_dict": proxy_model.state_dict(),
            "observation_dim": obs_dim,
            "train_pkl": str(train_pkl),
        },
        str(save_path),
    )
    print(f"[proxy] saved {save_path} (steps={n_steps}, pkl={train_pkl.name})")
    return save_path


def load_proxy(
    path: Path,
    *,
    device: str,
) -> tuple[Proxy, int]:
    """Load proxy checkpoint; returns (model, observation_dim)."""
    ck = torch.load(str(path), map_location="cpu")
    obs_dim = int(ck["observation_dim"])
    hidden = 256
    n_ens = 5
    proxy = Proxy(
        input_dim=obs_dim,
        hidden_dim=hidden,
        output_dim=1,
        n_ensembles=n_ens,
    )
    proxy.load_state_dict(ck["state_dict"])
    proxy = proxy.to(torch.device(str(device)))
    proxy.eval()
    return proxy, obs_dim


def score_x_rows(
    *,
    dataset: Any,
    proxy: Proxy,
    x_rows: np.ndarray,
    device: str,
) -> np.ndarray:
    """Score unnormalized observation rows; higher proxy score = better (matches y in PKL)."""
    dev = torch.device(str(device))
    x = np.asarray(x_rows, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    with torch.no_grad():
        xt = torch.tensor(x, device="cpu")
        xn = dataset.normalizer.normalize(xt).to(dev)
        yhat = proxy(xn).detach().cpu().numpy().reshape(-1)
    return yhat.astype(np.float64)


def select_proxy_best_oracle_f(
    *,
    proxy_scores: np.ndarray,
    oracle_f: np.ndarray,
) -> tuple[int, float, float]:
    """
    Pick trajectory with highest proxy score; return (idx, oracle_f[idx], legacy_min_f).

    English doc: ``legacy_min_f`` = min oracle over all rows (old last-step min metric).
    """
    ps = np.asarray(proxy_scores, dtype=np.float64).reshape(-1)
    of = np.asarray(oracle_f, dtype=np.float64).reshape(-1)
    if ps.size == 0:
        return 0, float("nan"), float("nan")
    idx = int(np.argmax(ps))
    return idx, float(of[idx]), float(np.min(of))

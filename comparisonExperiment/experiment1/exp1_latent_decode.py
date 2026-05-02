# -*- coding: utf-8 -*-
"""Decode exp1 latent design rows to native x using shared VAE (sidecar / merge manifest)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from comparisonExperiment.experiment1.exp1_vae import decode_latent_to_native_head, load_vae


def _sidecar_path(train_pkl: Path) -> Path:
    return train_pkl.parent / f"{train_pkl.name}.exp1_sidecar.json"


def _manifest_path(train_pkl: Path) -> Path:
    return train_pkl.parent / f"{train_pkl.name}.exp1_merge_manifest.json"


def decode_latent_matrix_to_native(
    train_pkl: Path,
    x_lat: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray | None:
    """Decode [N,latent] -> [N,d_x_native] using ``vae_shared.pt`` sidecar; merged PKL returns None."""
    tp = Path(train_pkl)
    if _manifest_path(tp).is_file():
        return None
    sc = _sidecar_path(tp)
    if not sc.is_file():
        return None
    raw = json.loads(sc.read_text(encoding="utf-8"))
    d_nat = int(raw["d_x_native"])
    vae_name = str(raw["vae_filename"])
    vae_path = tp.parent / vae_name
    if not vae_path.is_file():
        return None
    vae = load_vae(vae_path, device=device)
    xt = torch.from_numpy(np.asarray(x_lat, dtype=np.float32))
    out = decode_latent_to_native_head(vae, xt, d_x_native=d_nat, device=device)
    return out.numpy().astype(np.float32)

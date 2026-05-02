# -*- coding: utf-8 -*-
"""Per-task VAE: native design x -> shared latent (default 16) for heterogeneous exp1."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from comparisonExperiment.experiment1.data_formats import DuoTrajectoryPkl
from diffuser.models.vae import VAE


def build_task_vae(*, input_dim: int, latent_dim: int = 16) -> VAE:
    """English doc: Same VAE class as main DUO; small Transformer over one design token."""
    return VAE(
        input_dim=int(input_dim),
        latent_dim=int(latent_dim),
        d_model=256,
        nhead=4,
        num_layers=2,
        dropout=0.1,
    )


def fit_vae_on_points(
    x_flat: torch.Tensor,
    *,
    input_dim: int,
    latent_dim: int = 16,
    train_steps: int = 800,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device,
    kl_coef: float = 0.1,
) -> VAE:
    """Chinese comment: 在单任务点上快速训 VAE，用 mu 作为编码（轨迹构造时确定性）。"""
    vae = build_task_vae(input_dim=int(input_dim), latent_dim=int(latent_dim)).to(device)
    vae.train()
    opt = torch.optim.Adam(vae.parameters(), lr=float(lr))
    n = int(x_flat.shape[0])
    if n < 2:
        raise ValueError("need >=2 points for VAE training")
    for step in range(int(train_steps)):
        idx = torch.randint(0, n, (int(batch_size),), device=x_flat.device)
        xb = x_flat[idx].to(device)
        recon, mu, logvar, _z = vae(xb)
        recon_loss = F.mse_loss(recon, xb, reduction="sum") / float(xb.shape[0])
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        loss = recon_loss + float(kl_coef) * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    vae.eval()
    return vae


@torch.no_grad()
def native_traj_to_latent_traj(
    vae: VAE,
    obj: DuoTrajectoryPkl,
    *,
    device: torch.device,
) -> DuoTrajectoryPkl:
    """Encode each timestep design x -> latent mu; VAE input_dim must match last dim of points."""
    pts = obj.points
    nt, h, dx = pts.shape
    flat = pts.reshape(-1, dx).to(device)
    mu, _ = vae.encode(flat)
    lat = mu.reshape(nt, h, -1).cpu()
    return DuoTrajectoryPkl(
        points=lat,
        values=obj.values,
        pr=obj.pr,
        rtg=obj.rtg,
        timesteps=obj.timesteps,
    )


def pad_x_to_d_pad(x: torch.Tensor, *, d_pad: int) -> torch.Tensor:
    """x [..., d_nat] -> [..., d_pad] with zero padding on the right."""
    d_nat = int(x.shape[-1])
    if d_nat == int(d_pad):
        return x
    if d_nat > int(d_pad):
        return x[..., :d_pad]
    pad_shape = list(x.shape[:-1]) + [int(d_pad)]
    out = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
    out[..., :d_nat] = x
    return out


@torch.no_grad()
def native_traj_to_latent_padded(
    vae: VAE,
    obj: DuoTrajectoryPkl,
    *,
    d_pad: int,
    device: torch.device,
) -> DuoTrajectoryPkl:
    """English doc: Pad native trajectories to ``d_pad`` then encode with shared VAE (input_dim=d_pad)."""
    pts = pad_x_to_d_pad(obj.points, d_pad=int(d_pad))
    nt, h, dp = pts.shape
    flat = pts.reshape(-1, dp).to(device)
    mu, _ = vae.encode(flat)
    lat = mu.reshape(nt, h, -1).cpu()
    return DuoTrajectoryPkl(
        points=lat,
        values=obj.values,
        pr=obj.pr,
        rtg=obj.rtg,
        timesteps=obj.timesteps,
    )


@torch.no_grad()
def decode_latent_rows(
    vae: VAE,
    latent_x: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """latent_x: [*, latent_dim] -> padded x [*, vae.input_dim]."""
    z = latent_x.to(device)
    out = vae.decode(z)
    return out.cpu()


@torch.no_grad()
def decode_latent_to_native_head(
    vae: VAE,
    latent_x: torch.Tensor,
    *,
    d_x_native: int,
    device: torch.device,
) -> torch.Tensor:
    """Decode to padded design then take first ``d_x_native`` coordinates (oracle / UniSO native)."""
    full = decode_latent_rows(vae, latent_x, device=device)
    dn = min(int(d_x_native), int(full.shape[-1]))
    return full[..., :dn].contiguous()


def save_vae(path: Path, vae: VAE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vae": vae.state_dict(), "input_dim": int(vae.input_dim), "latent_dim": int(vae.latent_dim)}, path)


def load_vae(path: Path, *, device: torch.device) -> VAE:
    payload = torch.load(str(path), map_location="cpu")
    sd = payload["vae"]
    d_in = int(payload["input_dim"])
    d_lat = int(payload["latent_dim"])
    vae = build_task_vae(input_dim=d_in, latent_dim=d_lat)
    vae.load_state_dict(sd, strict=True)
    vae.to(device)
    vae.eval()
    return vae

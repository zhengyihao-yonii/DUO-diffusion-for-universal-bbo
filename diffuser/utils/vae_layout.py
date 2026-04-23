"""VAE latent layout: consistent filenames for trajectory PKLs and raw-train PKL paths."""

from __future__ import annotations

import re


def per_task_latent_train_filename(
    task: str,
    n_traj: int,
    horizon: int,
    k: int,
    eps: float,
    latent_dim: int,
) -> str:
    """例如 ``ant_1000x64_k20_eps0.05_vae_latent64_train.p``。"""
    return (
        f"{task}_{int(n_traj)}x{int(horizon)}_k{int(k)}_eps{float(eps)}"
        f"_vae_latent{int(latent_dim)}_train.p"
    )


def raw_train_pkl_path_from_latent_path(latent_train_path: str) -> str:
    """``..._vae_latent{d}_train.p`` → ``..._train.p``（用于单任务 VAE 自训时读原始观测轨迹）。"""
    return re.sub(r"_vae_latent\d+_train\.p$", "_train.p", latent_train_path)


def vae_state_pt_filename(latent_dim: int) -> str:
    """与 ``train_vae.py`` 一致：``vae_latent{d}.pt``。"""
    return f"vae_latent{int(latent_dim)}.pt"


def vae_train_dir_suffix(latent_dim: int) -> str:
    """
    ``trained_models/vae/<task>_frac..._dim{d}{suffix}/`` 目录后缀。

    - ``latent_dim == 32``：空串，与历史布局一致（同目录下放 ``vae_latent32.pt`` 等）。
    - 其它维度：``_latent{d}``，避免与 32 维共用目录导致 ``dataset_info.p`` / ``scaler_*.p`` / ``vae_info.p`` 互覆盖。
    """
    return "" if int(latent_dim) == 32 else f"_latent{int(latent_dim)}"


def generated_vae_info_filename(latent_dim: int) -> str:
    """``generated_datasets/.../`` 下 VAE 元数据文件名；32 维保持 ``vae_info.p``。"""
    if int(latent_dim) == 32:
        return "vae_info.p"
    return f"vae_info_latent{int(latent_dim)}.p"


def distance_matrix_cache_filename(latent_dim: int) -> str:
    """轨迹构图用的 pairwise 距离缓存；32 维保持 ``distance_vae.p``。"""
    if int(latent_dim) == 32:
        return "distance_vae.p"
    return f"distance_vae_latent{int(latent_dim)}.p"


def resolve_generated_vae_info_path(base_dir: str, latent_dim: int) -> str | None:
    """
    在 ``base_dir`` 下解析可读的 vae 元信息路径：优先 ``generated_vae_info_filename(d)``，
    ``latent_dim==32`` 时再尝试历史 ``vae_info.p``。
    """
    import os

    d = int(latent_dim)
    primary = os.path.join(base_dir, generated_vae_info_filename(d))
    if os.path.isfile(primary):
        return primary
    if d == 32:
        legacy = os.path.join(base_dir, "vae_info.p")
        if os.path.isfile(legacy):
            return legacy
    return None

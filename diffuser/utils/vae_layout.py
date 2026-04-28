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


def multitask_generated_dim_latent_suffix(fixed_dim: int, latent_dim: int) -> str:
    """
    多任务 ``generated_datasets`` 目录后缀，与 ``train_vae.py`` 中
    ``trained_models/vae/multi_*_dim{fixed}[_mt_<hex>]_latent{lat}`` 对齐；``_mt_*`` 与
    ``traj_params`` 的轨迹签名一致（有签名时存在）。便于把
    ``vae_info_latent*.p`` / mixed 与不同隐空间宽度区分。

    ``latent_dim == 32`` 时返回空串，保持历史 ``multi_<tok>_frac_sigma`` 路径。
    """
    if int(latent_dim) == 32:
        return ""
    return f"_dim{int(fixed_dim)}_latent{int(latent_dim)}"


def multitask_generated_candidate_rel_dirs(
    *,
    train_tasks_csv: str,
    frac: float,
    sigma: float,
    fixed_dim: int,
    latent_dim: int,
) -> list[str]:
    """
    相对 DUO 根目录的候选 ``generated_datasets/multi_*`` 路径（先带 ``_dim*_latent*``，后短路径）。
    用于加载 VAE / mixed：优先匹配用户已有的 ``..._dim128_latent64`` 目录。
    """
    from diffuser.utils.multitask_canon import canonical_train_tasks_csv, multitask_path_token

    tok = multitask_path_token(canonical_train_tasks_csv(train_tasks_csv))
    suf = multitask_generated_dim_latent_suffix(fixed_dim, latent_dim)
    out: list[str] = []
    if suf:
        out.append(
            f"generated_datasets/multi_{tok}_frac{frac}_sigma{sigma}{suf}"
        )
    out.append(f"generated_datasets/multi_{tok}_frac{frac}_sigma{sigma}")
    return out


def resolve_multitask_generated_root_for_vae(
    *,
    train_tasks_csv: str,
    frac: float,
    sigma: float,
    fixed_dim: int,
    latent_dim: int,
) -> str:
    """
    解析应包含 ``generated_vae_info_filename(latent_dim)`` 的目录。
    优先返回已存在元数据文件的目录；否则返回首选候选（便于报错信息指向一致路径）。
    """
    import os

    ld = int(latent_dim)
    rels = multitask_generated_candidate_rel_dirs(
        train_tasks_csv=train_tasks_csv,
        frac=frac,
        sigma=sigma,
        fixed_dim=fixed_dim,
        latent_dim=ld,
    )
    candidates = [
        os.path.normpath("./" + r) if not r.startswith("./") else r for r in rels
    ]
    for base in candidates:
        vip = resolve_generated_vae_info_path(base, ld)
        if vip is not None and os.path.isfile(vip):
            return base
    for base in candidates:
        if os.path.isdir(base):
            return base
    return candidates[0]


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


def resolve_vae_weights_path_for_eval(
    *,
    raw_path: str | None,
    vae_info_path: str,
    latent_dim: int,
    project_root: str,
    multitask_traj_signature: str | None = None,
) -> tuple[str | None, list[str]]:
    """
    Resolve path to ``vae_latent{d}.pt`` from metadata in ``vae_info`` pickle.

    ``construct_trajectories`` / historical runs may store a parent dir of the form
    ``..._dim{fd}_latent{d}``, while :func:`train_vae_main` saves under
    ``..._dim{fd}{mt_token}_latent{d}`` when ``multitask_traj_signature`` is set.
    Also accept project-root-relative paths and weights placed next to ``vae_info``.
    """
    import glob as _glob
    import os
    import re

    tried: list[str] = []
    ld = int(latent_dim)
    pt_name = vae_state_pt_filename(ld)
    root = os.path.abspath(project_root)

    def _attempt(p: str | None) -> str | None:
        if not p:
            return None
        np = os.path.normpath(p)
        if np not in tried:
            tried.append(np)
        return np if os.path.isfile(np) else None

    if raw_path:
        if os.path.isabs(raw_path):
            x = _attempt(raw_path)
            if x:
                return x, tried
        else:
            x = _attempt(os.path.join(root, raw_path))
            if x:
                return x, tried
            x = _attempt(raw_path)
            if x:
                return x, tried

    x = _attempt(os.path.join(os.path.dirname(os.path.abspath(vae_info_path)), pt_name))
    if x:
        return x, tried

    # Joint multitask VAE dir may include mt slug between _dim* and _latent* (see train_vae.py).
    if raw_path and multitask_traj_signature:
        from diffuser.utils.traj_params import multitask_vae_dir_token

        mt = multitask_vae_dir_token(multitask_traj_signature)
        ref = raw_path
        if not os.path.isabs(ref):
            ref = os.path.join(root, ref)
        pt_dir = os.path.dirname(os.path.normpath(ref))
        bn = os.path.basename(pt_dir)
        if mt not in bn and re.search(r"_dim\d+_latent\d+$", bn):
            alt_bn = re.sub(
                r"(_dim\d+)(_latent\d+)$",
                r"\1" + mt + r"\2",
                bn,
            )
            gv_root = os.path.dirname(pt_dir)
            alt_pt = os.path.join(gv_root, alt_bn, pt_name)
            x = _attempt(alt_pt)
            if x:
                return x, tried

    # Same multitask CSV prefix + any compatible VAE subdirectory (weights moved / renamed).
    if raw_path:
        ref = raw_path if os.path.isabs(raw_path) else os.path.join(root, raw_path)
        pt_dir = os.path.dirname(os.path.normpath(ref))
        bn = os.path.basename(pt_dir)
        seed = bn.split("_frac")[0] if "_frac" in bn else ""
        if seed:
            pat = os.path.join(root, "trained_models", "vae", "*", pt_name)
            for cand in sorted(_glob.glob(pat)):
                if seed in cand:
                    x = _attempt(cand)
                    if x:
                        return x, tried

    return None, tried

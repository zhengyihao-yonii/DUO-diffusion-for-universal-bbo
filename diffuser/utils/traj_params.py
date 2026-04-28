"""
多任务轨迹参数：统一标量 vs 每任务字典；生成与 data_path / mixed 文件名一致的签名。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

from diffuser.utils.multitask_canon import canonical_train_tasks_csv, multitask_path_token

# 与 construct_trajectories 中默认表一致（单任务缺省按任务名）
DEFAULT_NUM_TRAJECTORIES: dict[str, int] = {
    "tfbind8": 1000,
    "tfbind10": 1000,
    "superconductor": 4000,
    "ant": 4000,
    "dkitty": 4000,
    "gtopx2": 2000,
    "gtopx3": 2000,
    "gtopx4": 2000,
    "gtopx6": 2000,
}

DEFAULT_K: dict[str, int] = {
    "tfbind8": 50,
    "tfbind10": 50,
    "superconductor": 20,
    "ant": 20,
    "dkitty": 20,
    "gtopx2": 20,
    "gtopx3": 20,
    "gtopx4": 20,
    "gtopx6": 20,
}

DEFAULT_EPS: dict[str, float] = {
    "tfbind8": 0.05,
    "tfbind10": 0.05,
    "superconductor": 0.05,
    "ant": 0.05,
    "dkitty": 0.01,
    "gtopx2": 0.05,
    "gtopx3": 0.05,
    "gtopx4": 0.05,
    "gtopx6": 0.05,
}


def coerce_traj_param_dicts(
    tasks_list: list[str],
    n_traj: int | dict | None,
    k: int | dict | None,
    eps: float | dict | None,
) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    """将 CLI 标量 / 显式 dict 转为 per-task 字典。"""
    if isinstance(n_traj, dict):
        nd = {t: int(n_traj[t]) for t in tasks_list}
    elif n_traj is None:
        nd = {task: DEFAULT_NUM_TRAJECTORIES.get(task, 1000) for task in tasks_list}
    elif isinstance(n_traj, int):
        nd = {task: n_traj for task in tasks_list}
    else:
        raise TypeError("n_traj 须为 int、dict 或 None")

    if isinstance(k, dict):
        kd = {t: int(k[t]) for t in tasks_list}
    elif k is None:
        kd = {task: DEFAULT_K.get(task, 20) for task in tasks_list}
    elif isinstance(k, int):
        kd = {task: k for task in tasks_list}
    else:
        raise TypeError("k 须为 int、dict 或 None")

    if isinstance(eps, dict):
        ed = {t: float(eps[t]) for t in tasks_list}
    elif eps is None:
        ed = {task: DEFAULT_EPS.get(task, 0.05) for task in tasks_list}
    elif isinstance(eps, float) or isinstance(eps, int):
        ed = {task: float(eps) for task in tasks_list}
    else:
        raise TypeError("eps 须为 float、dict 或 None")

    return nd, kd, ed


def multitask_trajectory_signature(
    tasks: list[str],
    n_traj: dict[str, int],
    k: dict[str, int],
    eps: dict[str, float],
    horizon: int,
) -> str:
    """
    逻辑签名字符串；与 :func:`multitask_mixed_basename`、:func:`multitask_slug_id` 对应；
    各任务轨迹 pkl 在 ``generated_datasets/<task>_frac_sigma/``。
    全任务标量相同时退化为 ``{n}x{h}_k{k}_eps{eps}``。
    """
    ts = sorted(tasks)
    n0, k0, e0 = n_traj[ts[0]], k[ts[0]], eps[ts[0]]
    uniform = all(
        n_traj[t] == n0 and k[t] == k0 and math.isclose(eps[t], e0, rel_tol=0, abs_tol=1e-9)
        for t in ts
    )
    if uniform:
        return f"{n0}x{horizon}_k{k0}_eps{e0:g}"
    parts = []
    for t in ts:
        parts.append(f"{t}_n{n_traj[t]}_k{k[t]}_eps{eps[t]:g}")
    return "ptask__" + "__".join(parts)


def merge_traj_params_json(
    path: str,
    tasks: list[str],
    n_traj: dict[str, int],
    k: dict[str, int],
    eps: dict[str, float],
) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    """
    在已有 per-task 字典上合并 JSON（``construct_trajectories`` 先按 CLI 得到字典再 merge）。

    示例::

        {
          "gtopx2": {"n_traj": 2000, "k": 30, "eps": 0.05},
          "defaults": {"n_traj": 4000, "k": 20, "eps": 0.05}
        }
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("traj_params_json 根必须是对象")
    n_traj = dict(n_traj)
    k = dict(k)
    eps = dict(eps)
    defs = raw.get("defaults") or {}
    if isinstance(defs, dict):
        for t in tasks:
            if "n_traj" in defs:
                n_traj[t] = int(defs["n_traj"])
            if "k" in defs:
                k[t] = int(defs["k"])
            if "eps" in defs:
                eps[t] = float(defs["eps"])
    for key, val in raw.items():
        if key == "defaults" or key not in n_traj:
            continue
        if not isinstance(val, dict):
            continue
        if "n_traj" in val:
            n_traj[key] = int(val["n_traj"])
        if "k" in val:
            k[key] = int(val["k"])
        if "eps" in val:
            eps[key] = float(val["eps"])
    return n_traj, k, eps


def prepare_multitask_traj(
    tasks_list: list[str],
    n_traj: int | dict | None,
    k: int | dict | None,
    eps: float | dict | None,
    horizon: int,
    traj_params_json: str | None = None,
) -> tuple[dict[str, int], dict[str, int], dict[str, float], str]:
    """合并 JSON（可选）并返回签名（用于 data_path / mixed 文件名 / checkpoint）。"""
    n_d, k_d, e_d = coerce_traj_param_dicts(tasks_list, n_traj, k, eps)
    if traj_params_json:
        n_d, k_d, e_d = merge_traj_params_json(
            traj_params_json, tasks_list, n_d, k_d, e_d
        )
    sig = multitask_trajectory_signature(tasks_list, n_d, k_d, e_d, horizon)
    return n_d, k_d, e_d, sig


def multitask_slug_id(traj_signature: str) -> str:
    """
    短标识符，形式 ``mt_<16位hex>``，与完整 ``traj_signature``（如 ``ptask__...``）一一对应。
    用于 mixed 文件名、checkpoint 目录、RESULTS 超参段，避免路径过长（errno 36）。
    """
    h = hashlib.sha256(traj_signature.encode("utf-8")).hexdigest()[:16]
    return f"mt_{h}"


def multitask_mixed_basename(traj_signature: str, latent_dim: int = 32) -> str:
    """
    混合轨迹短文件名。

    - ``latent_dim == 32``：保持历史命名 ``mixed_mt_<hex>.p``。
    - 其它维度：``mixed_mt_<hex>_latent{d}.p``，避免与 32 维产物混用同一 mixed 文件。
    """
    core = multitask_slug_id(traj_signature)
    if int(latent_dim) == 32:
        return f"mixed_{core}.p"
    return f"mixed_{core}_latent{int(latent_dim)}.p"


def multitask_vae_dir_token(traj_signature: str) -> str:
    """
    多任务 VAE 保存目录在 ``..._dim{fd}`` 后插入的片段，与
    :func:`multitask_mixed_basename` 中 ``mt_<hex>`` 使用同一 :func:`multitask_slug_id`。
    当轨迹签名（n_traj/k/eps/horizon 等）变化时，会落到新目录，从而触发重新训练（若无权重）。
    """
    return f"_{multitask_slug_id(traj_signature)}"


def multitask_checkpoint_hyper_dir(
    sig: str,
    ret_infix: str,
    text_infix: str,
    mttextonly_infix: str,
) -> str:
    """
    ``trained_models/multi_<tasks>_frac_sigma/<本函数返回值>/seed*/`` 的中间目录名。

    使用 :func:`multitask_slug_id` + 条件后缀（与 train.py 中 returns/text/mttextonly 片段一致）。
    """
    return f"{multitask_slug_id(sig)}{ret_infix}{text_infix}{mttextonly_infix}"


def resolve_multitask_mixed_path(
    data_dir: str, sig: str | None, latent_dim: int = 32
) -> str:
    """
    优先短文件名 ``mixed_<slug_id>.p``（或带 ``_latent{d}`` 的变体）；其次旧版 ``mixed_<完整sig>.p``；
    再 ``mixed_trajectories_train.p``。

    当 ``latent_dim != 32`` 时 **不再** 回退到 ``mixed_mt_<hex>.p``（历史 32 维文件名），
    否则 evaluate 会静默读到 32 维轨迹却声称 latent64，导致与 checkpoint / proxy 维数错位。
    """
    import os

    ld = int(latent_dim)
    if sig:
        p_lat = os.path.join(data_dir, multitask_mixed_basename(sig, ld))
        if os.path.isfile(p_lat):
            return p_lat
        if ld != 32:
            raise FileNotFoundError(
                f"未找到 latent_dim={ld} 的多任务混合轨迹：{p_lat}。"
                f"（已禁用回退到 {multitask_mixed_basename(sig, 32)}，避免维数静默错位。）"
                f"请运行 construct_trajectories.py 并指定 --latent_dim {ld}，或检查 generated_datasets 目录。"
            )
        p_short = os.path.join(data_dir, multitask_mixed_basename(sig, 32))
        if os.path.isfile(p_short):
            return p_short
        p_long = os.path.join(data_dir, f"mixed_{sig}.p")
        if os.path.isfile(p_long):
            return p_long
    leg = os.path.join(data_dir, "mixed_trajectories_train.p")
    if os.path.isfile(leg):
        return leg
    if sig:
        raise FileNotFoundError(
            f"未找到多任务混合轨迹：{os.path.join(data_dir, multitask_mixed_basename(sig, ld))}、"
            f"{os.path.join(data_dir, multitask_mixed_basename(sig, 32))}、"
            f"{os.path.join(data_dir, f'mixed_{sig}.p')} 或 {leg}"
        )
    raise FileNotFoundError(f"未找到多任务混合轨迹：{leg}")


def _duo_repo_root() -> Path:
    """``DUO/`` 根目录（本文件位于 ``DUO/diffuser/utils/``）。"""
    return Path(__file__).resolve().parent.parent.parent


def multitask_mixed_paths_exist(
    data_dir: str, sig: str | None, latent_dim: int = 32
) -> bool:
    """是否与 :func:`resolve_multitask_mixed_path` 一致的「已存在」判定（不抛错）。"""
    import os

    ld = int(latent_dim)
    if not sig:
        leg = os.path.join(data_dir, "mixed_trajectories_train.p")
        return os.path.isfile(leg)
    p_lat = os.path.join(data_dir, multitask_mixed_basename(sig, ld))
    if os.path.isfile(p_lat):
        return True
    if ld != 32:
        return False
    p_short = os.path.join(data_dir, multitask_mixed_basename(sig, 32))
    if os.path.isfile(p_short):
        return True
    p_long = os.path.join(data_dir, f"mixed_{sig}.p")
    if os.path.isfile(p_long):
        return True
    leg = os.path.join(data_dir, "mixed_trajectories_train.p")
    return os.path.isfile(leg)


def ensure_multitask_mixed_trajectories(
    *,
    train_tasks_list: list[str],
    frac: float,
    sigma: float,
    seed: int,
    n_traj: int,
    k: int,
    eps: float,
    horizon: int,
    traj_params_json: str | None = None,
    fixed_dim: int = 128,
    skip_auto: bool = False,
    latent_dim: int = 32,
) -> None:
    """
    若 ``generated_datasets/multi_<tasks>_frac…/mixed_mt_*.p`` 不存在，则调用
    ``construct_trajectories.construct_trajectories`` 生成（与单独跑 construct 一致）。

    须在 **项目根 DUO** 为当前工作目录时调用，或在本函数内会临时 ``chdir`` 到该根目录。
    """
    csv = canonical_train_tasks_csv(",".join(train_tasks_list))
    tasks_sorted = [t.strip() for t in csv.split(",") if t.strip()]
    if len(tasks_sorted) < 2:
        return

    _, _, _, sig = prepare_multitask_traj(
        tasks_sorted, n_traj, k, eps, horizon, traj_params_json
    )
    train_tasks_str = multitask_path_token(csv)
    root = _duo_repo_root()
    from diffuser.utils.vae_layout import multitask_generated_candidate_rel_dirs

    rel_candidates = multitask_generated_candidate_rel_dirs(
        train_tasks_csv=csv,
        frac=float(frac),
        sigma=float(sigma),
        fixed_dim=int(fixed_dim),
        latent_dim=int(latent_dim),
    )
    data_dirs = [str(root / r) for r in rel_candidates]

    for data_dir in data_dirs:
        if multitask_mixed_paths_exist(data_dir, sig, latent_dim):
            return

    data_dir = data_dirs[0]
    rel_dir = rel_candidates[0]

    if skip_auto:
        resolve_multitask_mixed_path(data_dir, sig, latent_dim)

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from construct_trajectories import construct_trajectories

    print(
        f"[ensure_multitask_mixed_trajectories] 未找到混合轨迹，自动运行 construct_trajectories → {rel_dir}/mixed_*.p",
        flush=True,
    )
    _prev = os.getcwd()
    try:
        os.chdir(root)
        construct_trajectories(
            tasks_list=tasks_sorted,
            frac=frac,
            sigma=sigma,
            seed=seed,
            n_traj=n_traj,
            k=k,
            eps=eps,
            fixed_dim=fixed_dim,
            horizon=horizon,
            traj_params_json=traj_params_json,
            latent_dim=int(latent_dim),
        )
    finally:
        os.chdir(_prev)

    for dd in data_dirs:
        if multitask_mixed_paths_exist(dd, sig, latent_dim):
            return
    resolve_multitask_mixed_path(data_dir, sig, latent_dim)

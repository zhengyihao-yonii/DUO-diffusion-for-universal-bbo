"""
多任务轨迹参数：统一标量 vs 每任务字典；生成与 data_path / mixed 文件名一致的签名。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

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


def multitask_mixed_basename(traj_signature: str) -> str:
    """混合轨迹文件名，例如 ``mixed_mt_a1b2c3d4e5f67890.p``。"""
    return f"mixed_{multitask_slug_id(traj_signature)}.p"


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


def resolve_multitask_mixed_path(data_dir: str, sig: str | None) -> str:
    """
    优先短文件名 ``mixed_<slug_id>.p``；其次旧版 ``mixed_<完整sig>.p``；再 ``mixed_trajectories_train.p``。
    """
    import os

    if sig:
        p_short = os.path.join(data_dir, multitask_mixed_basename(sig))
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
            f"未找到多任务混合轨迹：{os.path.join(data_dir, multitask_mixed_basename(sig))}、"
            f"{os.path.join(data_dir, f'mixed_{sig}.p')} 或 {leg}"
        )
    raise FileNotFoundError(f"未找到多任务混合轨迹：{leg}")

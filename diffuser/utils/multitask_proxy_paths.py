"""多任务 / 单任务 real-world 下 proxy checkpoint 路径规则（供 train.py 与 evaluate.py 共用，避免 evaluate 导入 scripts.train 触发 CPU 线程副作用）。"""

from __future__ import annotations

import glob
import os


def multitask_proxy_prefix(task_name, Config):
    """与 trained_models/<task>_frac.../<n>x<h>_k<k>_eps<e>/seed<s>/ 一致。"""
    nd = getattr(Config, "traj_n_traj_dict", None)
    if nd is not None and task_name in nd:
        n = nd[task_name]
        k = getattr(Config, "traj_k_dict", {})[task_name]
        e = getattr(Config, "traj_eps_dict", {})[task_name]
    else:
        n, k, e = Config.n_traj, Config.k, Config.eps
    return (
        f"trained_models/{task_name}_frac{Config.frac}_sigma{Config.sigma}/"
        f"{n}x{Config.horizon}_k{k}_eps{e}/seed{Config.seed}/"
    )


def multitask_proxy_checkpoint_exists(prefix, Config):
    base = os.path.join(prefix, "proxy_checkpoint")
    if os.path.isfile(os.path.join(base, "state.pt")):
        return True
    if getattr(Config, "save_checkpoints", False):
        p = os.path.join(base, f"state_{Config.proxy_n_train_steps}.pt")
        if os.path.isfile(p):
            return True
        if glob.glob(os.path.join(base, "state_*.pt")):
            return True
    return False

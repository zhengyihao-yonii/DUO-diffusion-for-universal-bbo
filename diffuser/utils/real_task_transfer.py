"""
真实任务（real-task）从全任务 multitask text 预训练模型迁移：默认 ``mt_911054c35daad7e0_textcond_mttextonly``。

与 ``diffuser/datasets/real_world_fewshot``（LunarLander 等 JSON 数据）正交：后者为离线数据路径；
本模块仅负责 **预训练扩散 checkpoint 目录** 解析。
"""
from __future__ import annotations

import glob
import os
import re
from typing import Any

# 默认与当前主实验 sweep 的 multitask slug 一致（16 位 hex，不含 ``mt_`` 前缀）
DEFAULT_PRETRAINED_MT_HEX = os.environ.get(
    "GTG_REAL_TASK_PRETRAINED_MT_HEX", "911054c35daad7e0"
)
# 全 9 任务字典序 CSV（与 run_multitask.sh _FULL_MT_TASKS_CSV 一致）
DEFAULT_PRETRAINED_MULTITASK_CSV = os.environ.get(
    "GTG_REAL_TASK_PRETRAINED_MULTITASK_CSV",
    "ant,dkitty,gtopx2,gtopx3,gtopx4,gtopx6,superconductor,tfbind10,tfbind8",
)


def normalize_mt_hex(s: str) -> str:
    """接受 ``911054c35daad7e0``、``mt_9110...`` 或带 ``_textcond_mttextonly`` 的片段。"""
    t = s.strip().lower()
    if t.startswith("mt_"):
        t = t[3:]
    for suf in ("_textcond_mttextonly", "_textcond"):
        if t.endswith(suf):
            t = t[: -len(suf)]
    t = t.strip("_")
    if len(t) > 16:
        t = t[:16]
    return t


def pretrained_hyper_dir_name(mt_hex: str) -> str:
    h = normalize_mt_hex(mt_hex)
    return f"mt_{h}_textcond_mttextonly"


def resolve_multitask_pretrained_run_dir(
    *,
    multitask_train_tasks_csv: str,
    frac: float,
    sigma: float,
    mt_hex: str,
    seed: int,
) -> str:
    """``trained_models/multi_<token>_frac…/mt_<hex>_textcond_mttextonly/seed<n>/``（无尾斜杠）。"""
    from diffuser.utils.multitask_canon import multitask_path_token

    ts = multitask_path_token(multitask_train_tasks_csv)
    hyper = pretrained_hyper_dir_name(mt_hex)
    return os.path.join(
        f"trained_models/multi_{ts}_frac{frac}_sigma{sigma}",
        hyper,
        f"seed{int(seed)}",
    )


def resolve_diffusion_state_pt(ckpt_dir: str, config: Any | None = None) -> str | None:
    """
    与 ``scripts/evaluate._resolve_diffusion_checkpoint_path`` 一致：返回单个 ``.pt`` 文件路径。
    ``config`` 可为具有 ``n_train_steps`` / ``save_checkpoints`` 的对象或 ``None``。
    """
    if not os.path.isdir(ckpt_dir):
        return None
    st = os.path.join(ckpt_dir, "state.pt")
    if os.path.isfile(st):
        return st
    save_ck = True
    n_train = 0
    if config is not None:
        save_ck = bool(getattr(config, "save_checkpoints", True))
        n_train = int(getattr(config, "n_train_steps", 0) or 0)
    if save_ck and n_train > 0:
        p = os.path.join(ckpt_dir, f"state_{n_train}.pt")
        if os.path.isfile(p):
            return p
    matches = glob.glob(os.path.join(ckpt_dir, "state_*.pt"))
    if matches:

        def _step(path: str) -> int:
            m = re.search(r"state_(\d+)\.pt$", path)
            return int(m.group(1)) if m else -1

        return max(matches, key=_step)
    return None


def resolve_pretrained_diffusion_pt_for_real_task(
    *,
    multitask_train_tasks_csv: str,
    frac: float,
    sigma: float,
    mt_hex: str | None,
    pretrained_seed: int,
    config: Any | None = None,
) -> str | None:
    """预训练 multitask text 模型下的 ``state*.pt`` 绝对路径（相对于 cwd）。"""
    h = mt_hex if mt_hex is not None else DEFAULT_PRETRAINED_MT_HEX
    run_dir = resolve_multitask_pretrained_run_dir(
        multitask_train_tasks_csv=multitask_train_tasks_csv,
        frac=frac,
        sigma=sigma,
        mt_hex=h,
        seed=pretrained_seed,
    )
    ckpt_dir = os.path.join(run_dir, "checkpoint")
    return resolve_diffusion_state_pt(ckpt_dir, config)

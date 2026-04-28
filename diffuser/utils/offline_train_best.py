"""
与 ZipDataset / 训练管线一致的「离线训练子集」上标签 y 的最优值（越大越好），用于表格 D(best)。

真实任务（lunar_lander / robot_push / rover）：在传入与 ``construct_trajectories`` 相同的
few-shot 参数时，D(best) 为 **该 few-shot 池内** 原始 ``y`` 的最大值；未传参时退化为全量合并 JSON
（与旧行为一致）。其它任务逻辑不变。
"""
from __future__ import annotations

import numpy as np
import diffuser.numpy_design_bench_compat  # noqa: F401

import design_bench

from diffuser.datasets.sequence import TASKNAME2MAX_SAMPLES, TASKNAME2TASK
from diffuser.utils.soo_gtopx import is_gtopx_task, load_gtopx_offline_arrays
from diffuser.datasets.real_world_fewshot import (
    is_real_world_fewshot_task,
    load_real_world_raw,
    load_real_world_y_min_max_full,
)


def offline_full_dataset_y_bounds(
    task_name: str,
    *,
    frac: float,
    sigma: float,
    seed: int,
) -> tuple[float, float]:
    """
    真实任务：``fewshot_data`` 下合并全部 JSON 的 y 的 ``(min, max)``（与 nmax 全库归一化一致）。

    非真实任务：与 ``offline_training_best_y`` 所用离线子集上 ``y`` 的 min/max 一致（Design-Bench 子集）。
    """
    if is_real_world_fewshot_task(task_name):
        return load_real_world_y_min_max_full(task_name)
    task = design_bench.make(
        TASKNAME2TASK[task_name],
        dataset_kwargs=dict(
            max_samples=int(TASKNAME2MAX_SAMPLES[task_name] * frac),
            distribution=None,
            min_percentile=0,
        ),
    )
    if task_name.startswith("tfbind"):
        task.map_to_logits()
    y = np.asarray(task.y, dtype=np.float64).ravel()
    return float(np.min(y)), float(np.max(y))


def offline_full_dataset_best_y(
    task_name: str,
    *,
    frac: float,
    sigma: float,
    seed: int,
) -> float:
    """全离线库上 best y（越大越好）；真实任务即 ``fewshot_data`` 合并后 ``max(y)``。"""
    lo, hi = offline_full_dataset_y_bounds(
        task_name, frac=frac, sigma=sigma, seed=seed
    )
    return float(hi)


def offline_training_best_y(
    task_name: str,
    *,
    frac: float,
    sigma: float,
    seed: int,
    real_world_fewshot_k: int | None = None,
    real_world_fewshot_mode: str = "all",
    real_world_fewshot_seed: int | None = None,
) -> float:
    """
    与当前 Config.frac / sigma / seed 下 ZipDataset 使用的数据子集一致，
    返回该子集上原始回报 y 的最大值。

    真实任务可选 ``real_world_fewshot_*``：与 ``load_real_world_raw`` 子集一致；
    ``real_world_fewshot_seed is None`` 时用 ``seed``（通常为 Config.seed）。
    """
    if is_gtopx_task(task_name):
        _, _, _, _, y_subset_max = load_gtopx_offline_arrays(
            task_name, frac=frac, sigma=sigma, seed=seed
        )
        return float(y_subset_max)

    if is_real_world_fewshot_task(task_name):
        fs = int(real_world_fewshot_seed) if real_world_fewshot_seed is not None else int(seed)
        _x, y = load_real_world_raw(
            task_name,
            fewshot_k=real_world_fewshot_k,
            fewshot_mode=real_world_fewshot_mode,  # type: ignore[arg-type]
            fewshot_seed=fs,
        )
        y = np.asarray(y, dtype=np.float64).ravel()
        return float(np.max(y))

    task = design_bench.make(
        TASKNAME2TASK[task_name],
        dataset_kwargs=dict(
            max_samples=int(TASKNAME2MAX_SAMPLES[task_name] * frac),
            distribution=None,
            min_percentile=0,
        ),
    )
    if task_name.startswith("tfbind"):
        task.map_to_logits()
    return float(np.max(task.y))

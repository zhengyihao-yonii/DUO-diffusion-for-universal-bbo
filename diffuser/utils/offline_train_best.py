"""
与 ZipDataset / 训练管线一致的「离线训练子集」上标签 y 的最优值（越大越好），用于表格 D(best)。
非全局最优；亦非仅全库 min/max 参考（evaluate 里 DesignBenchFunctionWrapper 打印的那对）。
"""
from __future__ import annotations

import numpy as np
import diffuser.numpy_design_bench_compat  # noqa: F401

import design_bench

from diffuser.datasets.sequence import TASKNAME2MAX_SAMPLES, TASKNAME2TASK
from diffuser.utils.soo_gtopx import is_gtopx_task, load_gtopx_offline_arrays


def offline_training_best_y(task_name: str, *, frac: float, sigma: float, seed: int) -> float:
    """
    与当前 Config.frac / sigma / seed 下 ZipDataset 使用的数据子集一致，
    返回该子集上原始回报 y 的最大值。
    """
    if is_gtopx_task(task_name):
        _, _, _, _, y_subset_max = load_gtopx_offline_arrays(
            task_name, frac=frac, sigma=sigma, seed=seed
        )
        return float(y_subset_max)

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

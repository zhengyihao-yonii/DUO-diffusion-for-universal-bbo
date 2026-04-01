"""
SOO-Bench GTOPX 离线数据与 Oracle（benchmark 2/3/4/6，无约束）。
需已安装可编辑包: pip install -e thirdparty_benchmark/SOO-Bench
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

# 任务短名 -> SOO benchmark id（与 universal-offline-bbo gtopx_data_{id}_1 一致）
TASKNAME_TO_GTOPX_BENCHMARK: Dict[str, int] = {
    "gtopx2": 2,
    "gtopx3": 3,
    "gtopx4": 4,
    "gtopx6": 6,
}

GTOPX_TASK_NAMES = frozenset(TASKNAME_TO_GTOPX_BENCHMARK.keys())

# 各 benchmark 决策变量维数（与 SOO-Bench Task_GTOPX 一致）
TASKNAME_TO_VAR_NUM: Dict[str, int] = {
    "gtopx2": 22,
    "gtopx3": 18,
    "gtopx4": 26,
    "gtopx6": 22,
}

# 与 Design-Bench 的 TASKNAME2MAX_SAMPLES 类似：用于粗略上界（SOO 实际样本数为 var*1000 再经分位过滤）
TASKNAME2MAX_SAMPLES: Dict[str, int] = {
    name: TASKNAME_TO_VAR_NUM[name] * 1000 for name in TASKNAME_TO_VAR_NUM
}


def _ensure_soo_importable() -> None:
    try:
        from soo_bench.Taskdata import OfflineTask, set_use_cache  # noqa: F401
        return
    except ImportError:
        pass
    # 本文件位于 <项目根>/diffuser/utils/soo_gtopx.py → 向上两级为项目根（GTGdfgo 或 GTG）
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    workspace = os.path.dirname(project_root)
    env_root = os.environ.get("SOO_BENCH_ROOT", "").strip()
    candidates = [
        env_root if env_root else None,
        os.path.join(project_root, "thirdparty_benchmark", "SOO-Bench"),
        os.path.join(workspace, "GTGdfgo", "thirdparty_benchmark", "SOO-Bench"),
    ]
    for c in candidates:
        if not c or not os.path.isdir(c):
            continue
        if c not in sys.path:
            sys.path.insert(0, c)
    try:
        from soo_bench.Taskdata import OfflineTask, set_use_cache  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "无法导入 soo_bench。请任选其一：在已激活环境中执行 "
            "`pip install -e /path/to/SOO-Bench`；或设置环境变量 "
            "SOO_BENCH_ROOT 指向 SOO-Bench 仓库根目录（内含 soo_bench/ 包）。"
        ) from e


def load_gtopx_offline_arrays(
    task_name: str,
    *,
    frac: float = 1.0,
    sigma: float = 0.0,
    seed: int = 1,
    percentile_low: int = 25,
    percentile_high: int = 75,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    返回 (x, y_normalized_01, y_min_ref, y_max_ref)。
    y 用全分位样本上的 min/max 归一化到 [0,1]，与 Design-Bench + ZipDataset 行为一致。
    """
    if task_name not in TASKNAME_TO_GTOPX_BENCHMARK:
        raise KeyError(f"Unknown GTOPX task short name: {task_name}")
    bid = TASKNAME_TO_GTOPX_BENCHMARK[task_name]
    _ensure_soo_importable()
    from soo_bench.Taskdata import OfflineTask, set_use_cache

    set_use_cache(True)
    full_ot = OfflineTask("gtopx_data", benchmark=bid, seed=seed)
    full_ot.sample_bound(num=0, low=0, high=100)
    y_min = float(full_ot.y.min())
    y_max = float(full_ot.y.max())

    small_ot = OfflineTask("gtopx_data", benchmark=bid, seed=seed)
    small_ot.sample_bound(num=0, low=percentile_low, high=percentile_high)
    n = len(small_ot.x)
    if frac < 1.0:
        n_take = max(1, int(n * frac))
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, n_take, replace=False)
        x = small_ot.x[idx].astype(np.float32)
        y = small_ot.y[idx].astype(np.float64)
    else:
        x = small_ot.x.astype(np.float32)
        y = small_ot.y.astype(np.float64)

    denom = y_max - y_min + 1e-12
    y_norm = (y.squeeze(-1) - y_min) / denom
    if sigma > 0.0:
        rng = np.random.RandomState(seed + 17)
        y_norm = np.clip(
            y_norm + rng.randn(*y_norm.shape) * sigma,
            0.0,
            1.0,
        )
    return x, y_norm.astype(np.float32), y_min, y_max


class GtopxOracleTask:
    """与 design_bench task 类似，提供 predict(x) 与参考 y 范围。"""

    def __init__(self, task_name: str, seed: int = 1):
        if task_name not in TASKNAME_TO_GTOPX_BENCHMARK:
            raise KeyError(task_name)
        self.task_name = task_name
        self.benchmark_id = TASKNAME_TO_GTOPX_BENCHMARK[task_name]
        _ensure_soo_importable()
        from soo_bench.Taskdata import OfflineTask, set_use_cache

        set_use_cache(True)
        self._offline = OfflineTask("gtopx_data", benchmark=self.benchmark_id, seed=seed)
        self._offline.sample_bound(num=0, low=0, high=100)
        self.x = self._offline.x.astype(np.float32)
        self.y = np.asarray(self._offline.y, dtype=np.float64)
        self.min = float(self.y.min())
        self.max = float(self.y.max())

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        v, _ = self._offline.predict(x)
        return np.asarray(v, dtype=np.float64).reshape(-1, 1)


def is_gtopx_task(name: str) -> bool:
    return name in GTOPX_TASK_NAMES

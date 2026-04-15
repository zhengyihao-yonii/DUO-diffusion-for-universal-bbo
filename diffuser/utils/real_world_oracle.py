# -*- coding: utf-8 -*-
"""
真实任务（LunarLander / RobotPush / Rover）的 Oracle 评估。

实现位于 ``diffuser.real_world_sim``（由 mcts-transfer 迁入）。依赖：

- **LunarLander**：``gym`` + ``Box2D``（``pip install gymnasium[box2d]`` 或 ``gym[box2d]``）
- **RobotPush**：``pygame`` + ``Box2D``
- **Rover**：``numpy`` + ``scipy``

输入 ``x`` 为与 JSON 一致的设计向量（[0,1] 归一化，维数见 ``REAL_WORLD_FEWSHOT_TASK_SPECS``）。
"""
from __future__ import annotations

from typing import Callable

import numpy as np

_REAL_WORLD_TASKS = frozenset({"lunar_lander", "robot_push", "rover"})


def is_real_world_oracle_task(name: str) -> bool:
    return name in _REAL_WORLD_TASKS


def _lunar_predict_batch(x: np.ndarray, n_envs: int = 50) -> np.ndarray:
    """x: (N, 12) in [0,1]；返回每条轨迹在 n_envs 个随机种子下的平均回报（越大越好）。"""
    import torch

    from diffuser.real_world_sim.lunar_lander import simulate_lunar_rover

    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n = x.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        xt = torch.tensor(x[i], dtype=torch.float32)
        rewards = []
        for env_seed in range(n_envs):
            rewards.append(simulate_lunar_rover((xt, env_seed)))
        out[i] = float(np.mean(rewards))
    return out


def _push_factory() -> Callable[[np.ndarray], np.ndarray]:
    from diffuser.real_world_sim.push_oracle import PushOracle

    pr = PushOracle()

    def _fn(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.array([float(pr(xi)) for xi in x], dtype=np.float64)

    return _fn


def _rover_factory() -> Callable[[np.ndarray], np.ndarray]:
    from diffuser.real_world_sim.rover_oracle import RoverOracle

    rv = RoverOracle()

    def _fn(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return np.array([float(rv(xi)) for xi in x], dtype=np.float64)

    return _fn


_ORACLE_CACHE: dict[str, Callable[[np.ndarray], np.ndarray]] = {}


def get_real_world_oracle_predict_fn(task_key: str) -> Callable[[np.ndarray], np.ndarray]:
    """返回 ``f(x) -> y``，``x`` 为 (N, dim) numpy，``y`` 为 (N,)。"""
    if task_key in _ORACLE_CACHE:
        return _ORACLE_CACHE[task_key]
    if task_key == "lunar_lander":
        fn = _lunar_predict_batch
    elif task_key == "robot_push":
        fn = _push_factory()
    elif task_key == "rover":
        fn = _rover_factory()
    else:
        raise KeyError(task_key)
    _ORACLE_CACHE[task_key] = fn
    return fn


def oracle_predict(task_key: str, x: np.ndarray) -> np.ndarray:
    """对批量设计 ``x`` 调用真实 Oracle，返回与 ``x`` 行数一致的标量回报。"""
    return get_real_world_oracle_predict_fn(task_key)(x)


# 兼容旧代码：曾通过 MCTS_TRANSFER_ROOT 解析
def resolve_mcts_transfer_root():
    """已弃用：Oracle 已内置，始终返回 None。"""
    return None

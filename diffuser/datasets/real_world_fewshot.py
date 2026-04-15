# -*- coding: utf-8 -*-
"""
Real-world few-shot 任务（LunarLander、RobotPush、Rover）：数据来自
``fewshot_data/<TaskName>/{similar,unsimilar}/*.json``，每文件含 ``X``、``y``。

Few-shot 子集：``fewshot_k`` + ``fewshot_mode``（``random`` / ``worst``）；
亦可由环境变量 ``GTG_REAL_WORLD_FEWSHOT_K``、``GTG_REAL_WORLD_FEWSHOT_MODE`` 覆盖（与 construct / ZipDataset 一致）。
假设 ``y`` 越大越好，则 **worst** = ``y`` 最小的 k 个点。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import numpy as np

FewshotMode = Literal["all", "random", "worst"]

# 与 CLI / Config 默认值一致时可不设环境变量
ENV_FEWSHOT_K = "GTG_REAL_WORLD_FEWSHOT_K"
ENV_FEWSHOT_MODE = "GTG_REAL_WORLD_FEWSHOT_MODE"
ENV_FEWSHOT_SEED = "GTG_REAL_WORLD_FEWSHOT_SEED"

# GTGdfgo 任务短名 -> fewshot_data 下目录名
TASK_KEY_TO_DATA_DIR: dict[str, str] = {
    "lunar_lander": "LunarLander",
    "robot_push": "RobotPush",
    "rover": "Rover",
}

REAL_WORLD_FEWSHOT_TASK_SPECS: dict[str, dict] = {
    "lunar_lander": {"dim": 12},
    "robot_push": {"dim": 14},
    "rover": {"dim": 60},
}


def is_real_world_fewshot_task(name: str) -> bool:
    return name in REAL_WORLD_FEWSHOT_TASK_SPECS


def _gtgdfgo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_real_world_data_root() -> Path:
    """默认 ``<GTGdfgo>/fewshot_data``；可用环境变量 ``GTG_REAL_WORLD_FEWSHOT_DIR`` 覆盖。"""
    env = os.environ.get("GTG_REAL_WORLD_FEWSHOT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return _gtgdfgo_root() / "fewshot_data"


def _collect_json_paths(task_dir: Path) -> list[Path]:
    out: list[Path] = []
    for sub in ("similar", "unsimilar"):
        d = task_dir / sub
        if d.is_dir():
            out.extend(sorted(d.glob("*.json")))
    return out


def resolve_fewshot_params(
    fewshot_k: int | None,
    fewshot_mode: FewshotMode,
    fewshot_seed: int,
) -> tuple[int | None, FewshotMode, int]:
    """环境变量覆盖显式参数（未设置环境变量时保持传入值）。"""
    k = fewshot_k
    mode: FewshotMode = fewshot_mode
    seed = fewshot_seed
    if os.environ.get(ENV_FEWSHOT_K, "").strip():
        k = int(os.environ[ENV_FEWSHOT_K])
    if os.environ.get(ENV_FEWSHOT_MODE, "").strip():
        m = os.environ[ENV_FEWSHOT_MODE].strip().lower()
        if m not in ("all", "random", "worst"):
            raise ValueError(
                f"{ENV_FEWSHOT_MODE} 须为 all|random|worst，收到 {m!r}"
            )
        mode = m  # type: ignore[assignment]
    if os.environ.get(ENV_FEWSHOT_SEED, "").strip():
        seed = int(os.environ[ENV_FEWSHOT_SEED])
    return k, mode, seed


def select_real_world_fewshot(
    x: np.ndarray,
    y: np.ndarray,
    k: int | None,
    mode: FewshotMode,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    在已合并的 ``(x, y)`` 上取 few-shot 子集。
    ``all`` 或 ``k is None`` 或 ``k >= n``：返回全集。
    ``worst``：``y`` 最小的 k 条（越大越好）。
    ``random``：无放回随机 k 条。
    """
    n = len(y)
    if k is None or mode == "all" or k >= n:
        return x, y
    if k < 1:
        raise ValueError("fewshot_k 须 >= 1 或 None")
    if mode == "worst":
        idx = np.argsort(y.astype(np.float64))[:k]
        return x[idx], y[idx]
    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=k, replace=False)
        return x[idx], y[idx]
    raise ValueError(f"未知 fewshot_mode: {mode}")


def load_real_world_arrays_from_json(
    task_key: str,
    data_root: Path | None = None,
    *,
    fewshot_k: int | None = None,
    fewshot_mode: FewshotMode = "all",
    fewshot_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """合并 ``similar``/``unsimilar`` 下所有 JSON 的 ``X``、``y``，再按 few-shot 规则子采样。"""
    root = data_root if data_root is not None else default_real_world_data_root()
    dir_name = TASK_KEY_TO_DATA_DIR.get(task_key)
    if not dir_name:
        raise KeyError(f"未知 real-world 任务: {task_key}")
    task_dir = root / dir_name
    if not task_dir.is_dir():
        raise FileNotFoundError(
            f"Real-world 数据目录不存在: {task_dir}\n"
            f"请将数据置于 fewshot_data/{dir_name}/（含 similar/、unsimilar/ 与 *.json），"
            f"或设置 GTG_REAL_WORLD_FEWSHOT_DIR 指向含该子目录的路径。"
        )
    paths = _collect_json_paths(task_dir)
    if not paths:
        raise FileNotFoundError(
            f"未找到 JSON: {task_dir}/similar|unsimilar/*.json"
        )
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for jp in paths:
        with open(jp, "r", encoding="utf-8") as f:
            j = json.load(f)
        if "X" not in j or "y" not in j:
            raise KeyError(f"{jp} 须含键 'X' 与 'y'")
        x = np.asarray(j["X"], dtype=np.float32)
        y = np.asarray(j["y"], dtype=np.float32).reshape(-1)
        if x.shape[0] != len(y):
            raise ValueError(f"{jp}: X/y 行数不一致 {x.shape[0]} vs {len(y)}")
        xs.append(x)
        ys.append(y)
    x_cat = np.vstack(xs)
    y_cat = np.concatenate(ys)
    spec_dim = REAL_WORLD_FEWSHOT_TASK_SPECS[task_key]["dim"]
    if x_cat.shape[1] != spec_dim:
        raise ValueError(
            f"{task_key}: 期望设计维度 {spec_dim}，合并后得到 {x_cat.shape[1]}"
        )
    fk, fm, fs = resolve_fewshot_params(fewshot_k, fewshot_mode, fewshot_seed)
    return select_real_world_fewshot(x_cat, y_cat, fk, fm, fs)


def load_real_world_raw(
    task_key: str,
    data_root: Path | None = None,
    *,
    fewshot_k: int | None = None,
    fewshot_mode: FewshotMode = "all",
    fewshot_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    return load_real_world_arrays_from_json(
        task_key,
        data_root=data_root,
        fewshot_k=fewshot_k,
        fewshot_mode=fewshot_mode,
        fewshot_seed=fewshot_seed,
    )


def load_real_world_y_min_max_full(
    task_key: str,
    data_root: Path | None = None,
) -> tuple[float, float]:
    """全量合并 JSON（无 few-shot）上 ``y`` 的 min/max，供 Oracle 评估归一化与 D(best) 参考。"""
    _x, y = load_real_world_arrays_from_json(
        task_key,
        data_root=data_root,
        fewshot_k=None,
        fewshot_mode="all",
        fewshot_seed=0,
    )
    y = np.asarray(y, dtype=np.float64).ravel()
    return float(y.min()), float(y.max())


def load_real_world_for_pipeline(
    task_key: str,
    fixed_length: int,
    frac: float = 1.0,
    sigma: float = 0.0,
    data_root: Path | None = None,
    *,
    fewshot_k: int | None = None,
    fewshot_mode: FewshotMode = "all",
    fewshot_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    与 DesignBenchDatasetWrapper 对齐：x 填充/截断到 fixed_length；y 在**当前子集**上 min-max 到 [0,1] 再加 sigma 噪声。
    返回 processed_x [N,L], y_norm [N], original_dim

    few-shot 在合并 JSON 之后、``frac`` 随机子采样之前应用（先 worst/random k，再可选 frac）。
    """
    fk, fm, fs = resolve_fewshot_params(fewshot_k, fewshot_mode, fewshot_seed)
    x_raw, y_raw = load_real_world_raw(
        task_key,
        data_root=data_root,
        fewshot_k=fk,
        fewshot_mode=fm,
        fewshot_seed=fs,
    )
    n = len(x_raw)
    if frac < 1.0:
        rng = np.random.default_rng(42)
        n_take = max(1, int(n * frac))
        idx = rng.choice(n, size=n_take, replace=False)
        x_raw = x_raw[idx]
        y_raw = y_raw[idx]
    original_dim = REAL_WORLD_FEWSHOT_TASK_SPECS[task_key]["dim"]
    y_min, y_max = float(y_raw.min()), float(y_raw.max())
    if y_max <= y_min:
        y_norm = np.zeros(len(y_raw), dtype=np.float32)
    else:
        y_norm = (y_raw - y_min) / (y_max - y_min)
    if sigma > 0.0:
        y_norm = np.clip(
            y_norm + np.random.randn(*y_norm.shape).astype(np.float32) * sigma,
            0.0,
            1.0,
        )
    proc = np.zeros((len(x_raw), fixed_length), dtype=np.float32)
    for i in range(len(x_raw)):
        flat = x_raw[i].reshape(-1)
        d = min(len(flat), fixed_length)
        proc[i, :d] = flat[:d]
    return proc, y_norm.astype(np.float32), original_dim

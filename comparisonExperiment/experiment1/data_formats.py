from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import json
import pickle as pkl

import numpy as np
import torch


@dataclass(frozen=True)
class DuoTrajectoryPkl:
    """DUO single-task pkl format: [points, values, pr, rtg, timesteps]."""

    points: torch.Tensor  # [n_traj, horizon, d_x]
    values: torch.Tensor  # [n_traj, horizon]
    pr: torch.Tensor  # [n_traj, horizon]
    rtg: torch.Tensor  # [n_traj, horizon]
    timesteps: torch.Tensor  # [n_traj, horizon]

    def to_obj(self):
        return [self.points, self.values, self.pr, self.rtg, self.timesteps]


def save_duo_single_task_pkl(path: Path, obj: DuoTrajectoryPkl) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pkl.dump(obj.to_obj(), f)


def _as_token_list(x: np.ndarray) -> list[str]:
    # Chinese comment: 简单把每个维度转成 token 字符串，UniSO 的 data2str 也是类似思路。
    return [f"x{i}={float(v):+.4f}" for i, v in enumerate(x.reshape(-1).tolist())]


def save_uniso_jsonl(
    *,
    json_path: Path,
    metadata_path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    metadata_text: str,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    recs = [{"x": _as_token_list(x), "y": float(y)} for x, y in zip(xs, ys)]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(recs, f, indent=2, ensure_ascii=False)
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(str(metadata_text))


def flatten_trajs(points: torch.Tensor, values: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Flatten [n_traj, horizon, d] to [N, d] and [N]."""
    x = points.detach().cpu().numpy().reshape(-1, points.shape[-1])
    y = values.detach().cpu().numpy().reshape(-1)
    return x, y


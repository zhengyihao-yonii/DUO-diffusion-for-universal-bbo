"""轨迹构建阶段的设备与 pairwise 距离矩阵（与 GTG-main 一致，供 construct_trajectories 预计算 L2 矩阵）。"""
from __future__ import annotations

import os

import torch


def resolve_torch_device(explicit: str | None = None) -> torch.device:
    if explicit is not None and str(explicit).strip():
        return torch.device(str(explicit).strip())
    env = os.environ.get("GTG_DEVICE", "").strip()
    if env:
        return torch.device(env)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pairwise_l2_distance_matrix(points: torch.Tensor, chunk_rows: int = 1024) -> torch.Tensor:
    """
    计算 ``points`` (N, D) 的全 pairwise L2 距离矩阵 (N, N)，结果在 **CPU** float32。
    与 GTG-main 一致：CUDA 可用且未设置 ``GTG_DISTANCE_ON_GPU=0`` 时用 GPU 分块写回 CPU。
    """
    points = points.detach().float().contiguous()
    n = points.shape[0]
    if n == 0:
        return torch.empty(0, 0, dtype=torch.float32)

    use_gpu = os.environ.get("GTG_DISTANCE_ON_GPU", "1") != "0" and torch.cuda.is_available()
    if not use_gpu:
        return torch.cdist(points, points, p=2)

    dev = resolve_torch_device()
    if dev.type != "cuda":
        return torch.cdist(points, points, p=2)

    pts = points.to(dev, non_blocking=True)
    out = torch.empty(n, n, dtype=torch.float32)
    for i in range(0, n, chunk_rows):
        end = min(i + chunk_rows, n)
        block = torch.cdist(pts[i:end], pts, p=2)
        out[i:end] = block.cpu()
    return out

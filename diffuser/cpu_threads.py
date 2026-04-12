"""
限制 CPU 线程数，降低 OpenMP / BLAS / PyTorch 占满宿主机 CPU。

用法（任选其一）::

    export CPU_THREADS=4
    python train.py ... --cpu_threads 4

须在尽可能早的时机调用 :func:`maybe_apply_from_argv_and_env`（早于 numpy / sklearn /
torch 的首次数值计算）；并在 ``import torch`` 之后调用 :func:`apply_torch_cpu_threads_from_env`。
"""

from __future__ import annotations

import os
import sys
from typing import Optional

_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def parse_cpu_threads_arg(argv: Optional[list[str]] = None) -> Optional[int]:
    """
    从 ``sys.argv`` 解析 ``--cpu_threads N`` / ``--cpu_threads=N``；
    若无则读环境变量 ``CPU_THREADS``。
    """
    if argv is None:
        argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--cpu_threads" and i + 1 < len(argv):
            try:
                return max(1, int(argv[i + 1]))
            except ValueError:
                return None
        if a.startswith("--cpu_threads="):
            try:
                return max(1, int(a.split("=", 1)[1]))
            except ValueError:
                return None
        i += 1
    raw = os.environ.get("CPU_THREADS", "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        return None


def apply_env_cpu_threads(n: int) -> None:
    """设置常见 BLAS/OpenMP 线程环境变量，并写入 ``CPU_THREADS`` 供下游读取。"""
    n = max(1, int(n))
    s = str(n)
    for k in _ENV_KEYS:
        os.environ[k] = s
    os.environ["CPU_THREADS"] = s


def maybe_apply_from_argv_and_env(argv: Optional[list[str]] = None) -> Optional[int]:
    """
    进程入口尽早调用：在导入 numpy / sklearn / torch 之前执行。
    若未指定线程数则返回 ``None``，不改变默认行为。
    """
    n = parse_cpu_threads_arg(argv)
    if n is None:
        return None
    apply_env_cpu_threads(n)
    return n


def apply_torch_cpu_threads(n: int) -> None:
    """在 ``import torch`` 之后调用，限制 PyTorch intra-op 线程。"""
    import torch

    n = max(1, int(n))
    torch.set_num_threads(n)
    inter = min(4, max(1, n))
    try:
        torch.set_num_interop_threads(inter)
    except RuntimeError:
        pass


def apply_torch_cpu_threads_from_env() -> None:
    """若已设置 ``CPU_THREADS``（含 :func:`maybe_apply_from_argv_and_env` 写入），则配置 torch。"""
    raw = os.environ.get("CPU_THREADS", "").strip()
    if not raw:
        return
    try:
        apply_torch_cpu_threads(int(raw))
    except ValueError:
        pass


def dataloader_num_workers_cap(preferred: int = 4) -> int:
    """
    在限制 CPU 时避免 DataLoader 再开多进程 worker（与 BLAS 线程叠加易打满 CPU）。
    未设置 ``CPU_THREADS`` 时返回 ``preferred``。
    """
    if not os.environ.get("CPU_THREADS", "").strip():
        return preferred
    return 0

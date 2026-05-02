from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import math

import numpy as np
import torch


@dataclass(frozen=True)
class InstanceTransform:
    """Immutable affine instance map z -> x.

    Chinese comment: z 为母目标空间；x 为各任务异构设计空间（d_x 可不同）。
    """

    A: torch.Tensor  # [d_x, d_z]
    b: torch.Tensor  # [d_x]
    obs_noise_std: float = 0.0

    @property
    def d_x(self) -> int:
        return int(self.A.shape[0])

    def map(self, z: torch.Tensor, *, rng: torch.Generator) -> torch.Tensor:
        x = z @ self.A.T + self.b
        if float(self.obs_noise_std) > 0.0:
            eps = torch.randn_like(x, generator=rng) * float(self.obs_noise_std)
            x = x + eps
        return x


@dataclass(frozen=True)
class LatentObjective:
    """Shared latent objective f(z) (lower is better internally; we will convert to y_norm)."""

    name: Literal["branin", "ackley"]
    d_z: int

    def eval(self, z: torch.Tensor) -> torch.Tensor:
        if self.name == "branin":
            if int(self.d_z) != 2:
                raise ValueError("branin requires d_z=2")
            x1 = z[..., 0] * 5.0 + 2.5  # map [-1,1] -> [ -2.5, 7.5 ]
            x2 = z[..., 1] * 7.5 + 7.5  # map [-1,1] -> [ 0, 15 ]
            a = 1.0
            b = 5.1 / (4.0 * math.pi**2)
            c = 5.0 / math.pi
            r = 6.0
            s = 10.0
            t = 1.0 / (8.0 * math.pi)
            return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * torch.cos(x1) + s
        if self.name == "ackley":
            d = int(self.d_z)
            if d < 1:
                raise ValueError("ackley requires d_z>=1")
            x = z * 5.0
            a = 20.0
            b = 0.2
            c = 2 * math.pi
            s1 = torch.mean(x**2, dim=-1)
            s2 = torch.mean(torch.cos(c * x), dim=-1)
            return a * torch.exp(-b * torch.sqrt(s1 + 1e-12)) + torch.exp(s2) - a - math.e
        raise ValueError(f"unknown objective: {self.name}")


@dataclass(frozen=True)
class TaskSpec:
    """A single task instance specification."""

    task_id: str
    gap: float
    transform: InstanceTransform


@dataclass(frozen=True)
class TaskFamilySpec:
    """Family with heterogeneous train d_x and test d_x; shared mother objective in z."""

    objective: LatentObjective
    d_pad: int
    train_tasks: tuple[TaskSpec, ...]
    test_task: TaskSpec

    @property
    def max_d_x(self) -> int:
        return int(self.d_pad)

    def geometric_distance(self, *, mc: int = 4096, seed: int = 0) -> float:
        """Monte-Carlo E|| pad(x_test) - mean_i pad(x_train_i) || in shared padded R^{d_pad}."""
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        d_z = int(self.objective.d_z)
        z = torch.rand((int(mc), d_z), generator=gen) * 2.0 - 1.0
        d_pad = int(self.d_pad)

        def _pad(x: torch.Tensor) -> torch.Tensor:
            # x: [mc, d_x] -> [mc, d_pad]
            d_x = int(x.shape[-1])
            if d_x == d_pad:
                return x
            out = torch.zeros((x.shape[0], d_pad), dtype=x.dtype, device=x.device)
            out[:, :d_x] = x
            return out

        x_test = self.test_task.transform.map(z, rng=gen)
        xs = [t.transform.map(z, rng=gen) for t in self.train_tasks]
        pads = [_pad(x) for x in xs]
        x_mean = torch.stack(pads, dim=0).mean(dim=0)
        d = torch.linalg.norm(_pad(x_test) - x_mean, dim=-1).mean()
        return float(d.item())


def _rand_orthonormal(d_x: int, d_z: int, *, rng: np.random.Generator) -> np.ndarray:
    m = rng.normal(size=(d_x, d_z)).astype(np.float64)
    q, _r = np.linalg.qr(m)
    return q[:, :d_z]


def _pad_Ab(
    A: np.ndarray, b: np.ndarray, *, d_pad: int
) -> tuple[np.ndarray, np.ndarray]:
    d_x, d_z = A.shape
    Ap = np.zeros((d_pad, d_z), dtype=np.float64)
    Ap[:d_x] = A
    bp = np.zeros((d_pad,), dtype=np.float64)
    bp[:d_x] = b
    return Ap, bp


def make_family(
    *,
    objective_name: str = "branin",
    d_z: int = 2,
    train_d_x: tuple[int, ...] = (10, 12, 14, 18, 20),
    test_d_x: int = 16,
    d_pad: int = 32,
    gap: float = 0.0,
    obs_noise_std: float = 0.0,
    seed: int = 0,
) -> TaskFamilySpec:
    """Heterogeneous train observation dims + test dim; gap drifts test affine in padded mean space.

    English doc: Each train task gets its own random orthonormal A:[d_x_i,d_z]. Test task uses
    A_mean + gap * unit direction over padded train transforms (same spirit as single-d_x exp1).
    Shared DUO padding dimension ``d_pad`` (default 32) must be >= every native ``d_x``.
    """
    rng = np.random.default_rng(int(seed))
    on = str(objective_name).strip().lower()
    if on not in ("branin", "ackley"):
        raise ValueError(f"objective_name must be branin|ackley, got {objective_name!r}")
    obj = LatentObjective(name=on, d_z=int(d_z))  # type: ignore[arg-type]
    nt = len(train_d_x)
    if nt < 1:
        raise ValueError("train_d_x must be non-empty")
    if int(test_d_x) < 1:
        raise ValueError("test_d_x must be >= 1")
    d_pad = int(d_pad)
    d_nat_max = int(max(max(train_d_x), int(test_d_x)))
    if d_pad < d_nat_max:
        raise ValueError(f"d_pad={d_pad} must be >= max native observation dim ({d_nat_max})")

    train: list[TaskSpec] = []
    a_pads: list[np.ndarray] = []
    b_pads: list[np.ndarray] = []
    for i, d_i in enumerate(train_d_x):
        name = f"D_train_{i + 1}"
        Ai = _rand_orthonormal(int(d_i), int(d_z), rng=rng)
        bi = rng.normal(scale=0.2, size=(int(d_i),)).astype(np.float64)
        train.append(
            TaskSpec(
                task_id=name,
                gap=0.0,
                transform=InstanceTransform(
                    A=torch.from_numpy(Ai).to(dtype=torch.float32),
                    b=torch.from_numpy(bi).to(dtype=torch.float32),
                    obs_noise_std=float(obs_noise_std),
                ),
            )
        )
        Ap, bp = _pad_Ab(Ai, bi, d_pad=d_pad)
        a_pads.append(Ap)
        b_pads.append(bp)

    A_mean = np.mean(np.stack(a_pads, axis=0), axis=0)
    b_mean = np.mean(np.stack(b_pads, axis=0), axis=0)

    dA_dir = rng.normal(size=A_mean.shape)
    dA_dir = dA_dir / (np.linalg.norm(dA_dir) + 1e-12)
    db_dir = rng.normal(size=b_mean.shape)
    db_dir = db_dir / (np.linalg.norm(db_dir) + 1e-12)

    A_D_pad = A_mean + float(gap) * dA_dir
    b_D_pad = b_mean + float(gap) * db_dir
    A_D = A_D_pad[: int(test_d_x)].astype(np.float64)
    b_D = b_D_pad[: int(test_d_x)].astype(np.float64)

    test = TaskSpec(
        task_id=f"D_test_gap{float(gap):.3f}".replace(".", "p"),
        gap=float(gap),
        transform=InstanceTransform(
            A=torch.from_numpy(A_D).to(dtype=torch.float32),
            b=torch.from_numpy(b_D).to(dtype=torch.float32),
            obs_noise_std=float(obs_noise_std),
        ),
    )

    return TaskFamilySpec(
        objective=obj,
        d_pad=d_pad,
        train_tasks=tuple(train),
        test_task=test,
    )

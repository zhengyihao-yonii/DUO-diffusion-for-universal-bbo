from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import math

import numpy as np
import torch

from comparisonExperiment.experiment1.branin_standard import (
    branin_from_latent_coords,
    normalize_branin_domain,
)
from comparisonExperiment.experiment1.exp1_obs_noise import noise_std_per_dim


@dataclass(frozen=True)
class InstanceTransform:
    """Immutable affine instance map z -> x with per-dimension observation noise."""

    A: torch.Tensor  # [d_x, d_z]
    b: torch.Tensor  # [d_x]
    obs_noise_std: torch.Tensor  # [d_x]

    @staticmethod
    def from_scalar_noise(
        A: torch.Tensor,
        b: torch.Tensor,
        *,
        obs_noise_std: float = 0.0,
    ) -> InstanceTransform:
        dx = int(A.shape[0])
        std = torch.full((dx,), float(obs_noise_std), dtype=torch.float32)
        return InstanceTransform(A=A, b=b, obs_noise_std=std)

    @property
    def d_x(self) -> int:
        return int(self.A.shape[0])

    def map(self, z: torch.Tensor, *, rng: torch.Generator) -> torch.Tensor:
        x = z @ self.A.T + self.b
        std = self.obs_noise_std.to(device=x.device, dtype=x.dtype)
        if float(std.max().item()) > 0.0:
            eps = torch.randn_like(x, generator=rng) * std
            x = x + eps
        return x


BraninDomain = Literal["legacy", "standard"]


@dataclass(frozen=True)
class LatentObjective:
    """Shared latent objective f(z) (lower is better internally; we will convert to y_norm)."""

    name: Literal["branin", "ackley"]
    d_z: int
    branin_domain: BraninDomain = "legacy"

    def eval(self, z: torch.Tensor) -> torch.Tensor:
        if self.name == "branin":
            if int(self.d_z) != 2:
                raise ValueError("branin requires d_z=2")
            if str(self.branin_domain) == "standard":
                return branin_from_latent_coords(z)
            x1 = z[..., 0] * 5.0 + 2.5
            x2 = z[..., 1] * 7.5 + 7.5
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
    transform: InstanceTransform
    test_shift: str = ""
    f_bias: float = 0.0
    f_scale: float = 1.0

    @property
    def gap(self) -> float:
        """Legacy field for callers still reading ``gap``; always 0 (shift is in ``test_shift``)."""
        return 0.0

    def eval_raw_f(self, z: torch.Tensor, objective: LatentObjective) -> torch.Tensor:
        """Per-task Branin with small bias/scale; landscape plots ignore this perturbation."""
        base = objective.eval(z)
        return base * float(self.f_scale) + float(self.f_bias)


@dataclass(frozen=True)
class TaskFamilySpec:
    objective: LatentObjective
    d_pad: int
    train_tasks: tuple[TaskSpec, ...]
    test_task: TaskSpec
    test_shift: str = ""

    @property
    def max_d_x(self) -> int:
        return int(self.d_pad)

    def geometric_distance(self, *, mc: int = 4096, seed: int = 0) -> float:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        d_z = int(self.objective.d_z)
        z = torch.rand((int(mc), d_z), generator=gen) * 2.0 - 1.0
        d_pad = int(self.d_pad)

        def _pad(x: torch.Tensor) -> torch.Tensor:
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


def _task_from_params(
    task_id: str,
    p: object,
    *,
    test_shift: str = "",
) -> TaskSpec:
    from comparisonExperiment.experiment1.scene_metadata import TaskInstanceParams

    if isinstance(p, TaskInstanceParams):
        std = torch.from_numpy(np.asarray(p.obs_noise_std, dtype=np.float64)).to(torch.float32)
        return TaskSpec(
            task_id=task_id,
            test_shift=str(test_shift),
            transform=InstanceTransform(
                A=torch.from_numpy(p.A).to(dtype=torch.float32),
                b=torch.from_numpy(p.b).to(dtype=torch.float32),
                obs_noise_std=std,
            ),
            f_bias=float(p.f_bias),
            f_scale=float(p.f_scale),
        )
    raise TypeError(f"expected TaskInstanceParams, got {type(p)}")


def make_family(
    *,
    objective_name: str = "branin",
    d_z: int = 2,
    train_d_x: tuple[int, ...] = (5, 6, 7, 9, 10),
    test_d_x: int = 8,
    d_pad: int = 32,
    test_shift: str = "sim_low",
    gap: float | None = None,
    obs_noise_std: float = 0.0,
    obs_noise_base_std: float = 0.03,
    obs_noise_high_std: float = 0.40,
    obs_noise_high_dims: int = 1,
    seed: int = 0,
    branin_domain: BraninDomain = "legacy",
    family_mode: Literal["random_orth", "scene_aware"] = "random_orth",
    similarity_blend: float = 0.65,
    text_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> TaskFamilySpec:
    """Heterogeneous train/test dims; test instance selected by ``test_shift`` (not gap)."""
    if gap is not None and float(gap) > 1e-9:
        import warnings

        warnings.warn("make_family(gap=...) is deprecated; use test_shift=sim_low|sim_mid|sim_high", stacklevel=2)
    rng = np.random.default_rng(int(seed))
    on = str(objective_name).strip().lower()
    if on not in ("branin", "ackley"):
        raise ValueError(f"objective_name must be branin|ackley, got {objective_name!r}")
    bd: BraninDomain = (
        normalize_branin_domain(str(branin_domain))  # type: ignore[assignment]
        if on == "branin"
        else "legacy"
    )
    obj = LatentObjective(name=on, d_z=int(d_z), branin_domain=bd)  # type: ignore[arg-type]
    if len(train_d_x) < 1:
        raise ValueError("train_d_x must be non-empty")
    if int(test_d_x) < 1:
        raise ValueError("test_d_x must be >= 1")
    d_pad = int(d_pad)
    d_nat_max = int(max(max(train_d_x), int(test_d_x)))
    if d_pad < d_nat_max:
        raise ValueError(f"d_pad={d_pad} must be >= max native observation dim ({d_nat_max})")

    ts = str(test_shift).strip().lower()
    fm = str(family_mode).strip().lower()
    if fm == "scene_aware":
        from comparisonExperiment.experiment1.scene_metadata import build_scene_correlated_family

        params, _scenarios, _sim = build_scene_correlated_family(
            tuple(int(x) for x in train_d_x),
            int(test_d_x),
            test_shift=ts,
            seed=int(seed),
            similarity_blend=float(similarity_blend),
            obs_noise_base_std=float(obs_noise_base_std),
            obs_noise_high_std=float(obs_noise_high_std),
            obs_noise_high_dims=int(obs_noise_high_dims),
            model_name=str(text_encoder_model),
        )
        train = tuple(
            _task_from_params(f"D_train_{i + 1}", params[f"D_train_{i + 1}"])
            for i in range(len(train_d_x))
        )
        from comparisonExperiment.experiment1.scene_metadata import test_scenario_for_shift

        test_sc = test_scenario_for_shift(ts)
        test = _task_from_params(test_sc.task_id, params[test_sc.task_id], test_shift=ts)
        return TaskFamilySpec(
            objective=obj,
            d_pad=d_pad,
            train_tasks=train,
            test_task=test,
            test_shift=ts,
        )

    use_vec_noise = float(obs_noise_std) <= 0.0 and float(obs_noise_base_std) > 0.0
    train: list[TaskSpec] = []
    a_pads: list[np.ndarray] = []
    b_pads: list[np.ndarray] = []
    for i, d_i in enumerate(train_d_x):
        name = f"D_train_{i + 1}"
        Ai = _rand_orthonormal(int(d_i), int(d_z), rng=rng)
        bi = rng.normal(scale=0.2, size=(int(d_i),)).astype(np.float64)
        if use_vec_noise:
            nstd = noise_std_per_dim(
                int(d_i),
                rng=rng,
                base_std=float(obs_noise_base_std),
                n_high_noise_dims=int(obs_noise_high_dims),
                high_std=float(obs_noise_high_std),
            )
            tr = InstanceTransform(
                A=torch.from_numpy(Ai).to(dtype=torch.float32),
                b=torch.from_numpy(bi).to(dtype=torch.float32),
                obs_noise_std=torch.from_numpy(nstd).to(dtype=torch.float32),
            )
        else:
            tr = InstanceTransform.from_scalar_noise(
                torch.from_numpy(Ai).to(dtype=torch.float32),
                torch.from_numpy(bi).to(dtype=torch.float32),
                obs_noise_std=float(obs_noise_std),
            )
        train.append(
            TaskSpec(
                task_id=name,
                transform=tr,
                f_bias=float(rng.uniform(-1.5, 1.5)),
                f_scale=float(1.0 + rng.uniform(-0.012, 0.012)),
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
    A_D = A_mean[: int(test_d_x)].astype(np.float64)
    b_D = b_mean[: int(test_d_x)].astype(np.float64)

    if use_vec_noise:
        nstd = noise_std_per_dim(
            int(test_d_x),
            rng=rng,
            base_std=float(obs_noise_base_std),
            n_high_noise_dims=int(obs_noise_high_dims),
            high_std=float(obs_noise_high_std),
        )
        tr_test = InstanceTransform(
            A=torch.from_numpy(A_D).to(dtype=torch.float32),
            b=torch.from_numpy(b_D).to(dtype=torch.float32),
            obs_noise_std=torch.from_numpy(nstd).to(dtype=torch.float32),
        )
    else:
        tr_test = InstanceTransform.from_scalar_noise(
            torch.from_numpy(A_D).to(dtype=torch.float32),
            torch.from_numpy(b_D).to(dtype=torch.float32),
            obs_noise_std=float(obs_noise_std),
        )

    from comparisonExperiment.experiment1.scene_metadata import test_scenario_for_shift

    test_sc = test_scenario_for_shift(ts)
    test = TaskSpec(
        task_id=test_sc.task_id,
        test_shift=ts,
        transform=tr_test,
        f_bias=float(rng.uniform(-1.5, 1.5)),
        f_scale=float(1.0 + rng.uniform(-0.012, 0.012)),
    )

    return TaskFamilySpec(
        objective=obj,
        d_pad=d_pad,
        train_tasks=tuple(train),
        test_task=test,
        test_shift=ts,
    )

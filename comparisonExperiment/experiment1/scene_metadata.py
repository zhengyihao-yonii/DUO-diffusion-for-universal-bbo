# -*- coding: utf-8 -*-
"""
Scenario metadata + scene-correlated affine maps for Quality exp1 (v3).

English doc: Five train tasks with mixed domains (auto pair + chem/materials + aerospace).
Three held-out test tasks use ``sim_low`` / ``sim_mid`` / ``sim_high`` metadata similarity
to training (replacing the old gap coefficient). MiniLM similarity drives A blending only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from comparisonExperiment.experiment1.exp1_obs_noise import noise_std_per_dim

Role = Literal["train", "test"]
TestShift = Literal["sim_low", "sim_mid", "sim_high"]

VALID_TEST_SHIFTS: tuple[str, ...] = ("sim_low", "sim_mid", "sim_high")


@dataclass(frozen=True)
class TaskScenario:
    """One synthetic benchmark instance with a distinct application narrative."""

    task_id: str
    role: Role
    title: str
    domain: str
    description: str
    objective: str
    test_shift: str = ""
    related_train_ids: tuple[str, ...] = ()

    def similarity_text(self) -> str:
        """Text for MiniLM similarity (no shared boilerplate)."""
        return f"{self.title}. {self.description}"

    def metadata_text(self, *, d_x: int | None = None) -> str:
        """Five-line UniSO / main-benchmark layout."""
        _ = d_x
        return "\n".join(
            [
                f"Task: {self.task_id}",
                "Type: synthetic",
                f"Name: {self.title}",
                f"Description: {self.description}",
                f"Objective: {self.objective}",
            ]
        )


def default_train_scenarios() -> tuple[TaskScenario, ...]:
    """
    Five training tasks with structured metadata similarity (not all one domain).

    English doc: (1,2) automotive pair (high sim); (3,4) chemical vs materials (moderate);
    (5) aerospace (moderate to automotive engineering tone, low to chem/materials).
    """
    return (
        TaskScenario(
            task_id="D_train_1",
            role="train",
            title="Compact sedan gasoline ICE calibration",
            domain="automotive powertrain",
            description=(
                "Calibrate spark timing and wastegate duty on a 1.5 L turbocharged sedan engine "
                "for peak torque before emissions derate on a standard WLTC cycle."
            ),
            objective="Maximize sedan gasoline cruise fuel economy.",
        ),
        TaskScenario(
            task_id="D_train_2",
            role="train",
            title="Compact sedan hybrid power-split tuning",
            domain="automotive powertrain",
            description=(
                "Same compact sedan platform with a power-split hybrid: co-optimize engine on/off "
                "thresholds and battery discharge power during urban segments of WLTC."
            ),
            objective="Minimize sedan hybrid energy consumption on WLTC.",
        ),
        TaskScenario(
            task_id="D_train_3",
            role="train",
            title="Chemical process yield tuning",
            domain="process systems",
            description=(
                "Pilot-plant formulation and operating-point study for a specialty chemical blend: "
                "tune feed temperature and residence time to maximize batch yield."
            ),
            objective="Maximize steady-state chemical reactor yield.",
        ),
        TaskScenario(
            task_id="D_train_4",
            role="train",
            title="Polymer formulation screening",
            domain="materials science",
            description=(
                "Bench-scale formulation screening for thermoset polymer blends: optimize curing "
                "recipe and additive levels to maximize mechanical properties after aging."
            ),
            objective="Maximize aged tensile strength of polymer dogbone specimens.",
        ),
        TaskScenario(
            task_id="D_train_5",
            role="train",
            title="Aircraft wing aerodynamic design",
            domain="aerospace structures",
            description=(
                "Wind-tunnel wing design optimization: schedule camber and high-lift device "
                "settings to improve cruise efficiency, analogous to vehicle duty-cycle tuning."
            ),
            objective="Maximize cruise lift-to-drag ratio in the tunnel surrogate.",
        ),
    )


def test_scenario_for_shift(test_shift: str) -> TaskScenario:
    """Held-out test scenario; metadata similarity to D_train increases sim_low → sim_high."""
    sh = str(test_shift).strip().lower()
    if sh not in VALID_TEST_SHIFTS:
        raise ValueError(f"test_shift must be one of {VALID_TEST_SHIFTS}, got {test_shift!r}")
    if sh == "sim_low":
        return TaskScenario(
            task_id="D_test_sim_low",
            role="test",
            title="Semiconductor wafer CMP slurry formulation",
            domain="semiconductor manufacturing",
            description=(
                "Optimize abrasive slurry pH and oxidizer concentration for copper CMP on "
                "300 mm wafers to minimize dishing without throughput loss."
            ),
            objective="Minimize post-CMP wafer dishing depth.",
            test_shift=sh,
            related_train_ids=(),
        )
    if sh == "sim_mid":
        return TaskScenario(
            task_id="D_test_sim_mid",
            role="test",
            title="Automotive HVAC compressor map extension",
            domain="automotive thermal",
            description=(
                "Recalibrate electric compressor speed and superheat targets after a new "
                "refrigerant blend is introduced on a shared passenger-vehicle thermal rig."
            ),
            objective="Minimize cabin pull-down time under ISO 7730 comfort bounds.",
            test_shift=sh,
            related_train_ids=("D_train_1", "D_train_3"),
        )
    return TaskScenario(
        task_id="D_test_sim_high",
        role="test",
        title="Compact sedan gasoline cold-start calibration",
        domain="automotive powertrain",
        description=(
            "Cold-start enrichment and catalyst light-off strategy on the same 1.5 L turbo "
            "sedan gasoline engine family used in compact-sedan ICE training, new winter fuel."
        ),
        objective="Minimize sedan cold-start HC emissions during FTP warm-up.",
        test_shift=sh,
        related_train_ids=("D_train_1", "D_train_2"),
    )


def embed_scenarios_minilm(
    scenarios: tuple[TaskScenario, ...],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """Return L2-normalized embeddings [N, E] for ``similarity_text()``."""
    from sentence_transformers import SentenceTransformer

    texts = [s.similarity_text() for s in scenarios]
    model = SentenceTransformer(str(model_name))
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-12)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    e = np.asarray(embeddings, dtype=np.float64)
    return e @ e.T


@dataclass(frozen=True)
class TaskInstanceParams:
    """Per-task affine map, noise, and small objective perturbation (landscape uses canonical Branin)."""

    A: np.ndarray
    b: np.ndarray
    obs_noise_std: np.ndarray
    f_bias: float
    f_scale: float


def _random_instance_Ab(
    d_x: int, d_z: int, *, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    m = rng.normal(size=(int(d_x), int(d_z))).astype(np.float64)
    q, _r = np.linalg.qr(m)
    a = q[:, : int(d_z)]
    b = rng.normal(scale=0.2, size=(int(d_x),)).astype(np.float64)
    return a, b


def _small_objective_perturbation(
    task_id: str, *, rng: np.random.Generator
) -> tuple[float, float]:
    """English doc: Tiny per-task scale/bias; canonical Branin used for landscape either way."""
    seed_i = int(rng.integers(0, 2**31)) ^ (hash(task_id) & 0xFFFFFFFF)
    sub = np.random.default_rng(seed_i)
    f_scale = 1.0 + float(sub.uniform(-0.012, 0.012))
    f_bias = float(sub.uniform(-1.5, 1.5))
    return f_bias, f_scale


def build_scene_correlated_family(
    train_d_x: tuple[int, ...],
    test_d_x: int,
    *,
    test_shift: str,
    seed: int,
    similarity_blend: float = 0.65,
    obs_noise_base_std: float = 0.03,
    obs_noise_high_std: float = 0.40,
    obs_noise_high_dims: int = 1,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[dict[str, TaskInstanceParams], dict[str, TaskScenario], np.ndarray]:
    """
    English doc: Returns task params, scenarios, full cosine similarity matrix.

    A blending: ``A_j <- (1-w) A_j + w A_i`` with ``w = sim(i,j)^2 * similarity_blend``.
    Test A blends toward all trains weighted by metadata similarity (no gap shift).
    """
    rng = np.random.default_rng(int(seed))
    trains_all = default_train_scenarios()
    nt = len(train_d_x)
    if nt > len(trains_all):
        raise ValueError(f"scene_aware supports at most {len(trains_all)} train tasks, got {nt}")
    trains = trains_all[:nt]
    test = test_scenario_for_shift(str(test_shift))
    all_scenarios = trains + (test,)
    emb = embed_scenarios_minilm(all_scenarios, model_name=str(model_name))
    sim = cosine_similarity_matrix(emb)

    d_pad = int(max(max(train_d_x), int(test_d_x)))
    params: dict[str, TaskInstanceParams] = {}
    a_rand_pad: list[np.ndarray] = []
    b_rand_pad: list[np.ndarray] = []

    for d_i in train_d_x:
        ai, bi = _random_instance_Ab(int(d_i), 2, rng=rng)
        ap = np.zeros((d_pad, 2), dtype=np.float64)
        ap[: int(d_i), :] = ai
        bp = np.zeros((d_pad,), dtype=np.float64)
        bp[: int(d_i)] = bi
        a_rand_pad.append(ap)
        b_rand_pad.append(bp)

    a_blend_pad: list[np.ndarray] = []
    b_blend_pad: list[np.ndarray] = []
    for j, sc in enumerate(trains):
        aj = a_rand_pad[j].copy()
        bj = b_rand_pad[j].copy()
        for i in range(len(trains)):
            if i == j:
                continue
            w = float(sim[i, j] ** 2) * float(similarity_blend)
            if w > 1e-6:
                aj = (1.0 - w) * aj + w * a_rand_pad[i]
                bj = (1.0 - w) * bj + w * b_rand_pad[i]
        a_blend_pad.append(aj)
        b_blend_pad.append(bj)
        d_j = int(train_d_x[j])
        fb, fs = _small_objective_perturbation(sc.task_id, rng=rng)
        nstd = noise_std_per_dim(
            d_j,
            rng=rng,
            base_std=float(obs_noise_base_std),
            n_high_noise_dims=int(obs_noise_high_dims),
            high_std=float(obs_noise_high_std),
        )
        params[sc.task_id] = TaskInstanceParams(
            A=aj[:d_j, :].astype(np.float64),
            b=bj[:d_j].astype(np.float64),
            obs_noise_std=nstd,
            f_bias=fb,
            f_scale=fs,
        )

    n_tr = len(trains)
    test_idx = n_tr
    aj = np.zeros((d_pad, 2), dtype=np.float64)
    bj = np.zeros((d_pad,), dtype=np.float64)
    ai0, bi0 = _random_instance_Ab(int(test_d_x), 2, rng=rng)
    aj[: int(test_d_x), :] = ai0
    bj[: int(test_d_x)] = bi0
    for i in range(n_tr):
        w = float(sim[i, test_idx] ** 2) * float(similarity_blend)
        if w > 1e-6:
            aj = (1.0 - w) * aj + w * a_blend_pad[i]
            bj = (1.0 - w) * bj + w * b_blend_pad[i]
    d_te = int(test_d_x)
    fb, fs = _small_objective_perturbation(test.task_id, rng=rng)
    nstd = noise_std_per_dim(
        d_te,
        rng=rng,
        base_std=float(obs_noise_base_std),
        n_high_noise_dims=int(obs_noise_high_dims),
        high_std=float(obs_noise_high_std),
    )
    params[test.task_id] = TaskInstanceParams(
        A=aj[:d_te, :].astype(np.float64),
        b=bj[:d_te].astype(np.float64),
        obs_noise_std=nstd,
        f_bias=fb,
        f_scale=fs,
    )

    scenario_map = {s.task_id: s for s in all_scenarios}
    return params, scenario_map, sim


# Backward-compatible alias
def build_scene_correlated_transforms(
    train_d_x: tuple[int, ...],
    test_d_x: int,
    *,
    gap: float = 0.0,
    seed: int,
    similarity_blend: float = 0.65,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, TaskScenario], np.ndarray]:
    """Deprecated: map legacy gap to test_shift."""
    _ = float(gap)
    shift = VALID_TEST_SHIFTS[0]
    params, scenarios, sim = build_scene_correlated_family(
        train_d_x,
        test_d_x,
        test_shift=shift,
        seed=seed,
        similarity_blend=similarity_blend,
        model_name=model_name,
    )
    ab = {k: (v.A, v.b) for k, v in params.items()}
    return ab, scenarios, sim

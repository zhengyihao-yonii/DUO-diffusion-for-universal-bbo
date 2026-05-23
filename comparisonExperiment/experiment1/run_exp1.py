from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

# Chinese comment: 直接 python path/to/run_exp1.py 时保证可导入 comparisonExperiment.*
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from comparisonExperiment.experiment1.data_formats import (
    DuoTrajectoryPkl,
    save_duo_single_task_pkl,
    save_uniso_jsonl,
)
from comparisonExperiment.experiment1.exp1_vae import (
    fit_vae_on_points,
    native_traj_to_latent_padded,
    pad_x_to_d_pad,
    save_vae,
)
from comparisonExperiment.experiment1.branin_standard import normalize_branin_domain
from comparisonExperiment.experiment1.task_family import TaskFamilySpec, make_family


def _parse_int_tuple(s: str) -> tuple[int, ...]:
    parts = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not parts:
        raise ValueError("empty int list")
    return tuple(parts)


@dataclass(frozen=True)
class Exp1Config:
    out_root: Path
    objective: str = "branin"
    d_z: int = 2
    # Heterogeneous native design dims per train task + test task (UniSO / oracle use native x).
    train_d_x: tuple[int, ...] = (5, 6, 7, 9, 10)
    test_d_x: int = 8
    # Shared diffusion trajectory width (VAE mu in R^{latent_dim}), aligned with main DUO spirit.
    latent_dim: int = 8
    # Zero-pad native x to R^{d_pad} before shared VAE (default 32, main-experiment style).
    d_pad: int = 32
    vae_train_steps: int = 800
    obs_noise_std: float = 0.0
    # UniSO trains on points; we control points per task here (target ~2000).
    n_points: int = 2000
    # DUO consumes trajectories; we build 100x32 with replacement from the point pool.
    horizon: int = 64
    duo_n_traj: int = 100
    seed: int = 0
    # Held-out test scenario key: sim_low | sim_mid | sim_high (replaces legacy gap).
    test_shift: str = "sim_low"
    pkl_suffix: str = ""
    obs_noise_base_std: float = 0.03
    obs_noise_high_std: float = 0.40
    obs_noise_high_dims: int = 1
    # ``standard``: usual Branin x1∈[-5,10], x2∈[0,15]; ``legacy``: pre-v3 z scaling.
    branin_domain: str = "legacy"
    # ``scene_aware``: metadata scenarios + MiniLM-correlated A; ``random_orth``: v1/v2.
    family_mode: str = "random_orth"
    similarity_blend: float = 0.65
    text_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Where to write UniSO text datasets
    uniso_data_dir: Path = Path("../UniSO/data")
    # Few-shot: sample ``fewshot_budget`` points uniformly from the lowest-``tail_frac`` y fraction (worst designs).
    fewshot_budget: int = 100
    fewshot_tail_fracs: str = "0.1,0.2,0.5"  # comma-separated; one PKL + UniSO bundle per tail
    # D_train only: keep the best ``train_pool_best_frac`` fraction of points by y (higher y = better), then sample trajs.
    train_pool_best_frac: float = 0.9


def _gap_shift_qualifier_zh(gap: float) -> str:
    # 中文注释: 测试任务 gap 在描述里的中文程度副词（与主实验元信息风格分离字段）。
    """English doc: Qualitative label for test-instance shift (Chinese, embedded in metadata text)."""
    g = float(gap)
    if g <= 1e-9:
        return "几乎无"
    if g < 0.35:
        return "较小"
    return "较大"


def _synthetic_family_display_name(objective: str) -> str:
    on = str(objective).strip().lower()
    if on == "ackley":
        return "Ackley-class"
    return "Branin-class"


def _exp1_metadata_text(
    *,
    cfg: Exp1Config,
    task_id: str,
    is_train_domain: bool,
    instance_gap: float,
) -> str:
    """English doc: UniSO ``metadata_text`` aligned with main benchmark field layout (Task / Type / Name / Description / Objective)."""
    fam = _synthetic_family_display_name(cfg.objective)
    dz = int(cfg.d_z)
    lines: list[str] = [
        f"Task: {task_id}",
        "Type: synthetic",
    ]
    if is_train_domain:
        lines.append(f"Name: {fam} latent benchmark (train domain)")
        desc = (
            "Description: A synthetic expensive black-box simulator using the same metadata fields as the main "
            "real-task experiments. "
            f"The latent objective lives in [-1,1]^{dz} and is a smooth {fam.lower()} landscape in z; "
            "each instance maps z to a native design x through an affine map x = A z + b with random instance "
            "parameters, and exposes a min-max normalized score y in [0,1] (higher is better). "
            "Instances differ in how the latent is embedded into native design space; the trajectory model uses "
            "zero-padded native vectors and a shared encoder across tasks. "
            "This instance is a training-domain draw without a scripted held-out shift."
        )
    else:
        lines.append(f"Name: {fam} latent benchmark (test domain)")
        qual = _gap_shift_qualifier_zh(instance_gap)
        desc = (
            "Description: Same synthetic family and field layout as the training-domain tasks. "
            "This row is the held-out test instance. "
            f"Train-test instance shift (gap coefficient g): {float(instance_gap):.3f}（{qual}）."
        )
    lines.append(desc)
    lines.append("Objective: Maximize y.")
    return "\n".join(lines)


def _fewshot_metadata_suffix(*, tail_frac: float, tail_tag: str, budget: int) -> str:
    """English doc: Appended to test-task metadata for few-shot point pools."""
    return (
        "\n"
        "Supplement (few-shot): points are sampled uniformly from the lowest "
        f"{float(tail_frac):.3f} fraction of y (worst designs); budget K={int(budget)}; "
        f"regime_tag={tail_tag}. "
        "y remains min-max normalized with larger y better."
    )


def _minmax_01(y: torch.Tensor) -> torch.Tensor:
    lo = torch.min(y)
    hi = torch.max(y)
    if float((hi - lo).abs().item()) < 1e-12:
        return torch.zeros_like(y)
    return (y - lo) / (hi - lo)


def _make_point_pool(
    fam: TaskFamilySpec,
    *,
    task_id: str,
    n_points: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float, torch.Tensor]:
    """Generate iid points (x,y) for a task instance.

    Chinese comment: UniSO 训练输入是点 + metadata，这里只生成点，不额外“造轨迹”。
    Also returns raw latent objective ``f`` (lower is better) per point for ``dbest`` / filtering.
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1000 + hash(task_id) % 997)
    d_z = int(fam.objective.d_z)
    z = torch.rand((int(n_points), d_z), generator=gen) * 2.0 - 1.0
    ts = next(t for t in (list(fam.train_tasks) + [fam.test_task]) if t.task_id == task_id)
    x = ts.transform.map(z, rng=gen).to(torch.float32)
    f = ts.eval_raw_f(z, fam.objective).to(torch.float32)
    f_lo = float(torch.min(f).item())
    f_hi = float(torch.max(f).item())
    y = 1.0 - _minmax_01(f)  # y_norm in [0,1], higher better
    return x, y, f_lo, f_hi, f.detach()


def _restrict_pool_best_y_frac(
    xs: torch.Tensor,
    ys: torch.Tensor,
    f_raw: torch.Tensor,
    *,
    best_frac: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """English doc: Keep the best ``best_frac`` fraction of points by y (higher = better)."""
    bf = float(best_frac)
    if bf >= 1.0 - 1e-12:
        return xs, ys, f_raw
    n = int(ys.numel())
    k = max(1, int(math.ceil(bf * n)))
    order = torch.argsort(ys, descending=True)
    pick = order[:k]
    return xs[pick], ys[pick], f_raw[pick]


def _select_fewshot_points(
    *,
    xs: torch.Tensor,  # [N, d_x]
    ys: torch.Tensor,  # [N]
    y_max: float,
    budget_frac: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy: among ys <= y_max, sample up to floor(budget_frac*N)."""
    N = int(ys.numel())
    b = int(max(1, math.floor(float(budget_frac) * float(N))))
    mask = ys <= float(y_max)
    idx = torch.nonzero(mask, as_tuple=False).view(-1)
    if idx.numel() == 0:
        idx = torch.arange(N)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = idx[torch.randperm(idx.numel(), generator=gen)]
    take = perm[: min(int(perm.numel()), b)]
    return xs[take], ys[take]


def _tail_tag_for_frac(tail_frac: float) -> str:
    """English doc: e.g. 0.1 -> tail10p for stable filenames."""
    pct = int(round(float(tail_frac) * 100.0))
    return f"tail{pct}p"


def _select_fewshot_points_worst_tail(
    *,
    xs: torch.Tensor,
    ys: torch.Tensor,
    tail_frac: float,
    budget: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    English doc: ``ys`` higher is better (exp1 convention). Pool = lowest ``tail_frac`` fraction of ys;
    uniformly sample up to ``budget`` points with a deterministic RNG (seed + tail tag).

    中文注释: 从最差 tail_frac（按 y 升序尾部）中随机取 budget 个点；每个 tail 用独立子种子保证可复现。
    """
    N = int(ys.numel())
    if N < 1:
        raise ValueError("empty few-shot pool")
    tf = float(tail_frac)
    tf = max(1.0 / float(N), min(1.0, tf))
    n_tail = max(1, int(math.ceil(tf * float(N))))
    order = torch.argsort(ys, descending=False)
    idx_tail = order[:n_tail]
    gen = torch.Generator(device="cpu").manual_seed(
        int(seed) + int(round(tf * 10000.0)) + n_tail * 17
    )
    perm = idx_tail[torch.randperm(idx_tail.numel(), generator=gen)]
    b = int(max(1, min(int(budget), int(perm.numel()))))
    take = perm[:b]
    return xs[take], ys[take]


def _points_to_trajs(
    *,
    xs: torch.Tensor,  # [M, d_x]
    ys: torch.Tensor,  # [M]
    horizon: int,
    n_traj: int,
    seed: int,
) -> DuoTrajectoryPkl:
    """Build DUO-style trajectories by sampling points with replacement."""
    h = int(horizon)
    nt = int(n_traj)
    if nt < 1:
        raise ValueError("n_traj must be >= 1")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    M = int(ys.numel())
    if M <= 0:
        raise ValueError("Empty point pool")
    need = nt * h
    idx = torch.randint(low=0, high=M, size=(need,), generator=gen)
    x = xs[idx].reshape(nt, h, xs.shape[-1])
    y = ys[idx].reshape(nt, h)
    pr = 1.0 - y
    rtg = torch.flip(torch.cumsum(torch.flip(pr, dims=[1]), dim=1), dims=[1])
    ts = torch.arange(h).repeat(nt, 1)
    return DuoTrajectoryPkl(points=x, values=y, pr=pr, rtg=rtg, timesteps=ts)


def _vae_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _latent_pkl_name(task_id: str, *, horizon: int, n_traj: int, latent_dim: int) -> str:
    return f"{task_id}_h{int(horizon)}_n{int(n_traj)}_lat{int(latent_dim)}.pkl"


def _write_sidecar(
    pkl_path: Path, *, d_native: int, d_pad: int, latent_dim: int, vae_filename: str
) -> None:
    side = {
        "d_x_native": int(d_native),
        "d_pad": int(d_pad),
        "latent_dim": int(latent_dim),
        "vae_filename": str(vae_filename),
    }
    (pkl_path.parent / f"{pkl_path.name}.exp1_sidecar.json").write_text(
        json.dumps(side, indent=2), encoding="utf-8"
    )


def _write_one(cfg: Exp1Config) -> None:
    out_root = cfg.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    fam = make_family(
        objective_name=str(cfg.objective),
        d_z=int(cfg.d_z),
        train_d_x=tuple(int(x) for x in cfg.train_d_x),
        test_d_x=int(cfg.test_d_x),
        d_pad=int(cfg.d_pad),
        test_shift=str(cfg.test_shift),
        obs_noise_std=float(cfg.obs_noise_std),
        obs_noise_base_std=float(cfg.obs_noise_base_std),
        obs_noise_high_std=float(cfg.obs_noise_high_std),
        obs_noise_high_dims=int(cfg.obs_noise_high_dims),
        seed=int(cfg.seed),
        branin_domain=normalize_branin_domain(str(cfg.branin_domain)),  # type: ignore[arg-type]
        family_mode=str(cfg.family_mode),  # type: ignore[arg-type]
        similarity_blend=float(cfg.similarity_blend),
        text_encoder_model=str(cfg.text_encoder_model),
    )
    dist = fam.geometric_distance(seed=cfg.seed)

    scene_sim: list[list[float]] | None = None
    if str(cfg.family_mode).strip().lower() == "scene_aware":
        from comparisonExperiment.experiment1.scene_metadata import build_scene_correlated_family

        _params, _sc, sim_mat = build_scene_correlated_family(
            tuple(int(x) for x in cfg.train_d_x),
            int(cfg.test_d_x),
            test_shift=str(cfg.test_shift),
            seed=int(cfg.seed),
            similarity_blend=float(cfg.similarity_blend),
            obs_noise_base_std=float(cfg.obs_noise_base_std),
            obs_noise_high_std=float(cfg.obs_noise_high_std),
            obs_noise_high_dims=int(cfg.obs_noise_high_dims),
            model_name=str(cfg.text_encoder_model),
        )
        scene_sim = sim_mat.tolist()

    rec = {
        "objective": cfg.objective,
        "d_z": cfg.d_z,
        "branin_domain": normalize_branin_domain(str(cfg.branin_domain)),
        "family_mode": str(cfg.family_mode),
        "similarity_blend": float(cfg.similarity_blend),
        "scene_metadata_similarity": scene_sim,
        "test_shift": str(cfg.test_shift),
        "train_d_x": list(cfg.train_d_x),
        "test_d_x": int(cfg.test_d_x),
        "d_pad": int(fam.d_pad),
        "latent_dim": int(cfg.latent_dim),
        "geom_dist": dist,
        "obs_noise_base_std": float(cfg.obs_noise_base_std),
        "obs_noise_high_std": float(cfg.obs_noise_high_std),
        "obs_noise_high_dims": int(cfg.obs_noise_high_dims),
        "n_points_per_task": int(cfg.n_points),
        "duo": {"horizon": int(cfg.horizon), "n_traj": int(cfg.duo_n_traj)},
        "seed": cfg.seed,
        "train_tasks": [t.task_id for t in fam.train_tasks],
        "test_task": fam.test_task.task_id,
        "fewshot": {
            "mode": "worst_tail_frac",
            "budget": int(cfg.fewshot_budget),
            "tail_fracs": str(cfg.fewshot_tail_fracs),
        },
        "train_pool_best_frac": float(cfg.train_pool_best_frac),
    }
    (out_root / "family_meta.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    _shift_tag = f"{str(cfg.test_shift).strip()}{str(cfg.pkl_suffix)}"
    duo_ds_root = Path("generated_datasets") / f"exp1_{_shift_tag}"
    duo_ds_root.mkdir(parents=True, exist_ok=True)
    dev = _vae_device()
    duo_train_merge_latent: list[DuoTrajectoryPkl] = []
    d_pad = int(fam.d_pad)
    uniso_data_dir = cfg.uniso_data_dir.resolve()
    uniso_data_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for t in list(fam.train_tasks) + [fam.test_task]:
        d_nat = int(t.transform.d_x)
        x_pool, y_pool, f_lo, f_hi, f_raw = _make_point_pool(
            fam, task_id=t.task_id, n_points=int(cfg.n_points), seed=int(cfg.seed)
        )
        is_train_task = str(t.task_id).startswith("D_train_")
        if is_train_task and float(cfg.train_pool_best_frac) < 1.0:
            x_pool, y_pool, f_raw = _restrict_pool_best_y_frac(
                x_pool,
                y_pool,
                f_raw,
                best_frac=float(cfg.train_pool_best_frac),
            )
        dbest = float(torch.min(f_raw).item())
        xs = x_pool.detach().cpu().numpy()
        ys = y_pool.detach().cpu().numpy()
        if str(cfg.family_mode).strip().lower() == "scene_aware":
            from comparisonExperiment.experiment1.scene_metadata import (
                default_train_scenarios,
                test_scenario_for_shift,
            )

            trains = default_train_scenarios()
            test_sc = test_scenario_for_shift(str(cfg.test_shift))
            sc_by_id = {s.task_id: s for s in trains + (test_sc,)}
            sc = sc_by_id.get(str(t.task_id))
            meta = (
                sc.metadata_text(d_x=d_nat)
                if sc is not None
                else _exp1_metadata_text(
                    cfg=cfg,
                    task_id=str(t.task_id),
                    is_train_domain=bool(is_train_task),
                    instance_gap=0.0,
                )
            )
        else:
            meta = _exp1_metadata_text(
                cfg=cfg,
                task_id=str(t.task_id),
                is_train_domain=bool(is_train_task),
                instance_gap=0.0,
            )
        nstd = t.transform.obs_noise_std.detach().cpu().numpy()
        meta_struct = {
            "exp": "comparison1_exp1",
            "task_id": t.task_id,
            "objective": cfg.objective,
            "branin_domain": normalize_branin_domain(str(cfg.branin_domain)),
            "family_mode": str(cfg.family_mode),
            "test_shift": str(t.test_shift) if not is_train_task else "",
            "d_z": cfg.d_z,
            "d_x": d_nat,
            "d_pad": d_pad,
            "latent_dim": int(cfg.latent_dim),
            "geom_dist": float(dist),
            "f_min": float(f_lo),
            "f_max": float(f_hi),
            "f_bias": float(t.f_bias),
            "f_scale": float(t.f_scale),
            "A": t.transform.A.detach().cpu().numpy().tolist(),
            "b": t.transform.b.detach().cpu().numpy().tolist(),
            "obs_noise_std": nstd.tolist(),
            "metadata_text": meta,
            "dbest": float(dbest),
            "train_pool_best_frac": float(cfg.train_pool_best_frac)
            if is_train_task
            else None,
            "n_points_after_pool_filter": int(x_pool.shape[0]),
        }
        save_uniso_jsonl(
            json_path=uniso_data_dir / f"exp1_{t.task_id}.json",
            metadata_path=uniso_data_dir / f"exp1_{t.task_id}.metadata",
            xs=xs,
            ys=ys,
            metadata_text=meta,
        )
        (uniso_data_dir / f"exp1_{t.task_id}.meta.json").write_text(
            json.dumps(meta_struct, indent=2), encoding="utf-8"
        )
        native_obj = _points_to_trajs(
            xs=x_pool,
            ys=y_pool,
            horizon=int(cfg.horizon),
            n_traj=int(cfg.duo_n_traj),
            seed=int(cfg.seed),
        )
        rows.append(
            {
                "t": t,
                "d_nat": d_nat,
                "x_pool": x_pool,
                "y_pool": y_pool,
                "meta": meta,
                "meta_struct": meta_struct,
                "native_obj": native_obj,
            }
        )

    x_pad_blocks = [
        pad_x_to_d_pad(r["x_pool"], d_pad=d_pad).reshape(-1, d_pad)  # type: ignore[arg-type]
        for r in rows
    ]
    x_cat = torch.cat(x_pad_blocks, dim=0)
    vae_name = "vae_shared.pt"
    vae_path = duo_ds_root / vae_name
    vae = fit_vae_on_points(
        x_cat,
        input_dim=d_pad,
        latent_dim=int(cfg.latent_dim),
        train_steps=int(cfg.vae_train_steps),
        device=dev,
    )
    save_vae(vae_path, vae)

    for r in rows:
        t = r["t"]  # type: ignore[assignment]
        d_nat = int(r["d_nat"])  # type: ignore[arg-type]
        native_obj = r["native_obj"]  # type: ignore[assignment]
        meta = str(r["meta"])  # type: ignore[arg-type]
        lat_obj = native_traj_to_latent_padded(
            vae, native_obj, d_pad=d_pad, device=dev  # type: ignore[arg-type]
        )
        pkl_n = _latent_pkl_name(t.task_id, horizon=cfg.horizon, n_traj=cfg.duo_n_traj, latent_dim=cfg.latent_dim)
        pkl_path = duo_ds_root / pkl_n
        save_duo_single_task_pkl(pkl_path, lat_obj)
        _write_sidecar(
            pkl_path,
            d_native=d_nat,
            d_pad=d_pad,
            latent_dim=int(cfg.latent_dim),
            vae_filename=vae_name,
        )
        if t.task_id.startswith("D_train_"):
            duo_train_merge_latent.append(lat_obj)

        if str(t.task_id).startswith("D_test_"):
            x_pool = r["x_pool"]  # type: ignore[assignment]
            xs_t = x_pool.to(torch.float32)
            ys_t = r["y_pool"].to(torch.float32)  # type: ignore[union-attr]
            tail_parts = [x.strip() for x in str(cfg.fewshot_tail_fracs).split(",") if x.strip()]
            for tf_str in tail_parts:
                tail_frac = float(tf_str)
                tail_tag = _tail_tag_for_frac(tail_frac)
                fs_x, fs_y = _select_fewshot_points_worst_tail(
                    xs=xs_t,
                    ys=ys_t,
                    tail_frac=tail_frac,
                    budget=int(cfg.fewshot_budget),
                    seed=int(cfg.seed),
                )
                fs_meta_text = meta + _fewshot_metadata_suffix(
                    tail_frac=float(tail_frac),
                    tail_tag=str(tail_tag),
                    budget=int(cfg.fewshot_budget),
                )
                save_uniso_jsonl(
                    json_path=uniso_data_dir / f"exp1_{t.task_id}_fewshot_{tail_tag}.json",
                    metadata_path=uniso_data_dir / f"exp1_{t.task_id}_fewshot_{tail_tag}.metadata",
                    xs=fs_x.numpy(),
                    ys=fs_y.numpy(),
                    metadata_text=fs_meta_text,
                )
                (uniso_data_dir / f"exp1_{t.task_id}_fewshot_{tail_tag}.meta.json").write_text(
                    json.dumps(
                        {
                            **dict(r["meta_struct"]),  # type: ignore[arg-type]
                            **{
                                "fewshot_tail_frac": float(tail_frac),
                                "fewshot_tail_tag": tail_tag,
                                "fewshot_budget": int(cfg.fewshot_budget),
                                "fewshot_points": int(fs_y.numel()),
                                "metadata_text": fs_meta_text,
                            },
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                fs_tr = _points_to_trajs(
                    xs=fs_x,
                    ys=fs_y,
                    horizon=int(cfg.horizon),
                    n_traj=int(cfg.duo_n_traj),
                    seed=int(cfg.seed),
                )
                fs_lat = native_traj_to_latent_padded(vae, fs_tr, d_pad=d_pad, device=dev)
                fs_name = (
                    f"{t.task_id}_fewshot_{tail_tag}_h{cfg.horizon}_n{int(cfg.duo_n_traj)}_lat{int(cfg.latent_dim)}.pkl"
                )
                fs_pkl = duo_ds_root / fs_name
                save_duo_single_task_pkl(fs_pkl, fs_lat)
                _write_sidecar(
                    fs_pkl,
                    d_native=d_nat,
                    d_pad=d_pad,
                    latent_dim=int(cfg.latent_dim),
                    vae_filename=vae_name,
                )

    if duo_train_merge_latent:
        pts = torch.cat([o.points for o in duo_train_merge_latent], dim=0)
        vals = torch.cat([o.values for o in duo_train_merge_latent], dim=0)
        pr = torch.cat([o.pr for o in duo_train_merge_latent], dim=0)
        rtg = torch.cat([o.rtg for o in duo_train_merge_latent], dim=0)
        ts = torch.cat([o.timesteps for o in duo_train_merge_latent], dim=0)
        merged = DuoTrajectoryPkl(points=pts, values=vals, pr=pr, rtg=rtg, timesteps=ts)
        merged_path = duo_ds_root / (
            f"train_merged_h{cfg.horizon}_n{int(pts.shape[0])}_lat{int(cfg.latent_dim)}.pkl"
        )
        save_duo_single_task_pkl(merged_path, merged)
        n_tr = int(cfg.duo_n_traj)
        segs: list[dict[str, object]] = []
        row0 = 0
        for tsk in fam.train_tasks:
            segs.append(
                {
                    "task_id": tsk.task_id,
                    "traj_row_start": row0,
                    "traj_row_end": row0 + n_tr,
                    "d_x_native": int(tsk.transform.d_x),
                    "vae_filename": "vae_shared.pt",
                }
            )
            row0 += n_tr
        man = {
            "merged": True,
            "latent_dim": int(cfg.latent_dim),
            "horizon": int(cfg.horizon),
            "segments": segs,
        }
        (merged_path.parent / f"{merged_path.name}.exp1_merge_manifest.json").write_text(
            json.dumps(man, indent=2), encoding="utf-8"
        )

    print(f"[exp1] wrote latent DUO pkls + VAE under {duo_ds_root}")
    print(f"[exp1] wrote UniSO native-x data under {uniso_data_dir} (exp1_*.json/.metadata)")
    print(f"[exp1] family meta: {out_root/'family_meta.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", type=str, default="branin", choices=["branin", "ackley"])
    ap.add_argument("--d_z", type=int, default=2)
    ap.add_argument(
        "--train_d_x",
        type=str,
        default="5,6,7,9,10",
        help="Comma-separated native obs dims for D_train_1..K (heterogeneous tasks).",
    )
    ap.add_argument(
        "--test_d_x",
        type=int,
        default=8,
        help="Native obs dim for D_test_gap*.",
    )
    ap.add_argument(
        "--d_x",
        type=int,
        default=0,
        help="Legacy uniform dim: if >0 and --train_d_x omitted, use d_x for each of n_train_tasks.",
    )
    ap.add_argument("--latent_dim", type=int, default=8, help="VAE / diffusion trajectory channel width.")
    ap.add_argument(
        "--d_pad",
        type=int,
        default=32,
        help="DUO shared padding dim before VAE (must be >= every train_d_x and test_d_x).",
    )
    ap.add_argument("--vae_train_steps", type=int, default=800, help="Adam steps per-task VAE on point pool.")
    ap.add_argument("--obs_noise_std", type=float, default=0.0)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--n_points", type=int, default=2000)
    ap.add_argument("--duo_n_traj", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--n_train_tasks",
        type=int,
        default=5,
        help="Used only when building uniform train dims from --d_x (all same dim).",
    )
    ap.add_argument(
        "--fewshot_budget",
        type=int,
        default=100,
        help="Few-shot trajectory construction: number of design points sampled per tail regime.",
    )
    ap.add_argument(
        "--fewshot_tail_fracs",
        type=str,
        default="0.1,0.2,0.5",
        help="Comma-separated tail fractions (worst-y pools); one PKL set per entry.",
    )
    ap.add_argument(
        "--train_pool_best_frac",
        type=float,
        default=0.9,
        help="D_train only: keep this top fraction of points by y (higher=better) before trajectory sampling.",
    )
    ap.add_argument(
        "--test_shifts",
        type=str,
        default="sim_low,sim_mid,sim_high",
        help="Comma-separated held-out test scenarios (metadata similarity low→high vs train).",
    )
    ap.add_argument(
        "--gaps",
        type=str,
        default="",
        help="Deprecated: if set, mapped to test_shifts 0→sim_low, 0.25→sim_mid, 0.5→sim_high.",
    )
    ap.add_argument("--obs_noise_base_std", type=float, default=0.03)
    ap.add_argument("--obs_noise_high_std", type=float, default=0.40)
    ap.add_argument("--obs_noise_high_dims", type=int, default=1)
    ap.add_argument(
        "--out_root",
        type=str,
        default="results/comparison1/exp1",
        help="Base output dir under DUO/",
    )
    ap.add_argument(
        "--uniso_data_dir",
        type=str,
        default="../UniSO/data",
        help="Where to write UniSO data/*.json and *.metadata",
    )
    ap.add_argument(
        "--pkl_suffix",
        type=str,
        default="",
        help="Append to exp1_<test_shift> PKL dir names (e.g. _3 for v3 line).",
    )
    ap.add_argument(
        "--branin_domain",
        type=str,
        default="legacy",
        choices=["legacy", "standard", "gtg"],
        help="Branin: standard uses usual plot box x1∈[-5,10], x2∈[0,15] (gtg is alias).",
    )
    ap.add_argument(
        "--family_mode",
        type=str,
        default="random_orth",
        choices=["random_orth", "scene_aware"],
        help="scene_aware: scenario metadata + MiniLM-correlated affine A.",
    )
    ap.add_argument(
        "--similarity_blend",
        type=float,
        default=0.65,
        help="Weight for blending A toward similar tasks (scene_aware only).",
    )
    ap.add_argument(
        "--text_encoder_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="MiniLM model for scene embedding / A correlation.",
    )
    args = ap.parse_args()
    _pkl_suffix = str(args.pkl_suffix).strip()

    raw_train = str(args.train_d_x).strip()
    if raw_train:
        train_d_x = _parse_int_tuple(raw_train)
    elif int(args.d_x) > 0:
        k = max(1, int(args.n_train_tasks))
        train_d_x = tuple(int(args.d_x) for _ in range(k))
    else:
        train_d_x = (5, 6, 7, 9, 10)

    _gap_map = {0.0: "sim_low", 0.25: "sim_mid", 0.5: "sim_high"}
    raw_shifts = str(args.test_shifts).strip()
    if str(args.gaps).strip():
        gaps = [float(x.strip()) for x in str(args.gaps).split(",") if x.strip()]
        shifts = [_gap_map.get(g, "sim_low") for g in gaps]
    elif raw_shifts:
        shifts = [x.strip() for x in raw_shifts.split(",") if x.strip()]
    else:
        shifts = ["sim_low", "sim_mid", "sim_high"]

    for shift in shifts:
        _st = f"{shift}{_pkl_suffix}"
        cfg = Exp1Config(
            out_root=Path(args.out_root) / f"shift_{shift}",
            objective=str(args.objective),
            d_z=int(args.d_z),
            train_d_x=train_d_x,
            test_d_x=int(args.test_d_x),
            latent_dim=int(args.latent_dim),
            d_pad=int(args.d_pad),
            vae_train_steps=int(args.vae_train_steps),
            obs_noise_std=float(args.obs_noise_std),
            obs_noise_base_std=float(args.obs_noise_base_std),
            obs_noise_high_std=float(args.obs_noise_high_std),
            obs_noise_high_dims=int(args.obs_noise_high_dims),
            horizon=int(args.horizon),
            n_points=int(args.n_points),
            duo_n_traj=int(args.duo_n_traj),
            seed=int(args.seed),
            test_shift=str(shift),
            pkl_suffix=_pkl_suffix,
            branin_domain=normalize_branin_domain(str(args.branin_domain)),
            family_mode=str(args.family_mode),
            similarity_blend=float(args.similarity_blend),
            text_encoder_model=str(args.text_encoder_model),
            uniso_data_dir=Path(args.uniso_data_dir),
            fewshot_budget=int(args.fewshot_budget),
            fewshot_tail_fracs=str(args.fewshot_tail_fracs),
            train_pool_best_frac=float(args.train_pool_best_frac),
        )
        _write_one(cfg)


if __name__ == "__main__":
    main()


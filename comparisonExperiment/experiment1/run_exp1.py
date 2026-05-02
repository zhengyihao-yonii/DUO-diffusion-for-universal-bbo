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
    train_d_x: tuple[int, ...] = (10, 12, 14, 18, 20)
    test_d_x: int = 16
    # Shared diffusion trajectory width (VAE mu in R^{latent_dim}), aligned with main DUO spirit.
    latent_dim: int = 16
    # Zero-pad native x to R^{d_pad} before shared VAE (default 32, main-experiment style).
    d_pad: int = 32
    vae_train_steps: int = 800
    obs_noise_std: float = 0.0
    # UniSO trains on points; we control points per task here (target ~2000).
    n_points: int = 2000
    # DUO consumes trajectories; we build 100x32 with replacement from the point pool.
    horizon: int = 32
    duo_n_traj: int = 100
    seed: int = 0
    # Train/test instance distance control
    gap: float = 0.0
    # Where to write UniSO text datasets
    uniso_data_dir: Path = Path("../UniSO/data")
    # Few-shot selection policy (fixed budget from test-task points)
    fewshot_y_max: float = 0.5
    fewshot_budget_frac: float = 0.10  # 10%N


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
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Generate iid points (x,y) for a task instance.

    Chinese comment: UniSO 训练输入是点 + metadata，这里只生成点，不额外“造轨迹”。
    """
    gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1000 + hash(task_id) % 997)
    d_z = int(fam.objective.d_z)
    z = torch.rand((int(n_points), d_z), generator=gen) * 2.0 - 1.0
    ts = next(t for t in (list(fam.train_tasks) + [fam.test_task]) if t.task_id == task_id)
    x = ts.transform.map(z, rng=gen).to(torch.float32)
    f = fam.objective.eval(z).to(torch.float32)
    f_lo = float(torch.min(f).item())
    f_hi = float(torch.max(f).item())
    y = 1.0 - _minmax_01(f)  # y_norm in [0,1], higher better
    return x, y, f_lo, f_hi


def _select_fewshot_points(
    *,
    xs: torch.Tensor,  # [N, d_x]
    ys: torch.Tensor,  # [N]
    y_max: float,
    budget_frac: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select few-shot points: among ys <= y_max, sample up to floor(budget_frac*N)."""
    N = int(ys.numel())
    b = int(max(1, math.floor(float(budget_frac) * float(N))))
    mask = ys <= float(y_max)
    idx = torch.nonzero(mask, as_tuple=False).view(-1)
    if idx.numel() == 0:
        # fallback: sample from all points
        idx = torch.arange(N)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = idx[torch.randperm(idx.numel(), generator=gen)]
    take = perm[: min(int(perm.numel()), b)]
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
        gap=float(cfg.gap),
        obs_noise_std=float(cfg.obs_noise_std),
        seed=int(cfg.seed),
    )
    dist = fam.geometric_distance(seed=cfg.seed)

    rec = {
        "objective": cfg.objective,
        "d_z": cfg.d_z,
        "train_d_x": list(cfg.train_d_x),
        "test_d_x": int(cfg.test_d_x),
        "d_pad": int(fam.d_pad),
        "latent_dim": int(cfg.latent_dim),
        "gap": cfg.gap,
        "geom_dist": dist,
        "n_points_per_task": int(cfg.n_points),
        "duo": {"horizon": int(cfg.horizon), "n_traj": int(cfg.duo_n_traj)},
        "seed": cfg.seed,
        "train_tasks": [t.task_id for t in fam.train_tasks],
        "test_task": fam.test_task.task_id,
        "fewshot": {
            "y_max": cfg.fewshot_y_max,
            "budget_frac": cfg.fewshot_budget_frac,
        },
    }
    (out_root / "family_meta.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    duo_ds_root = Path("generated_datasets") / f"exp1_gap{cfg.gap:.3f}".replace(".", "p")
    duo_ds_root.mkdir(parents=True, exist_ok=True)
    dev = _vae_device()
    duo_train_merge_latent: list[DuoTrajectoryPkl] = []
    d_pad = int(fam.d_pad)
    uniso_data_dir = cfg.uniso_data_dir.resolve()
    uniso_data_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for t in list(fam.train_tasks) + [fam.test_task]:
        d_nat = int(t.transform.d_x)
        x_pool, y_pool, f_lo, f_hi = _make_point_pool(
            fam, task_id=t.task_id, n_points=int(cfg.n_points), seed=int(cfg.seed)
        )
        xs = x_pool.detach().cpu().numpy()
        ys = y_pool.detach().cpu().numpy()
        meta = (
            "You are given a synthetic black-box optimization task. "
            "Each input is a design vector x in a task-specific dimension, and y is the normalized "
            "performance in [0,1] (higher is better).\n"
            "Tasks are heterogeneous in input dimension; DUO trajectories use a shared VAE on zero-padded x in R^{d_pad} "
            f"then a common {int(cfg.latent_dim)}-D latent for diffusion (similar spirit to main DUO).\n"
            f"task_id={t.task_id}\n"
            f"family={cfg.objective} (branin/ackley-like)\n"
            f"observation_dim_native={d_nat}\n"
            f"observation_dim_padded={d_pad}\n"
            f"latent_dim_diffusion={int(cfg.latent_dim)}\n"
            f"instance_gap={t.gap}\n"
            f"train_test_geom_dist={dist:.6f}\n"
        )
        meta_struct = {
            "exp": "comparison1_exp1",
            "task_id": t.task_id,
            "objective": cfg.objective,
            "d_z": cfg.d_z,
            "d_x": d_nat,
            "d_pad": d_pad,
            "latent_dim": int(cfg.latent_dim),
            "gap": float(t.gap),
            "geom_dist": float(dist),
            "f_min": float(f_lo),
            "f_max": float(f_hi),
            "A": t.transform.A.detach().cpu().numpy().tolist(),
            "b": t.transform.b.detach().cpu().numpy().tolist(),
            "obs_noise_std": float(t.transform.obs_noise_std),
            "metadata_text": meta,
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

        if t.task_id.startswith("D_test_gap"):
            x_pool = r["x_pool"]  # type: ignore[assignment]
            xs_t = x_pool.to(torch.float32)
            ys_t = r["y_pool"].to(torch.float32)  # type: ignore[union-attr]
            fs_x, fs_y = _select_fewshot_points(
                xs=xs_t,
                ys=ys_t,
                y_max=float(cfg.fewshot_y_max),
                budget_frac=float(cfg.fewshot_budget_frac),
                seed=int(cfg.seed),
            )
            fs_meta_text = meta + (
                f"fewshot_y_max={cfg.fewshot_y_max}\n"
                f"fewshot_budget_frac={cfg.fewshot_budget_frac}\n"
                "Setting: few-shot offline pool for adaptation / evaluation.\n"
            )
            save_uniso_jsonl(
                json_path=uniso_data_dir / f"exp1_{t.task_id}_fewshot.json",
                metadata_path=uniso_data_dir / f"exp1_{t.task_id}_fewshot.metadata",
                xs=fs_x.numpy(),
                ys=fs_y.numpy(),
                metadata_text=fs_meta_text,
            )
            (uniso_data_dir / f"exp1_{t.task_id}_fewshot.meta.json").write_text(
                json.dumps(
                    {
                        **dict(r["meta_struct"]),  # type: ignore[arg-type]
                        **{
                            "fewshot_y_max": float(cfg.fewshot_y_max),
                            "fewshot_budget_frac": float(cfg.fewshot_budget_frac),
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
            fs_name = f"{t.task_id}_fewshot_h{cfg.horizon}_lat{int(cfg.latent_dim)}.pkl"
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
        default="10,12,14,18,20",
        help="Comma-separated native obs dims for D_train_1..K (heterogeneous tasks).",
    )
    ap.add_argument(
        "--test_d_x",
        type=int,
        default=16,
        help="Native obs dim for D_test_gap*.",
    )
    ap.add_argument(
        "--d_x",
        type=int,
        default=0,
        help="Legacy uniform dim: if >0 and --train_d_x omitted, use d_x for each of n_train_tasks.",
    )
    ap.add_argument("--latent_dim", type=int, default=16, help="VAE / diffusion trajectory channel width.")
    ap.add_argument(
        "--d_pad",
        type=int,
        default=32,
        help="DUO shared padding dim before VAE (must be >= every train_d_x and test_d_x).",
    )
    ap.add_argument("--vae_train_steps", type=int, default=800, help="Adam steps per-task VAE on point pool.")
    ap.add_argument("--obs_noise_std", type=float, default=0.0)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--n_points", type=int, default=2000)
    ap.add_argument("--duo_n_traj", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--n_train_tasks",
        type=int,
        default=5,
        help="Used only when building uniform train dims from --d_x (all same dim).",
    )
    ap.add_argument("--fewshot_y_max", type=float, default=0.5)
    ap.add_argument("--fewshot_budget_frac", type=float, default=0.10)
    ap.add_argument(
        "--gaps",
        type=str,
        default="0.0,0.25,0.5,1.0",
        help="Comma-separated gap values for test task D (e.g. 0,0.2,0.5,1.0).",
    )
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
    args = ap.parse_args()

    raw_train = str(args.train_d_x).strip()
    if raw_train:
        train_d_x = _parse_int_tuple(raw_train)
    elif int(args.d_x) > 0:
        k = max(1, int(args.n_train_tasks))
        train_d_x = tuple(int(args.d_x) for _ in range(k))
    else:
        train_d_x = (10, 12, 14, 18, 20)

    gaps = [float(x.strip()) for x in str(args.gaps).split(",") if x.strip()]
    for g in gaps:
        cfg = Exp1Config(
            out_root=Path(args.out_root) / f"gap{g:.3f}".replace(".", "p"),
            objective=str(args.objective),
            d_z=int(args.d_z),
            train_d_x=train_d_x,
            test_d_x=int(args.test_d_x),
            latent_dim=int(args.latent_dim),
            d_pad=int(args.d_pad),
            vae_train_steps=int(args.vae_train_steps),
            obs_noise_std=float(args.obs_noise_std),
            horizon=int(args.horizon),
            n_points=int(args.n_points),
            duo_n_traj=int(args.duo_n_traj),
            seed=int(args.seed),
            gap=float(g),
            uniso_data_dir=Path(args.uniso_data_dir),
            fewshot_y_max=float(args.fewshot_y_max),
            fewshot_budget_frac=float(args.fewshot_budget_frac),
        )
        _write_one(cfg)


if __name__ == "__main__":
    main()


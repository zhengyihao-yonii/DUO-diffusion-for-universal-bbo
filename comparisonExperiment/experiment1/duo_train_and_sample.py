from __future__ import annotations

import argparse
import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys
import re

# Make script runnable via:
#   python comparisonExperiment/experiment1/duo_train_and_sample.py
# without requiring manual PYTHONPATH.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from comparisonExperiment.experiment1.exp1_latent_decode import decode_latent_matrix_to_native

from config import exp1_diffusion_aligned as _e1a
from diffuser.datasets.sequence import PointRegretDataset
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.models.temporal import Proxy, TemporalUnet
from diffuser.utils.training import Trainer, configure_wandb_step_axes


@dataclass(frozen=True)
class DuoRunResult:
    """Saved artifact for comparison1.

    English doc: store candidate designs from DUO sampling for unified oracle evaluation.
    """

    candidates_x: np.ndarray  # [N, d_x]
    aux: dict[str, Any]


def _to_jsonl_rows(x: np.ndarray, *, method: str) -> list[dict[str, Any]]:
    xs = np.asarray(x, dtype=np.float32)
    return [{"method": method, "x": xs[i].tolist()} for i in range(xs.shape[0])]


def _save_candidates_jsonl(path: Path, *, x: np.ndarray, method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _to_jsonl_rows(x, method=method)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _flatten_points_from_sample(
    *,
    sample: torch.Tensor,  # [B,H,dx+1] normalized
    dataset: PointRegretDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_unnorm, y_unnorm) flattened across horizon."""
    dx = int(dataset.observation_dim)
    # Normalizer stats live on CPU; move tensors to CPU before unnormalize.
    x_norm = sample[..., :dx].detach().to("cpu")
    y_norm = sample[..., dx : dx + 1].detach().to("cpu")
    x = dataset.normalizer.unnormalize(x_norm).detach().cpu().numpy().reshape(-1, dx)
    y = (
        dataset.normalizer_values.unnormalize(y_norm)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        type=str,
        default="train_and_sample",
        choices=["train", "sample", "train_and_sample"],
        help="train: only train; sample: only sample (requires --load_ckpt); train_and_sample: both.",
    )
    ap.add_argument("--train_pkl", type=str, required=True)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--train_steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_traj", type=int, default=256)
    ap.add_argument("--topk_points", type=int, default=128)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument(
        "--save_ckpt",
        type=str,
        default="",
        help="Optional checkpoint path to save after training.",
    )
    ap.add_argument(
        "--load_ckpt",
        type=str,
        default="",
        help="Checkpoint path to load before sampling.",
    )
    ap.add_argument(
        "--log_file",
        type=str,
        default="",
        help=(
            "Write full training/eval logs here (default: <resolved_out_jsonl_dir>/<stem>[__phase].log). "
            "Default log name is derived from --out_jsonl stem so runs do not overwrite each other."
        ),
    )
    ap.add_argument(
        "--artifact_group",
        type=str,
        default="",
        help=(
            "Optional subdirectory under .../seed{N}/ for all artifacts (e.g. st_duo | st_text | mt_label | mt_text). "
            "Resolves to .../seed{N}/<group>/<basename> for --out_jsonl, --save_ckpt, and default --log_file."
        ),
    )
    ap.add_argument(
        "--run_phase",
        type=str,
        default="",
        help=(
            "Optional tag for default log/meta filenames only, e.g. train | finetune | sample (suffix __<phase>). "
            "Use when the same --out_jsonl stem is reused across phases."
        ),
    )
    ap.add_argument("--use_proxy_filter", action="store_true")
    ap.add_argument("--proxy_steps", type=int, default=1000)
    ap.add_argument("--proxy_lr", type=float, default=2e-4)
    ap.add_argument(
        "--per_t_loss_bins",
        type=int,
        default=20,
        help="Number of per-timestep loss bins (default: 20).",
    )
    ap.add_argument(
        "--log_freq",
        type=int,
        default=0,
        help="Trainer log interval (steps); 0 = ant_config default (50).",
    )
    ap.add_argument(
        "--save_freq",
        type=int,
        default=0,
        help="Trainer.save interval (steps); 0 = ant_config default (5000).",
    )
    ap.add_argument(
        "--wandb",
        action="store_true",
        default=True,
        help="Enable wandb logging (default: on).",
    )
    ap.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging.",
    )
    ap.add_argument("--wandb_project", type=str, default="decdiff-opt")
    ap.add_argument("--wandb_run_name", type=str, default="")
    ap.add_argument("--wandb_group", type=str, default="comparison1-exp1")
    args = ap.parse_args()
    if bool(args.no_wandb):
        args.wandb = False

    out_path = _resolve_artifact_path(
        Path(args.out_jsonl),
        seed=int(args.seed),
        group=str(args.artifact_group),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    phase_tag = str(args.run_phase).strip()
    phase_suffix = f"__{phase_tag}" if phase_tag else ""
    if str(args.log_file).strip():
        log_path = _resolve_artifact_path(
            Path(args.log_file),
            seed=int(args.seed),
            group=str(args.artifact_group),
        )
    else:
        log_path = out_path.parent / f"{out_path.stem}{phase_suffix}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as _lf:
        with contextlib.redirect_stdout(_lf), contextlib.redirect_stderr(_lf):
            _run(args, out_path=out_path)


def _run(args: argparse.Namespace, *, out_path: Path) -> None:

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    data_path = Path(args.train_pkl)
    if not data_path.exists():
        raise FileNotFoundError(str(data_path))

    dataset = PointRegretDataset(
        horizon=int(args.horizon),
        data_path=str(data_path),
        context_length=0,
        regret=False,
        include_returns=False,
        task_name=None,
        task_text_embeds=None,
        include_task_idx=False,
    )

    transition_dim = int(dataset.observation_dim + dataset.action_dim)
    # English doc: UNet + diffusion match ``config/ant_config.py`` (main train).
    model = TemporalUnet(
        horizon=int(args.horizon),
        transition_dim=transition_dim,
        cond_dim=0,
        dim=int(_e1a.UNET_DIM),
        dim_mults=_e1a.UNET_DIM_MULTS,
        returns_condition=False,
        task_condition=False,
        num_tasks=1,
        condition_dropout=float(_e1a.UNET_CONDITION_DROPOUT),
        text_condition=False,
    )
    diffusion = GaussianDiffusion(
        model=model,
        horizon=int(args.horizon),
        observation_dim=int(dataset.observation_dim),
        action_dim=int(dataset.action_dim),
        n_timesteps=int(_e1a.N_DIFFUSION_STEPS),
        n_sample_timesteps=int(_e1a.N_SAMPLE_TIMESTEPS),
        loss_type=str(_e1a.LOSS_TYPE),
        clip_denoised=bool(_e1a.CLIP_DENOISED),
        predict_epsilon=bool(_e1a.PREDICT_EPSILON),
        action_weight=float(_e1a.ACTION_WEIGHT),
        returns_condition=False,
        condition_guidance_w=float(_e1a.CONDITION_GUIDANCE_W),
        condition_guidance_w_task=float(_e1a.CONDITION_GUIDANCE_W_TASK),
        condition_guidance_w_text=float(_e1a.CONDITION_GUIDANCE_W_TEXT),
    )
    device = torch.device(str(args.device))
    diffusion = diffusion.to(device)

    # Optional proxy: used only for filtering sampled x (matches main-experiment spirit).
    proxy_dataset = None
    proxy_model = None
    if bool(args.use_proxy_filter):
        # Chinese comment: 用训练集点(x)->y 的简单 ensemble MLP 当 proxy，筛选采样候选。
        with open(str(data_path), "rb") as f:
            obj = torch.load(f) if False else None  # unreachable; keep mypy quiet
        # Reuse PointRegretDataset internals: flatten original arrays directly from dataset.
        x_flat = dataset.points.reshape(-1, int(dataset.observation_dim))
        y_flat = dataset.values.reshape(-1, 1)

        class _ProxyDs(torch.utils.data.Dataset):
            def __init__(self, x0: torch.Tensor, y0: torch.Tensor) -> None:
                self.x0 = x0
                self.y0 = y0
                # Keep Trainer compatibility: it expects proxy_dataset.data_y in __init__.
                self.data_x = x0
                self.data_y = y0

            def __len__(self) -> int:
                return int(self.x0.shape[0])

            def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
                xi = dataset.normalizer.normalize(self.x0[i])
                yi = y0 = self.y0[i]
                if yi.ndim == 0:
                    yi = yi.view(1)
                return xi, yi

        proxy_dataset = _ProxyDs(x_flat, y_flat)
        proxy_model = Proxy(
            input_dim=int(dataset.observation_dim),
            hidden_dim=256,
            output_dim=1,
            n_ensembles=5,
        )
        proxy_model = proxy_model.to(device)

    _log_f = int(args.log_freq) if int(args.log_freq) > 0 else int(_e1a.MAIN_LOG_FREQ)
    _ts = int(args.train_steps)
    _save_f = int(args.save_freq) if int(args.save_freq) > 0 else int(_e1a.MAIN_SAVE_FREQ)
    _save_f = max(1, min(_save_f, _ts))
    trainer = Trainer(
        diffusion_model=diffusion,
        proxy_model=proxy_model,
        dataset=dataset,
        proxy_dataset=proxy_dataset,
        renderer=None,
        ema_decay=0.995,
        train_batch_size=int(args.batch_size),
        train_lr=float(args.lr),
        proxy_train_lr=float(args.proxy_lr),
        gradient_accumulate_every=int(args.grad_accum),
        log_freq=max(1, _log_f),
        sample_freq=0,
        save_freq=_save_f,
        proxy_save_freq=int(args.proxy_steps) + 1,
        train_device=str(args.device),
        save_checkpoints=False,
    )

    wb = None
    if bool(args.wandb):
        try:
            import wandb as _wandb

            from diffuser.utils.wandb_auth import init_wandb_run

            wb = _wandb
            # Use fixed per-timestep loss bins for stable comparison experiments.
            os.environ["DUO_LOG_PER_T_LOSS"] = "1"
            os.environ["DUO_LOG_PER_T_LOSS_BINS"] = str(int(args.per_t_loss_bins))
            init_wandb_run(
                str(args.wandb_project),
                name=str(args.wandb_run_name).strip() or f"exp1_duo_{out_path.parent.name}",
                group=str(args.wandb_group),
                config={
                    "train_pkl": str(args.train_pkl),
                    "horizon": int(args.horizon),
                    "train_steps": int(args.train_steps),
                    "proxy_steps": int(args.proxy_steps),
                    "batch_size": int(args.batch_size),
                    "lr": float(args.lr),
                    "proxy_lr": float(args.proxy_lr),
                    "mode": str(args.mode),
                    "use_proxy_filter": bool(args.use_proxy_filter),
                },
            )
            configure_wandb_step_axes(
                include_proxy_axis=bool(args.use_proxy_filter),
                include_finetune_axis=True,
            )
        except Exception as e:
            print(f"[wandb] init failed, continue without wandb: {e}")
            wb = None

    if str(args.load_ckpt).strip():
        load_p = _resolve_artifact_path(
            Path(args.load_ckpt),
            seed=int(args.seed),
            group=str(getattr(args, "artifact_group", "")),
        )
        trainer.load_from_path(str(load_p))

    # Let Trainer know total steps for optional schedules.
    setattr(trainer, "_total_train_steps", int(args.train_steps))

    do_train = str(args.mode) in ("train", "train_and_sample")
    do_sample = str(args.mode) in ("sample", "train_and_sample")

    if do_train:
        if bool(args.use_proxy_filter) and proxy_model is not None and proxy_dataset is not None:
            setattr(trainer, "_total_proxy_steps", int(args.proxy_steps))
            trainer.train_proxy(int(args.proxy_steps))
        trainer.train(int(args.train_steps))
        if str(args.save_ckpt).strip():
            ckpt_path = _resolve_artifact_path(
                Path(args.save_ckpt),
                seed=int(args.seed),
                group=str(args.artifact_group),
            )
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": int(trainer.step),
                    "model": trainer.model.state_dict(),
                    "ema": trainer.ema_model.state_dict(),
                },
                str(ckpt_path),
            )
            print(f"[duo] saved checkpoint: {ckpt_path}")

    if not do_sample:
        if wb is not None:
            try:
                wb.finish()
            except Exception:
                pass
        return

    # ---- sample ----
    trainer.ema_model.to(device)
    cond = {
        # conditional_sample/apply_conditioning expects tensor-like values with .to(...)
        "ctx_len": torch.zeros((int(args.sample_traj),), dtype=torch.long, device=device)
    }
    sample_out = trainer.ema_model.conditional_sample(cond, horizon=int(args.horizon), verbose=False)
    sample = sample_out[0] if isinstance(sample_out, tuple) else sample_out
    x, y = _flatten_points_from_sample(sample=sample, dataset=dataset)

    if bool(args.use_proxy_filter) and proxy_model is not None:
        proxy_model = proxy_model.to(device)
        with torch.no_grad():
            # Normalizer stats live on CPU; normalize on CPU then move to device for proxy.
            x_cpu = torch.tensor(np.asarray(x, dtype=np.float32), device="cpu")
            x_cpu_n = dataset.normalizer.normalize(x_cpu)
            x_tn = x_cpu_n.to(device)
            yhat = proxy_model(x_tn).detach().cpu().numpy().reshape(-1)
        key = yhat
    else:
        key = y

    k = min(int(args.topk_points), int(key.shape[0]))
    top_idx = np.argsort(key)[-k:]
    x_top = x[top_idx]
    x_export = decode_latent_matrix_to_native(data_path, x_top, device=device)
    if x_export is not None:
        x_top = x_export
    elif (data_path.parent / f"{data_path.name}.exp1_merge_manifest.json").is_file():
        print(
            "[warn] train_pkl is latent train_merged: jsonl stores latent x; "
            "use per-task pkl for sampling if you need native candidates for oracle eval."
        )

    _save_candidates_jsonl(out_path, x=x_top, method="duo")

    meta = {
        "train_pkl": str(data_path),
        "horizon": int(args.horizon),
        "train_steps": int(args.train_steps),
        "sample_traj": int(args.sample_traj),
        "topk_points": int(k),
        "use_proxy_filter": bool(args.use_proxy_filter),
    }
    _phase_tag = str(getattr(args, "run_phase", "")).strip()
    _phase_suffix = f"__{_phase_tag}" if _phase_tag else ""
    meta_name = f"{out_path.stem}{_phase_suffix}_run_meta.json"
    (out_path.parent / meta_name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[duo] saved candidates: {out_path} (n={x_top.shape[0]})")
    if wb is not None:
        try:
            wb.finish()
        except Exception:
            pass


def _seeded_path(path: Path, *, seed: int) -> Path:
    """
    Ensure output path is grouped by seed:
      foo/bar.json -> foo/seed0/bar.json
    If any parent already matches seed\\d+, keep as-is.
    """
    if re.search(r"^seed\d+$", path.parent.name):
        return path
    return path.parent / f"seed{int(seed)}" / path.name


def _resolve_artifact_path(path: Path, *, seed: int, group: str = "") -> Path:
    """Apply seed directory, then optional model/method group (e.g. four DUO variants)."""
    p = _seeded_path(path, seed=seed)
    raw = (group or "").strip()
    if not raw:
        return p
    g_safe = re.sub(r"[^\w\-.+]", "_", raw)
    if len(g_safe) > 120:
        g_safe = g_safe[:120]
    return p.parent / g_safe / p.name


if __name__ == "__main__":
    main()


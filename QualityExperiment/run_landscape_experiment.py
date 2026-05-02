#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI: latent landscape + DUO denoise traces on synthetic exp1 tasks, log to wandb.

English doc: Train checkpoints with ``comparisonExperiment/experiment1/duo_train_and_sample.py``
or main DUO pipeline; pass PKL + ``exp1_*.meta.json`` + EMA ckpt here.

Shift phases (**shift_zero_shot** / **shift_few_shot**) **require** held-out task text embeddings
for ``st_text`` and ``mt_text``: supply ``--held_out_text_embed_npy`` **or** rely on
``metadata_text`` in meta JSON + ``sentence-transformers`` (see ``--text_encoder_model``).

中文注释: 批量编排见 ``python -m QualityExperiment.run_quality_suite``。
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from comparisonExperiment.experiment1.task_family import LatentObjective

from comparisonExperiment.experiment1.exp1_latent_decode import decode_latent_matrix_to_native

from QualityExperiment.latent_geometry import latent_objective_grid, x_to_z_least_squares
from QualityExperiment.metadata_embed import resolve_shift_text_embedding
from QualityExperiment.trace_sampling import (
    TraceResult,
    load_ab_from_exp1_meta,
    sample_latent_trace,
)
from QualityExperiment.wandb_plots import (
    figure_to_wandb_image,
    log_combined_trajectory_table,
    log_trajectory_table,
    plot_latent_panel,
)


def _make_projector(meta_json: Path, train_pkl: Path, *, device: str):
    """Map diffusion observation rows (latent or native) -> mother z for 2D landscape."""
    a, b = load_ab_from_exp1_meta(meta_json)
    dev = torch.device(str(device))
    tp = Path(train_pkl)
    man = tp.parent / f"{tp.name}.exp1_merge_manifest.json"

    def project(u_rows: np.ndarray) -> np.ndarray:
        if man.is_file():
            raise ValueError(
                "train_pkl is train_merged latent: use a single-task *_lat*.pkl for Quality plots "
                "(merged rows mix multiple VAEs)."
            )
        u_rows = np.asarray(u_rows, dtype=np.float32)
        x_native = decode_latent_matrix_to_native(tp, u_rows, device=dev)
        if x_native is None:
            x_native = u_rows.astype(np.float64)
        return x_to_z_least_squares(a, b, x_native).astype(np.float32)

    return project


def _run_one_method(
    *,
    tag: str,
    ckpt: Path | None,
    train_pkl: Path,
    meta_json: Path,
    horizon: int,
    sample_batch: int,
    stride: int,
    device: str,
    model_overrides: dict | None,
    task_text_embeds: np.ndarray | None,
    include_task_idx: bool,
    task_idx: int,
    text_embed_override: np.ndarray | None,
) -> TraceResult | None:
    if ckpt is None or not ckpt.is_file():
        print(f"[skip] {tag}: missing ckpt {ckpt}")
        return None
    proj = _make_projector(meta_json, train_pkl, device=device)
    return sample_latent_trace(
        train_pkl=train_pkl,
        ckpt_path=ckpt,
        horizon=horizon,
        sample_batch=sample_batch,
        device=device,
        project_z=proj,
        step_callback_stride=stride,
        task_text_embeds=task_text_embeds,
        include_task_idx=include_task_idx,
        task_idx=task_idx,
        model_overrides=model_overrides,
        text_embed_override=text_embed_override,
    )


def _is_shift_phase(phase_label: str) -> bool:
    pl = str(phase_label).lower()
    return "shift" in pl


def _ckpt_ok(p: str) -> bool:
    return bool(str(p).strip()) and Path(p).is_file()


@dataclass
class LandscapeRunConfig:
    """English doc: One wandb run = one (meta_json, train_pkl, phase) bundle."""

    meta_json: Path
    train_pkl: Path
    horizon: int = 32
    sample_batch: int = 32
    stride: int = 3
    device: str = "cuda"
    wandb_project: str = "duo-quality-landscape"
    wandb_group: str = "exp1"
    wandb_run_name: str = ""
    ckpt_st_duo: str = ""
    ckpt_st_text: str = ""
    ckpt_mt_label: str = ""
    ckpt_mt_text: str = ""
    task_text_embeds_npy: str = ""
    task_idx: int = 0
    mt_num_tasks: int = 5
    no_traj_table: bool = False
    skip_methods: frozenset[str] = field(default_factory=frozenset)
    log_combined_table: bool = True
    phase_label: str = ""
    task_id_label: str = ""
    held_out_text_embed_npy: str = ""
    text_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_out_dir: str = ""


def run_landscape_experiment_core(cfg: LandscapeRunConfig) -> dict[str, np.ndarray]:
    """
    English doc: Execute sampling + wandb logging; returns ``traces`` dict for optional reuse.

    Caller must not init wandb beforehand; this function calls ``wandb.init`` / ``finish``.
    """
    meta = Path(cfg.meta_json)
    raw_meta = json.loads(meta.read_text(encoding="utf-8"))
    obj_name = str(raw_meta.get("objective", "branin"))
    d_z = int(raw_meta.get("d_z", 2))
    if d_z != 2:
        raise ValueError("Landscape plots require d_z=2 in meta_json.")

    obj = LatentObjective(name=obj_name, d_z=2)  # type: ignore[arg-type]
    z1, z2, fg = latent_objective_grid(obj, n=140)

    text_embeds = None
    if str(cfg.task_text_embeds_npy).strip():
        text_embeds = np.load(str(cfg.task_text_embeds_npy))

    shift = _is_shift_phase(cfg.phase_label)
    need_held = shift and (
        _ckpt_ok(cfg.ckpt_st_text) or _ckpt_ok(cfg.ckpt_mt_text)
    )
    held_vec: np.ndarray | None = None
    if need_held:
        expl = Path(cfg.held_out_text_embed_npy) if str(cfg.held_out_text_embed_npy).strip() else None
        held_vec = resolve_shift_text_embedding(
            meta,
            explicit_npy=expl if expl is not None and expl.is_file() else None,
            encoder_model=str(cfg.text_encoder_model),
        )

    def _text_dim() -> int:
        if text_embeds is not None:
            return int(text_embeds.shape[1])
        if held_vec is not None:
            return int(held_vec.shape[0])
        return 384

    ted = _text_dim()

    import wandb as wb

    wb.init(
        project=str(cfg.wandb_project),
        name=str(cfg.wandb_run_name).strip() or Path(cfg.train_pkl).stem,
        group=str(cfg.wandb_group),
        config={
            "meta_json": str(meta),
            "train_pkl": str(cfg.train_pkl),
            "horizon": int(cfg.horizon),
            "sample_batch": int(cfg.sample_batch),
            "phase": str(cfg.phase_label),
            "task_id": str(cfg.task_id_label),
            "task_idx": int(cfg.task_idx),
            "shift_held_out_embed": bool(held_vec is not None),
            "text_encoder_model": str(cfg.text_encoder_model),
        },
    )

    proj_for_oracle = _make_projector(meta, Path(cfg.train_pkl), device=str(cfg.device))

    methods: list[tuple[str, Path | None, dict | None, bool]] = [
        ("st_duo", Path(cfg.ckpt_st_duo) if cfg.ckpt_st_duo else None, None, False),
        (
            "st_text",
            Path(cfg.ckpt_st_text) if cfg.ckpt_st_text else None,
            {"text_condition": True, "text_embed_input_dim": ted},
            False,
        ),
        (
            "mt_label",
            Path(cfg.ckpt_mt_label) if cfg.ckpt_mt_label else None,
            {
                "task_condition": True,
                "num_tasks": int(cfg.mt_num_tasks),
            },
            True,
        ),
        (
            "mt_text",
            Path(cfg.ckpt_mt_text) if cfg.ckpt_mt_text else None,
            {
                "text_condition": True,
                "task_condition": True,
                "num_tasks": int(cfg.mt_num_tasks),
                "text_embed_input_dim": ted,
            },
            True,
        ),
    ]

    slug = f"{meta.stem}_{cfg.phase_label or 'run'}".strip("_")

    traces: dict[str, np.ndarray] = {}
    combined_rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    for tag, ckpt, overrides, use_task in methods:
        if tag in cfg.skip_methods:
            print(f"[skip] {tag}: in skip_methods")
            continue
        if tag in ("st_text", "mt_text"):
            if ckpt is None or not ckpt.is_file():
                continue
            if text_embeds is None and held_vec is None:
                print(f"[skip] {tag}: need task_text_embeds_npy or shift metadata embedding")
                continue
        override_vec = None
        if tag in ("st_text", "mt_text") and held_vec is not None:
            override_vec = held_vec
        tr = _run_one_method(
            tag=tag,
            ckpt=ckpt,
            train_pkl=Path(cfg.train_pkl),
            meta_json=meta,
            horizon=int(cfg.horizon),
            sample_batch=int(cfg.sample_batch),
            stride=int(cfg.stride),
            device=str(cfg.device),
            model_overrides=overrides,
            task_text_embeds=text_embeds,
            include_task_idx=use_task,
            task_idx=int(cfg.task_idx),
            text_embed_override=override_vec,
        )
        if tr is None:
            continue
        traces[tag] = tr.z_steps
        out_dir = Path(cfg.local_out_dir).resolve() if str(cfg.local_out_dir).strip() else Path(cfg.train_pkl).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npz = out_dir / f"quality_trace_{tag}_{slug}.npz"
        np.savez_compressed(
            str(out_npz),
            z_steps=tr.z_steps,
            x_last=tr.raw_x_last,
            timesteps=np.asarray(tr.timestep_indices, dtype=np.int32),
        )
        print(f"[save] {out_npz}")

        z_fin = proj_for_oracle(tr.raw_x_last)
        zt = torch.from_numpy(z_fin.astype(np.float32))
        raw_f = obj.eval(zt).detach().cpu().numpy().astype(np.float64)
        if not cfg.no_traj_table:
            log_trajectory_table(
                method=tag,
                z_steps=tr.z_steps,
                raw_f_final=raw_f,
                table_key=f"quality_table/trajectories_{tag}",
            )
        combined_rows.append((tag, tr.z_steps, raw_f))

    if traces:
        fig = plot_latent_panel(
            z1_grid=z1,
            z2_grid=z2,
            f_grid=fg,
            traces=traces,
            title=f"latent traces ({meta.stem})",
        )
        if str(cfg.local_out_dir).strip():
            png_path = Path(cfg.local_out_dir).resolve() / f"{slug}_latent_landscape.png"
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(png_path), format="png", bbox_inches="tight", dpi=150)
            print(f"[save] {png_path}")
        wb.log({"quality_figure/latent_landscape": figure_to_wandb_image(fig)})

    if cfg.log_combined_table and combined_rows and not cfg.no_traj_table:
        log_combined_trajectory_table(
            rows=combined_rows,
            table_key="quality_table/trajectories_all_methods",
        )

    wb.finish()
    return traces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_json", type=str, required=True, help="exp1_<task>.meta.json")
    ap.add_argument("--train_pkl", type=str, required=True)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--sample_batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--wandb_project", type=str, default="duo-quality-landscape")
    ap.add_argument("--wandb_group", type=str, default="exp1")
    ap.add_argument("--wandb_run_name", type=str, default="")
    ap.add_argument("--ckpt_st_duo", type=str, default="")
    ap.add_argument("--ckpt_st_text", type=str, default="")
    ap.add_argument("--ckpt_mt_label", type=str, default="")
    ap.add_argument("--ckpt_mt_text", type=str, default="")
    ap.add_argument("--task_text_embeds_npy", type=str, default="", help="[T,E] floats")
    ap.add_argument("--task_idx", type=int, default=0)
    ap.add_argument("--mt_num_tasks", type=int, default=5)
    ap.add_argument("--no_traj_table", action="store_true")
    ap.add_argument(
        "--phase_label",
        type=str,
        default="",
        help="e.g. train_domain | shift_zero_shot — triggers metadata embedding for text ckpts.",
    )
    ap.add_argument(
        "--held_out_text_embed_npy",
        type=str,
        default="",
        help="Optional [E] vector; else encode metadata_text from meta_json using text_encoder_model.",
    )
    ap.add_argument(
        "--text_encoder_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    ap.add_argument(
        "--local_out_dir",
        type=str,
        default="",
        help="Save PNG + NPZ here (default: next to train_pkl).",
    )
    args = ap.parse_args()

    cfg = LandscapeRunConfig(
        meta_json=Path(args.meta_json),
        train_pkl=Path(args.train_pkl),
        horizon=int(args.horizon),
        sample_batch=int(args.sample_batch),
        stride=int(args.stride),
        device=str(args.device),
        wandb_project=str(args.wandb_project),
        wandb_group=str(args.wandb_group),
        wandb_run_name=str(args.wandb_run_name),
        ckpt_st_duo=str(args.ckpt_st_duo),
        ckpt_st_text=str(args.ckpt_st_text),
        ckpt_mt_label=str(args.ckpt_mt_label),
        ckpt_mt_text=str(args.ckpt_mt_text),
        task_text_embeds_npy=str(args.task_text_embeds_npy),
        task_idx=int(args.task_idx),
        mt_num_tasks=int(args.mt_num_tasks),
        no_traj_table=bool(args.no_traj_table),
        phase_label=str(args.phase_label),
        held_out_text_embed_npy=str(args.held_out_text_embed_npy),
        text_encoder_model=str(args.text_encoder_model),
        local_out_dir=str(args.local_out_dir),
    )
    run_landscape_experiment_core(cfg)


if __name__ == "__main__":
    main()

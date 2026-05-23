#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI: latent landscape + DUO denoise traces on synthetic exp1 tasks, log to wandb.

English doc: Train checkpoints with ``QualityExperiment/train_exp1_checkpoints`` or
``comparisonExperiment/experiment1/duo_train_and_sample.py``; trace layout must match ckpt
(see ``quality_trace_arch.py``). Text-axis CFG scale: ``quality_text_condition_cfg``.

Shift phases (**shift_zero_shot** / **shift_few_shot**) **require** held-out task text embeddings
for ``st_text`` and ``mt_text``: supply ``--held_out_text_embed_npy`` **or** rely on
``metadata_text`` in meta JSON + ``sentence-transformers`` (see ``--text_encoder_model``).

中文注释: 批量编排见 ``python -m QualityExperiment.run_quality_suite``。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from QualityExperiment.quality_text_condition_cfg import CONDITION_GUIDANCE_W_TEXT as _Q_W_TEXT
from QualityExperiment.quality_trace_arch import (
    ACTION_WEIGHT,
    CLIP_DENOISED,
    CONDITION_GUIDANCE_W,
    CONDITION_GUIDANCE_W_TASK,
    LOSS_TYPE,
    N_SAMPLE_TIMESTEPS,
    N_TIMESTEPS,
    UNET_CONDITION_DROPOUT,
    UNET_DIM,
    UNET_DIM_MULTS,
)

# English doc: Layout must match Quality quartet checkpoints on disk; text CFG from ``quality_text_condition_cfg``.
_TRACE_DIFF_OVERRIDES: dict[str, object] = {
    "n_timesteps": N_TIMESTEPS,
    "n_sample_timesteps": N_SAMPLE_TIMESTEPS,
    "dim": UNET_DIM,
    "dim_mults": UNET_DIM_MULTS,
    "condition_dropout": UNET_CONDITION_DROPOUT,
    "loss_type": LOSS_TYPE,
    "clip_denoised": CLIP_DENOISED,
    "action_weight": ACTION_WEIGHT,
    "condition_guidance_w": CONDITION_GUIDANCE_W,
    "condition_guidance_w_task": CONDITION_GUIDANCE_W_TASK,
    "condition_guidance_w_text": float(_Q_W_TEXT),
}

from comparisonExperiment.experiment1.task_family import LatentObjective
from QualityExperiment.quality_proxy import resolve_proxy_path_for_eval

from comparisonExperiment.experiment1.exp1_latent_decode import decode_latent_matrix_to_native

from QualityExperiment.latent_geometry import (
    objective_grid_for_meta,
    x_to_z_least_squares,
    z_path_to_branin_xy,
)
from QualityExperiment.metadata_embed import resolve_shift_text_embedding
from QualityExperiment.trace_sampling import (
    TraceResult,
    load_ab_from_exp1_meta,
    sample_latent_trace,
)
from QualityExperiment.wandb_plots import (
    figure_to_wandb_image,
    filter_traces_for_landscape,
    log_combined_trajectory_table,
    log_trajectory_table,
    plot_branin_physical_panel,
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


def _context_length_for_ckpt(ckpt: Path | None, cfg: LandscapeRunConfig) -> int:
    """English doc: FS checkpoints (``*_fs_*``) use few-shot ctx; main ckpts use train ctx."""
    if ckpt is None or not ckpt.is_file():
        return int(cfg.context_length_train)
    if "_fs_" in ckpt.stem:
        return int(cfg.context_length_fewshot)
    return int(cfg.context_length_train)


def _run_one_method(
    *,
    tag: str,
    ckpt: Path | None,
    train_pkl: Path,
    meta_json: Path,
    horizon: int,
    context_length: int,
    sample_batch: int,
    stride: int,
    device: str,
    model_overrides: dict | None,
    task_text_embeds: np.ndarray | None,
    include_task_idx: bool,
    task_idx: int,
    text_embed_override: np.ndarray | None,
    prefix_seed: int = 0,
    use_proxy_filter: bool = True,
    phase_label: str = "",
    task_id_label: str = "",
) -> TraceResult | None:
    if ckpt is None or not ckpt.is_file():
        print(f"[skip] {tag}: missing ckpt {ckpt}")
        return None
    proj = _make_projector(meta_json, train_pkl, device=device)
    raw = json.loads(Path(meta_json).read_text(encoding="utf-8"))
    obj = LatentObjective(name=str(raw["objective"]), d_z=int(raw.get("d_z", 2)))
    proxy_p = None
    if use_proxy_filter:
        proxy_p = resolve_proxy_path_for_eval(
            Path(ckpt),
            phase_label=str(phase_label),
            task_id=str(task_id_label),
        )
        if not proxy_p.is_file():
            print(
                f"[warn] {tag}: proxy missing {proxy_p} "
                f"(phase={phase_label!r} task={task_id_label!r}), legacy min f"
            )
            proxy_p = None
    return sample_latent_trace(
        train_pkl=train_pkl,
        ckpt_path=ckpt,
        horizon=horizon,
        context_length=int(context_length),
        sample_batch=sample_batch,
        device=device,
        project_z=proj,
        step_callback_stride=stride,
        task_text_embeds=task_text_embeds,
        include_task_idx=include_task_idx,
        task_idx=task_idx,
        model_overrides=model_overrides,
        text_embed_override=text_embed_override,
        prefix_seed=int(prefix_seed),
        proxy_ckpt=proxy_p,
        objective=obj,
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
    horizon: int = 64
    context_length_train: int = 32
    context_length_fewshot: int = 16
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
    mt_num_tasks: int = 4
    no_traj_table: bool = False
    no_landscape_figure: bool = False
    skip_methods: frozenset[str] = field(default_factory=frozenset)
    log_combined_table: bool = True
    max_rep_trajs: int = 4
    min_z_sep: float = 0.25
    plot_all_trajs: bool = False
    split_landscape_per_method: bool = True
    phase_label: str = ""
    task_id_label: str = ""
    held_out_text_embed_npy: str = ""
    text_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_out_dir: str = ""
    use_proxy_filter: bool = True


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

    from comparisonExperiment.experiment1.branin_standard import normalize_branin_domain

    bd = normalize_branin_domain(str(raw_meta.get("branin_domain", "legacy")))
    obj = LatentObjective(
        name=obj_name, d_z=2, branin_domain=bd  # type: ignore[arg-type]
    )
    ax0, ax1, fg = objective_grid_for_meta(raw_meta, obj, n=140)
    use_branin_xy = obj_name == "branin" and bd == "standard"

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

    from diffuser.utils.wandb_auth import init_wandb_run

    init_wandb_run(
        str(cfg.wandb_project),
        name=str(cfg.wandb_run_name).strip() or Path(cfg.train_pkl).stem,
        group=str(cfg.wandb_group),
        config={
            "meta_json": str(meta),
            "train_pkl": str(cfg.train_pkl),
            "horizon": int(cfg.horizon),
            "context_length_train": int(cfg.context_length_train),
            "context_length_fewshot": int(cfg.context_length_fewshot),
            "sample_batch": int(cfg.sample_batch),
            "phase": str(cfg.phase_label),
            "task_id": str(cfg.task_id_label),
            "task_idx": int(cfg.task_idx),
            "shift_held_out_embed": bool(held_vec is not None),
            "text_encoder_model": str(cfg.text_encoder_model),
            "quality_condition_guidance_w_text": float(_Q_W_TEXT),
        },
    )

    proj_for_oracle = _make_projector(meta, Path(cfg.train_pkl), device=str(cfg.device))

    methods: list[tuple[str, Path | None, dict | None, bool]] = [
        ("st_duo", Path(cfg.ckpt_st_duo) if cfg.ckpt_st_duo else None, dict(_TRACE_DIFF_OVERRIDES), False),
        (
            "st_text",
            Path(cfg.ckpt_st_text) if cfg.ckpt_st_text else None,
            {**_TRACE_DIFF_OVERRIDES, "text_condition": True, "text_embed_input_dim": ted},
            False,
        ),
        (
            "mt_label",
            Path(cfg.ckpt_mt_label) if cfg.ckpt_mt_label else None,
            {
                **_TRACE_DIFF_OVERRIDES,
                "task_condition": True,
                "num_tasks": int(cfg.mt_num_tasks),
            },
            True,
        ),
        (
            "mt_text",
            Path(cfg.ckpt_mt_text) if cfg.ckpt_mt_text else None,
            {
                **_TRACE_DIFF_OVERRIDES,
                "text_condition": True,
                "task_condition": True,
                "num_tasks": int(cfg.mt_num_tasks),
                "text_embed_input_dim": ted,
            },
            True,
        ),
    ]

    slug = f"{meta.stem}_{cfg.phase_label or 'run'}".strip("_")

    def _stable_prefix_seed(method_tag: str) -> int:
        """English doc: Deterministic across processes (unlike built-in hash())."""
        digest = hashlib.md5(f"{slug}:{method_tag}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % (2**31)

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
            context_length=_context_length_for_ckpt(ckpt, cfg),
            sample_batch=int(cfg.sample_batch),
            stride=int(cfg.stride),
            device=str(cfg.device),
            model_overrides=overrides,
            task_text_embeds=text_embeds,
            include_task_idx=use_task,
            task_idx=int(cfg.task_idx),
            text_embed_override=override_vec,
            prefix_seed=_stable_prefix_seed(tag),
            use_proxy_filter=bool(cfg.use_proxy_filter),
            phase_label=str(cfg.phase_label),
            task_id_label=str(cfg.task_id_label),
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
            proxy_best_idx=np.int32(tr.proxy_best_idx),
            proxy_best_f=np.float64(tr.proxy_best_f),
            legacy_min_f=np.float64(tr.legacy_min_f),
            proxy_scores=(
                tr.proxy_scores.astype(np.float32)
                if tr.proxy_scores is not None
                else np.zeros(0, dtype=np.float32)
            ),
        )
        print(
            f"[save] {out_npz} proxy_best_f={tr.proxy_best_f:.4f} "
            f"legacy_min_f={tr.legacy_min_f:.4f}"
        )

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

    if traces and not bool(cfg.no_landscape_figure):
        raw_f_map = {tag: rf for tag, _zs, rf in combined_rows}
        plot_traces = traces
        if not bool(cfg.plot_all_trajs):
            plot_traces = filter_traces_for_landscape(
                traces,
                raw_f_map,
                max_rep_trajs=int(cfg.max_rep_trajs),
                min_z_sep=float(cfg.min_z_sep),
            )
        out_dir = (
            Path(cfg.local_out_dir).resolve()
            if str(cfg.local_out_dir).strip()
            else Path(cfg.train_pkl).parent
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        if bool(cfg.split_landscape_per_method):
            for tag, zpath in plot_traces.items():
                if use_branin_xy:
                    xy = z_path_to_branin_xy(zpath)
                    fig = plot_branin_physical_panel(
                        x1_grid=ax0,
                        x2_grid=ax1,
                        f_grid=fg,
                        traces_xy={tag: xy},
                        title=f"Branin ({meta.stem})",
                        method_label=tag,
                    )
                    png_name = f"{slug}_branin_xy_{tag}.png"
                    wb_key = f"quality_figure/branin_xy_{tag}"
                else:
                    fig = plot_latent_panel(
                        z1_grid=ax0,
                        z2_grid=ax1,
                        f_grid=fg,
                        traces={tag: zpath},
                        title=f"latent traces ({meta.stem}) — {tag}",
                    )
                    png_name = f"{slug}_latent_{tag}.png"
                    wb_key = f"quality_figure/latent_{tag}"
                png_path = out_dir / png_name
                fig.savefig(str(png_path), format="png", bbox_inches="tight", dpi=150)
                print(f"[save] {png_path}")
                wb.log({wb_key: figure_to_wandb_image(fig)})
        else:
            if use_branin_xy:
                xy_traces = {t: z_path_to_branin_xy(zp) for t, zp in plot_traces.items()}
                fig = plot_branin_physical_panel(
                    x1_grid=ax0,
                    x2_grid=ax1,
                    f_grid=fg,
                    traces_xy=xy_traces,
                    title=f"Branin ({meta.stem})",
                )
                png_name = f"{slug}_branin_xy_combined.png"
            else:
                fig = plot_latent_panel(
                    z1_grid=ax0,
                    z2_grid=ax1,
                    f_grid=fg,
                    traces=plot_traces,
                    title=f"latent traces ({meta.stem})",
                )
                png_name = f"{slug}_latent_landscape.png"
            png_path = out_dir / png_name
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
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--context_length_train", type=int, default=32)
    ap.add_argument("--context_length_fewshot", type=int, default=16)
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
    ap.add_argument("--mt_num_tasks", type=int, default=4)
    ap.add_argument("--no_traj_table", action="store_true")
    ap.add_argument(
        "--no_landscape_figure",
        action="store_true",
        help="Skip latent landscape PNG + wandb image (still writes quality_trace_*.npz).",
    )
    ap.add_argument(
        "--max_rep_trajs",
        type=int,
        default=4,
        help="Landscape figure: max representative denoise trajectories per method (best-f + z-sep).",
    )
    ap.add_argument(
        "--min_z_sep",
        type=float,
        default=0.25,
        help="Min Euclidean distance in z between selected representative trajectories.",
    )
    ap.add_argument(
        "--plot_all_trajs",
        action="store_true",
        help="Landscape figure: plot every sampled trajectory (no representative filtering).",
    )
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
        context_length_train=int(args.context_length_train),
        context_length_fewshot=int(args.context_length_fewshot),
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
        no_landscape_figure=bool(args.no_landscape_figure),
        max_rep_trajs=int(args.max_rep_trajs),
        min_z_sep=float(args.min_z_sep),
        plot_all_trajs=bool(args.plot_all_trajs),
        phase_label=str(args.phase_label),
        held_out_text_embed_npy=str(args.held_out_text_embed_npy),
        text_encoder_model=str(args.text_encoder_model),
        local_out_dir=str(args.local_out_dir),
    )
    run_landscape_experiment_core(cfg)


if __name__ == "__main__":
    main()

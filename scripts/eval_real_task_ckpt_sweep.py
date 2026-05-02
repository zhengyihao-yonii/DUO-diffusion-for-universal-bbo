#!/usr/bin/env python3
"""
Sweep real-task few-shot evaluate.py over several fine-tune checkpoints (by epoch).

Writes per seed under MODEL_ROOT/seed{seed}/:
  - eval_w{wfile}_{E}epochs.log
  - eval_summary_w{wfile}_{E}epochs.json   (requires evaluate --eval_summary_json_out)

Optionally logs metrics to Weights & Biases as curves with step=epoch.

Example:
  cd /data/xk/zyh_dfgo/DUO && PYTHONPATH=. python3 scripts/eval_real_task_ckpt_sweep.py \\
    --project-root /data/xk/zyh_dfgo/DUO \\
    --run-meta results/real_task/few_shot/lunar_lander_frac1.0_sigma0.0/fs_k100_worst/100x64_k20_eps0.05_latent64_tsbias0.5_lr0.0002_mt911054c35daad7e0_ft/run_meta_*.env \\
    --model-root results/real_task/few_shot/lunar_lander_frac1.0_sigma0.0/fs_k100_worst/100x64_k20_eps0.05_latent64_tsbias0.5_lr0.0002_mt911054c35daad7e0_ft \\
    --epoch-checkpoints 10,20,30,40,50,60 \\
    --seeds 0,1,2,3,4,5,6,7 \\
    --w-text 8.0 \\
    --wandb-project decdiff-opt \\
    --wandb-run-name lunar_lander_fs_ckpt_curve
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping


def _parse_run_meta_env(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line == "---":
            continue
        if "=" not in line:
            continue
        for part in line.split():
            if "=" not in part:
                continue
            k, _, v = part.partition("=")
            k = k.strip()
            if not k:
                continue
            out[k] = v.strip()
    return out


def _w_file_token(w: float) -> str:
    s = f"{float(w):.6f}"
    if "." in s:
        s = s.rstrip("0")
        if s.endswith("."):
            s += "0"
    return s.replace(".", "p")


def _fewshot_ckpt_dir(meta: Mapping[str, str], *, project_root: Path, seed: int) -> Path:
    """
    Must match run_real_tasks.sh:_fewshot_trained_ckpt_dir.
    """
    t = meta["TASK"]
    frac = meta.get("FRAC", "1.0")
    sigma = meta.get("SIGMA", "0.0")
    n_traj = meta.get("N_TRAJ", "100")
    horizon = meta.get("HORIZON", "64")
    k = meta.get("K", "20")
    eps = meta.get("EPS", "0.05")
    latent = meta.get("LATENT_DIM", "32")

    # Suffixes are precomputed by run_real_tasks.sh and written to run_meta.
    diff_suf = meta.get("DIFF_TRAIN_SUF", "")
    lr_suf = meta.get("LR_SUF", "")
    txt_infix = meta.get("GTG_TEXTCOND_PATH_INFIX", "_textcond")
    mto_infix = meta.get("GTG_MTTEXTONLY_PATH_INFIX", "_mttextonly")
    ret_infix = meta.get("GTG_RETCOND_PATH_INFIX", "_retcond")
    use_ret = meta.get("USE_RETURNS", "0").strip() == "1"
    ret = ret_infix if use_ret else ""
    lat = f"_latent{latent}" if str(latent) != "32" else ""

    ckpt = (
        project_root
        / "trained_models"
        / f"{t}_frac{frac}_sigma{sigma}"
        / f"{n_traj}x{horizon}_k{k}_eps{eps}_fewshot_ft{ret}{txt_infix}{mto_infix}{diff_suf}{lr_suf}{lat}"
        / f"seed{int(seed)}"
        / "checkpoint"
    )
    return ckpt


def _build_eval_cmd(
    *,
    meta: Mapping[str, str],
    project_root: Path,
    seed: int,
    epoch_label: int,
    w_text: float,
    ckpt_path: Path,
    json_path: Path | None,
) -> list[str]:
    t = meta["TASK"]
    frac = meta.get("FRAC", "1.0")
    sigma = meta.get("SIGMA", "0.0")
    n_traj = meta.get("N_TRAJ", "100")
    k = meta.get("K", "20")
    eps = meta.get("EPS", "0.05")
    horizon = meta.get("HORIZON", "64")
    latent = meta.get("LATENT_DIM", "32")
    tbp = meta.get("TRAIN_TIMESTEP_BIAS_POWER", "0.0")
    tmg = meta.get("TRAIN_LOSS_MIN_SNR_GAMMA", "0.0")
    thf = meta.get("TRAIN_HALF_TBIAS_FRAC", "0.7")
    hlm = meta.get("TRAIN_HALF_LR_MULT", "1.0")
    lr = meta.get("TRAIN_LEARNING_RATE", "").strip()

    cmd: list[str] = [
        sys.executable,
        str(project_root / "evaluate.py"),
        "--train_tasks",
        t,
        "--real_task_text_only_finetune",
        "--pretrained_multitask_train_tasks",
        meta.get("PRETRAINED_MULTITASK_TRAIN_TASKS", ""),
        "--ctx_len",
        str(int(meta.get("FEWSHOT_CTX_LEN", "8") or 8)),
        "--frac",
        frac,
        "--sigma",
        sigma,
        "--n_traj",
        n_traj,
        "--k",
        k,
        "--eps",
        eps,
        "--horizon",
        horizon,
        "--seed",
        str(seed),
        "--latent_dim",
        latent,
        "--train_timestep_bias_power",
        tbp,
        "--train_loss_min_snr_gamma",
        tmg,
        "--train_half_timestep_bias_frac",
        thf,
        "--train_half_lr_mult",
        hlm,
        "--condition_guidance_w_text",
        str(w_text),
        "--train_epochs",
        str(epoch_label),
        "--load_diffusion_checkpoint",
        str(ckpt_path),
    ]
    if json_path is not None:
        cmd += ["--eval_summary_json_out", str(json_path)]
    if lr:
        cmd += ["--learning_rate", lr]
    pf = meta.get("PROXY_FILTER", "").strip()
    if pf in ("0", "1"):
        cmd += ["--proxy_filter", pf]
    return cmd


def _wandb_log_curve(
    *,
    wandb_run,
    epoch: int,
    task: str,
    metrics: Mapping[str, float],
) -> None:
    payload = {"epoch": int(epoch)}
    for k, v in metrics.items():
        payload[f"{task}/{k}"] = float(v)
    wandb_run.log(payload, step=int(epoch))


def _parse_single_task_metrics_from_log(text: str, *, task: str) -> dict[str, float] | None:
    """
    Parse metrics from evaluate stdout for single-task runs.

    Expected lines (examples):
      [lunar_lander] max_ep_reward: 1.23, median: 0.1, mean: 0.2
      [lunar_lander] nmax_ep_reward: 0.9, nmedian: 0.8, nmean: 0.7
      [lunar_lander] top8_mean (raw oracle, fitting diag): 0.5
      [lunar_lander] ntop8_mean (normalized oracle, fitting diag): 0.4
    """
    import re

    out: dict[str, float] = {}
    m1 = re.search(
        rf"^\[{re.escape(task)}\]\s+max_ep_reward:\s*([-\d\.eE+]+),\s*median:\s*([-\d\.eE+]+),\s*mean:\s*([-\d\.eE+]+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if m1:
        out["max"] = float(m1.group(1))
        out["median"] = float(m1.group(2))
        out["mean"] = float(m1.group(3))
    m2 = re.search(
        rf"^\[{re.escape(task)}\]\s+nmax_ep_reward:\s*([-\d\.eE+]+),\s*nmedian:\s*([-\d\.eE+]+),\s*nmean:\s*([-\d\.eE+]+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if m2:
        out["nmax"] = float(m2.group(1))
        out["nmedian"] = float(m2.group(2))
        out["nmean"] = float(m2.group(3))
    m3 = re.search(
        rf"^\[{re.escape(task)}\]\s+top8_mean\s+.*:\s*([-\d\.eE+]+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if m3:
        out["top8_mean"] = float(m3.group(1))
    m4 = re.search(
        rf"^\[{re.escape(task)}\]\s+ntop8_mean\s+.*:\s*([-\d\.eE+]+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if m4:
        out["ntop8_mean"] = float(m4.group(1))
    return out if out else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--run-meta", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument(
        "--epoch-checkpoints",
        type=str,
        required=True,
        help="Comma-separated epochs to evaluate (e.g. 10,20,60).",
    )
    p.add_argument("--seeds", type=str, required=True, help="Comma-separated seeds.")
    p.add_argument("--w-text", type=float, default=8.0)
    p.add_argument("--force", action="store_true", help="Re-run even if json exists.")
    p.add_argument(
        "--no-json",
        action="store_true",
        help="Do not write eval_summary JSON; log curves to wandb by parsing eval logs.",
    )
    p.add_argument("--wandb-project", type=str, default="")
    p.add_argument("--wandb-run-name", type=str, default="")
    args = p.parse_args()

    if not args.run_meta.is_file():
        raise SystemExit(f"--run-meta not found: {args.run_meta}")
    meta = _parse_run_meta_env(args.run_meta.read_text(encoding="utf-8"))

    epochs = [int(x.strip()) for x in str(args.epoch_checkpoints).split(",") if x.strip()]
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    wtok = _w_file_token(float(args.w_text))

    project_root = args.project_root.resolve()
    model_root = args.model_root.resolve()
    model_root.mkdir(parents=True, exist_ok=True)

    task = meta.get("TASK", "")
    if not task:
        raise SystemExit("run-meta missing TASK=")

    wandb_run = None
    if str(args.wandb_project).strip():
        try:
            import wandb

            wandb_run = wandb.init(
                project=str(args.wandb_project).strip(),
                name=str(args.wandb_run_name).strip() or f"{model_root.name}_fs_ckpt",
                config=dict(meta) | {"epochs": epochs, "seeds": seeds, "w_text": float(args.w_text)},
            )
            if hasattr(wandb, "define_metric"):
                wandb.define_metric("epoch")
                wandb.define_metric(f"{task}/*", step_metric="epoch")
        except Exception as e:
            print(f"[wandb] init failed, continue without wandb: {e}", file=sys.stderr)
            wandb_run = None

    for seed in seeds:
        out_seed_dir = model_root / f"seed{seed}"
        out_seed_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = _fewshot_ckpt_dir(meta, project_root=project_root, seed=int(seed))
        if not ckpt_dir.is_dir():
            print(f"[skip] missing ckpt dir: {ckpt_dir}", file=sys.stderr)
            continue

        spe = int(meta.get("REAL_TASK_N_STEPS_PER_EPOCH", meta.get("N_STEPS_PER_EPOCH", "100")))
        if spe <= 0:
            spe = 100

        for ep in epochs:
            ckpt_path = ckpt_dir / f"state_{int(ep) * int(spe)}.pt"
            log_path = out_seed_dir / f"eval_w{wtok}_{ep}epochs.log"
            json_path = None if bool(args.no_json) else (out_seed_dir / f"eval_summary_w{wtok}_{ep}epochs.json")

            if json_path is not None and json_path.is_file() and (not args.force):
                data = json.loads(json_path.read_text(encoding="utf-8"))
                tm = (data.get("tasks") or {}).get(task) or {}
                if wandb_run is not None and tm:
                    _wandb_log_curve(wandb_run=wandb_run, epoch=int(ep), task=task, metrics=tm)
                continue
            if json_path is None and log_path.is_file() and (not args.force) and wandb_run is not None:
                txt = log_path.read_text(encoding="utf-8", errors="replace")
                tm = _parse_single_task_metrics_from_log(txt, task=task)
                if tm:
                    _wandb_log_curve(wandb_run=wandb_run, epoch=int(ep), task=task, metrics=tm)
                    continue

            if not ckpt_path.is_file():
                log_path.write_text(f"missing checkpoint: {ckpt_path}\n", encoding="utf-8")
                continue

            cmd = _build_eval_cmd(
                meta=meta,
                project_root=project_root,
                seed=int(seed),
                epoch_label=int(ep),
                w_text=float(args.w_text),
                ckpt_path=ckpt_path,
                json_path=json_path,
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as lf:
                lf.write("[cmd] " + " ".join(cmd) + "\n")
                lf.flush()
                proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=os.environ.copy())
            if proc.returncode != 0:
                continue

            if wandb_run is not None:
                if json_path is not None:
                    try:
                        data = json.loads(json_path.read_text(encoding="utf-8"))
                        tm = (data.get("tasks") or {}).get(task) or {}
                        if tm:
                            _wandb_log_curve(wandb_run=wandb_run, epoch=int(ep), task=task, metrics=tm)
                    except Exception:
                        pass
                else:
                    try:
                        txt = log_path.read_text(encoding="utf-8", errors="replace")
                        tm = _parse_single_task_metrics_from_log(txt, task=task)
                        if tm:
                            _wandb_log_curve(wandb_run=wandb_run, epoch=int(ep), task=task, metrics=tm)
                    except Exception:
                        pass

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()


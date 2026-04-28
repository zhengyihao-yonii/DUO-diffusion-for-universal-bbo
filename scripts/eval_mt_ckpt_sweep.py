#!/usr/bin/env python3
"""
Sweep multitask evaluate.py over several diffusion checkpoints (by train-step / epoch).

Writes per seed:
  - eval_w{wfile}_{E}epochs.log               (stdout/stderr from evaluate)
  - eval_summary_w{wfile}_{E}epochs.json      (requires evaluate --eval_summary_json_out)

Example:
  cd /data/xk/zyh_dfgo/DUO && PYTHONPATH=. python3 scripts/eval_mt_ckpt_sweep.py \\
    --run-meta results/epoch1500/mt_911054c35daad7e0_textcond_mttextonly_ce0.005_tsbias0.5_lr0.0002/run_meta_20260426_213639.env \\
    --epoch-checkpoints 500,800,1000,1200,1400,1500 \\
    --seeds 0,1,2,3 \\
    --w-text 8.0
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
            key, _, val = part.partition("=")
            key = key.strip()
            if not key:
                continue
            out[key] = val.strip()
    return out


def _w_file_token(w: float) -> str:
    """
    Stable filename token for w_text.

    Examples:
      8.0  -> "8p0"   (always keep one decimal place)
      8.5  -> "8p5"
      8.25 -> "8p25"
    """
    # Use a fixed precision, then strip trailing zeros but keep at least one decimal.
    s = f"{float(w):.6f}"
    if "." in s:
        s = s.rstrip("0")
        if s.endswith("."):
            s += "0"
    return s.replace(".", "p")


def _build_eval_cmd(
    *,
    meta: Mapping[str, str],
    project_root: Path,
    seed: int,
    train_epochs_run: int,
    ckpt_train_steps: int,
    w_text: float,
    json_path: Path,
) -> list[str]:
    train_tasks = meta["TRAIN_TASKS"]
    frac = meta.get("FRAC", "1.0")
    sigma = meta.get("SIGMA", "0.0")
    n_traj = meta.get("N_TRAJ", "1000")
    k = meta.get("K", "50")
    eps = meta.get("EPS", "0.05")
    horizon = meta.get("HORIZON", "64")
    latent = meta.get("LATENT_DIM", "32")
    tbp = meta.get("TRAIN_TIMESTEP_BIAS_POWER", "0.0")
    tmg = meta.get("TRAIN_LOSS_MIN_SNR_GAMMA", "0.0")
    thf = meta.get("TRAIN_HALF_TBIAS_FRAC", "0.7")
    hlm = meta.get("TRAIN_HALF_LR_MULT", "1.0")
    run_suffix = meta.get("RUN_SUFFIX", "")
    tpj = meta.get("TRAJ_PARAMS_JSON", "")
    text_enc = meta.get(
        "TEXT_ENCODER", "sentence-transformers/all-MiniLM-L6-v2"
    )

    cmd: list[str] = [
        sys.executable,
        str(project_root / "evaluate.py"),
        "--train_tasks",
        train_tasks,
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
        "--use_text_condition",
        "--multitask_text_only",
        "--text_encoder_model",
        text_enc,
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
        "--train_epochs",
        str(train_epochs_run),
        "--diffusion_ckpt_train_steps",
        str(ckpt_train_steps),
        "--condition_guidance_w_text",
        str(w_text),
        "--eval_summary_json_out",
        str(json_path),
    ]
    lr = meta.get("TRAIN_LEARNING_RATE", "").strip()
    if lr:
        cmd += ["--learning_rate", lr]
    if run_suffix:
        cmd += ["--run_suffix", run_suffix]
    if tpj:
        cmd += ["--traj_params_json", tpj]
    pf = meta.get("PROXY_FILTER", "").strip()
    if pf in ("0", "1"):
        cmd += ["--proxy_filter", pf]
    return cmd


def _try_rebuild_json_from_log(
    *,
    log_path: Path,
    json_path: Path,
    meta: Mapping[str, str],
    train_epochs_run: int,
    ckpt_train_steps: int,
    n_steps_per_epoch: int,
    w_text: float,
) -> bool:
    """
    Best-effort: when eval log exists but JSON is missing, parse the log and write JSON.

    Expected log contains:
      - "[evaluate] 加载扩散权重: <path>"
      - "=== 多任务评估汇总 ... ===" table with columns including top16 / nt16.
    """
    if not log_path.is_file() or log_path.stat().st_size <= 0:
        return False
    try:
        txt = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False

    # Parse loadpath.
    m_lp = re.search(r"^\[evaluate\]\s+加载扩散权重:\s*(.+?)\s*$", txt, flags=re.MULTILINE)
    loadp = m_lp.group(1).strip() if m_lp else ""
    m_step = re.search(r"state_(\d+)\.pt$", loadp)
    ckpt_step_from_file = int(m_step.group(1)) if m_step else None
    equiv_epochs = (ckpt_step_from_file // n_steps_per_epoch) if (ckpt_step_from_file is not None and n_steps_per_epoch > 0) else None

    # Parse the multitask summary table rows.
    # Example row (after our update):
    #   ant                 123.0000   ...   456.0000 |     0.1234 ...
    row_re = re.compile(
        r"^\s*([A-Za-z0-9_]+)\s+"
        r"([-\d\.eE+]+)\s+([-\d\.eE+]+)\s+([-\d\.eE+]+)\s+([-\d\.eE+]+)\s+\|\s+"
        r"([-\d\.eE+]+)\s+([-\d\.eE+]+)\s+([-\d\.eE+]+)\s+([-\d\.eE+]+)\s*$"
    )
    tasks: dict[str, dict[str, float]] = {}
    for ln in txt.splitlines():
        m = row_re.match(ln)
        if not m:
            continue
        t = m.group(1)
        try:
            tasks[t] = {
                "max": float(m.group(2)),
                "median": float(m.group(3)),
                "mean": float(m.group(4)),
                "top16_mean": float(m.group(5)),
                "nmax": float(m.group(6)),
                "nmedian": float(m.group(7)),
                "nmean": float(m.group(8)),
                "ntop16_mean": float(m.group(9)),
            }
        except ValueError:
            continue

    if not tasks:
        return False

    payload: dict[str, object] = {
        "is_multitask": True,
        "train_tasks": [t.strip() for t in str(meta.get("TRAIN_TASKS", "")).split(",") if t.strip()],
        "train_epochs": train_epochs_run,
        "diffusion_ckpt_train_steps": ckpt_train_steps,
        "diffusion_checkpoint_train_steps_from_file": ckpt_step_from_file,
        "eval_equiv_train_epochs": equiv_epochs,
        "n_steps_per_epoch": int(n_steps_per_epoch),
        "condition_guidance_w_text": float(w_text),
        "diffusion_checkpoint_loadpath": loadp,
        "tasks": tasks,
        "_rebuilt_from_log": str(log_path),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--run-meta", type=Path, required=True)
    p.add_argument(
        "--epoch-checkpoints",
        type=str,
        required=True,
        help="Comma-separated train epochs to evaluate (e.g. 500,800,1500); "
        "step = epoch * n_steps_per_epoch.",
    )
    p.add_argument("--seeds", type=str, required=True, help="Comma-separated seeds.")
    p.add_argument("--w-text", type=float, default=8.0)
    p.add_argument(
        "--train-epochs-run",
        type=int,
        default=None,
        help="TRAIN_EPOCHS of the finished run (default: read from run-meta TRAIN_EPOCHS).",
    )
    p.add_argument("--n-steps-per-epoch", type=int, default=100)
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-run even if the corresponding eval log/json already exist.",
    )
    args = p.parse_args()

    run_meta = args.run_meta
    if run_meta.is_dir():
        raise SystemExit(
            f"--run-meta expects a file path, got a directory: {run_meta}\n"
            "Tip: if you used RUN_META=... inline, bash expands $RUN_META before the "
            "assignment. Do this instead:\n"
            "  RUN_META='path/to/run_meta.env'; PYTHONPATH=. python3 ... --run-meta \"$RUN_META\""
        )
    if not run_meta.is_file():
        raise SystemExit(f"--run-meta file not found: {run_meta}")
    meta = _parse_run_meta_env(run_meta.read_text(encoding="utf-8"))
    project_root = args.project_root.resolve()
    model_root = (project_root / meta.get("MODEL_ROOT", "")).resolve()
    if not model_root.is_dir():
        raise SystemExit(f"MODEL_ROOT is not a directory: {model_root}")

    train_epochs_run = int(
        args.train_epochs_run
        if args.train_epochs_run is not None
        else int(meta.get("TRAIN_EPOCHS", "1500"))
    )
    epochs = [int(x.strip()) for x in args.epoch_checkpoints.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    wtok = _w_file_token(float(args.w_text))

    os.environ.setdefault("PYTHONPATH", str(project_root))
    for seed in seeds:
        seed_dir = model_root / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for ep in epochs:
            ckpt_steps = ep * int(args.n_steps_per_epoch)
            if ckpt_steps > train_epochs_run * int(args.n_steps_per_epoch):
                print(
                    f"[skip] seed={seed} epoch={ep} -> steps={ckpt_steps} "
                    f"beyond run train_epochs={train_epochs_run}",
                    flush=True,
                )
                continue
            log_path = seed_dir / f"eval_w{wtok}_{ep}epochs.log"
            json_path = seed_dir / f"eval_summary_w{wtok}_{ep}epochs.json"
            if not args.force:
                log_ok = log_path.is_file() and log_path.stat().st_size > 0
                json_ok = json_path.is_file() and json_path.stat().st_size > 0
                if json_ok:
                    print(
                        f"[skip] seed={seed} ep={ep} -> already exists: {json_path.name}",
                        flush=True,
                    )
                    continue
                if log_ok:
                    rebuilt = _try_rebuild_json_from_log(
                        log_path=log_path,
                        json_path=json_path,
                        meta=meta,
                        train_epochs_run=train_epochs_run,
                        ckpt_train_steps=ckpt_steps,
                        n_steps_per_epoch=int(args.n_steps_per_epoch),
                        w_text=float(args.w_text),
                    )
                    if rebuilt:
                        print(
                            f"[skip] seed={seed} ep={ep} -> rebuilt json from log: {json_path.name}",
                            flush=True,
                        )
                        continue
                if log_ok and json_ok:
                    print(
                        f"[skip] seed={seed} ep={ep} -> already exists: "
                        f"{log_path.name} + {json_path.name}",
                        flush=True,
                    )
                    continue
            cmd = _build_eval_cmd(
                meta=meta,
                project_root=project_root,
                seed=seed,
                train_epochs_run=train_epochs_run,
                ckpt_train_steps=ckpt_steps,
                w_text=float(args.w_text),
                json_path=json_path,
            )
            print(f"[run] seed={seed} ep={ep} -> {log_path}", flush=True)
            with open(log_path, "ab", buffering=0) as lf:
                lf.write(f"\n### CMD: {' '.join(cmd)}\n".encode())
                rc = subprocess.call(cmd, cwd=str(project_root), stdout=lf, stderr=subprocess.STDOUT)
            if rc != 0:
                print(f"[warn] exit {rc} seed={seed} ep={ep}", flush=True)


if __name__ == "__main__":
    main()

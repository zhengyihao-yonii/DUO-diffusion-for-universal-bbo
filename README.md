# GTGdfgo

Diffusion-based offline optimization on Design-Bench–style tasks (builds on the GTG / decision-diffuser stack). This repo adds multitask training, optional text conditioning from `task_metadata/`, and shell helpers `run_multitask.sh` / `run_singletask.sh`.

**Base dependencies** (MuJoCo, PyTorch, design-bench, jaynes, etc.) match the upstream GTG setup; use the same conda stack as your GTG checkout or install the packages listed in the original GTG README.

---

## Quick start

```bash
# Multitask: construct trajectories → train → eval (see run_multitask.sh for positional args)
bash run_multitask.sh "dkitty,ant" 3 1000 50 0.05 64 1.0 0.0
```

Single-task wrapper: `bash run_singletask.sh <task> ...` (see script header).

Core Python entrypoints: `construct_trajectories.py`, `train.py`, `evaluate.py`.

**Evaluate tables:** `bash run_analyze_eval.sh` writes everything under **`results/analysis_table/`**: `eval_comparison*`, `eval_comparison_m12*`, `eval_comparison_all*`; UniSO baselines are read from **`results/analysis_table/uniso_result.tex`** (optional **`d_best.json`**). The `_all` table (`-final`) uses **one DFGO column per subfolder** of `results/text_conditioned_only/all_frac1.0_sigma0.0/` (override base name with `EVAL_ALL_TASK_FRAC_SIG`); column titles are the subfolder names (key hyperparameters). Same as `python scripts/analyze_eval_results.py --mode …`.

**Results layout:** Single-task runs use `results/<task>_multiple_runs[_retcond]` only (VAE + trajectory + diffusion, no text). Multitask with text conditioning + `--multitask_text_only` defaults to `results/text_conditioned_only/all_frac<F>_sigma<S>/w<w_text>_.../` for the full 8-task set, or `text_conditioned_only/<token>_frac<F>_sigma<S>/...` for subgroups; the fallback `multitask_*` directory uses `TEXTCOND_MTTEXTONLY_SUFFIX` (default `_textcond_mttextonly`) when both modes are on.

---

## Environment variables (optional)

Set these in the shell **before** launching scripts (unless noted).

| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | Physical GPU id(s) visible to PyTorch (e.g. `0` or `2`). |
| `GPU_ID` | If `CUDA_VISIBLE_DEVICES` is **unset**, `scripts/gpu_env.sh` sets it from `GPU_ID`. |
| `GTG_DEVICE` | Torch device for VAE / construct helpers (`cuda`, `cuda:0`, `cpu`). Overrides auto-selection in `train_vae` / `resolve_torch_device`. |
| `GTG_DISTANCE_ON_GPU` | Construct trajectory distance matrix: default use GPU when available; set to `0` to force CPU. |
| `CPU_THREADS` | Cap OpenMP / BLAS / PyTorch CPU threads (e.g. `4`). Same effect as `--cpu_threads N` on `train.py` / `evaluate.py` / `construct_trajectories.py` / `train_vae.py`. |
| `PYTHON` | Python interpreter path (default in `run_*.sh` points to a conda `gtg` env; override if needed). |
| `PROJECT` | Repo root; default is the directory containing the invoked `run_*.sh`. |
| `RESULTS` | Override auto-generated `results/...` run directory (see `run_multitask.sh` / `run_singletask.sh` comments). |
| `TEXT_ENCODER_MODEL` | Absolute path to an offline sentence-transformers snapshot (used when text conditioning is enabled). |
| `USE_RETURNS`, `USE_TEXT_CONDITION`, `TRAIN_EXTRA`, `EVAL_EXTRA_CMD`, `EVAL_ALL`, `START_SEED`, `AUTO_CONTINUE`, `EVAL_ONLY`, … | Pipeline behavior; **full list and defaults** are documented in the header comments of `run_multitask.sh` and `run_singletask.sh`. |

---

## Multitask checkpoints

Training and evaluation share one directory: `trained_models/multi_<tasks>_frac.../` (no per-eval-task suffix). See `run_multitask.sh` comments for `EVAL_ALL` / `EVAL_ONLY`.

---

## Task text metadata

Short English blurbs per task for optional text conditioning: see `task_metadata/README.md`.

---

## Optional: SOO benchmarks

`thirdparty_benchmark/` and `bash scripts/setup_soo_bench.sh` — details in `thirdparty_benchmark/README.md`.

# DUO

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

**Evaluate tables:** `bash run_analyze_eval.sh` writes **`results/analysis_table/max_short.{csv,tex}`**, **`nmax.tex`** (normalized baselines from **`uniso_nresult.tex`** plus DUO), **`max_extended.{csv,tex}`**, and **`text_conditioned_result_analysis.tex`** (when using `-final`); **`bash run_analyze_eval.sh -sweep-w`** writes **`max_ablation.{tex,csv}`** (text CFG `w` ablation from `results/eval_sweep_w_text/`). **UniSO-T Improved** (single column from **`uniso_result.tex`**, not max over UniSO-T/N) feeds the wide tables (optional **`d_best.json`**). The text-conditioned analysis table (`-final`) uses **one DUO column per subfolder** of `results/text_conditioned_only/all_frac1.0_sigma0.0/` (override base name with `EVAL_ALL_TASK_FRAC_SIG`); column titles are the subfolder names (key hyperparameters). Same as `python scripts/analyze_eval_results.py --mode …`.

**Results layout:** Single-task runs use `results/<task>_multiple_runs[_retcond]` only (VAE + trajectory + diffusion, no text). Multitask with text conditioning + `--multitask_text_only` defaults to `results/text_conditioned_only/all_frac<F>_sigma<S>/w<w_text>_.../` for the full 9-task set (ant, dkitty, gtopx2–6, superconductor, tfbind10, tfbind8), or `text_conditioned_only/<token>_frac<F>_sigma<S>/...` for subgroups; the fallback `multitask_*` directory uses `TEXTCOND_MTTEXTONLY_SUFFIX` (default `_textcond_mttextonly`) when both modes are on.

### Hyperparameter documentation (maintainers)

When you add or change CLI flags, `run_*.sh` defaults, `train.py` / `evaluate.py` knobs, or `Config` fields, **update this README** (and the relevant script header comments) so defaults and environment variables stay discoverable.

### Real-task pipeline (`run_real_tasks.sh`)

| Variable / flag | Default | Meaning |
|-----------------|---------|---------|
| `FEWSHOT_K` | `128` if **unset** | Few-shot subset size on merged JSON (before `frac`). Use an **empty** `FEWSHOT_K=` in the environment if you want **no** k-subsampling (`fewshot_k=None`). |
| `FEWSHOT_MODE` | `worst` | Few-shot selection: smallest `y` (objective “worst” when larger is better). |
| `TRAIN_EPOCHS` (`run_real_tasks.sh` few-shot) | `40` | Passed to `train.py --train_epochs`. Few-shot **fine-tuning** uses fewer epochs than full multitask pretrain (typically `TRAIN_EPOCHS=200` via `run_multitask.sh`). Override with `export TRAIN_EPOCHS=200` if you want a long fine-tune. |
| `--latent_observation_dim` | unset | Optional: override latent / observation channels for **real-task zero-shot** eval if `vae_info.p` is missing or wrong (else read from `vae_info` or **32**). |

**Few-shot diffusion fine-tuning (not training from random init):** With `--real_task_text_only_finetune`, `train.py` sets `load_diffusion_checkpoint` to the multitask text pretrained `state*.pt`, and `Trainer.__init__` calls `load_from_path`: **both** `model` and `ema_model` weights are loaded from that file, and `trainer.step` resumes from the checkpoint’s global step. The Adam optimizer is **new** (no optimizer state from pretrain). Training then runs for `train_epochs × n_steps_per_epoch` **additional** gradient steps on the single-task trajectory data. Default `--train_epochs` is **40** for this mode and **200** for non–real-task runs when omitted (`train.py`).

**Re-running few-shot** with new defaults or hyperparameters: you do **not** need to delete the **multitask pretrained** tree under `trained_models/multi_*` (shared base). Delete or rename only the **single-task few-shot run** you want to replace: `trained_models/<task>_frac<F>_sigma<S>/<n_traj>x<h>_k<k>_eps<eps>_..._fewshot_ft/seed<seed>/` (especially `checkpoint/`). Or point `RESULTS` / `REAL_TASK_RESULTS_ROOT` to a new directory so logs don’t overwrite. `generated_datasets/` trajectory pkls can stay unless you changed construct parameters.

**Real-task zero-shot (lunar_lander / robot_push / rover):** `--real_task_zero_shot_eval` **always** uses **no trajectory pkl** for conditioning: `ctx_len=0`, no injected context frames (optional **task** / **text** global conditioning only). `MinimalTrajectoryDataset` satisfies the `Trainer` / `DataLoader` interface; oracle physical dim comes from `REAL_WORLD_FEWSHOT_TASK_SPECS`. Outputs use a `_nctx` filename suffix. Requires local `generated_datasets/.../vae_info.p` (and VAE weights) for decode.

**Implementation notes:** `diffuser/utils/training.py` sets `drop_last=False` when `len(dataset) < batch_size` so an empty `DataLoader` cannot spin forever under `cycle()`. `diffuser/utils/progress.py` uses flushed `print` for redirected logs.

---

## Environment variables (optional)

Set these in the shell **before** launching Python or `run_*.sh` (unless noted). CLI flags usually override when both exist.

### Device, threading, logging

| Variable | Purpose |
|----------|---------|
| `CUDA_VISIBLE_DEVICES` | Physical GPU id(s) visible to PyTorch (e.g. `0` or `2`). |
| `GPU_ID` | If `CUDA_VISIBLE_DEVICES` is **unset**, `scripts/gpu_env.sh` can set it from `GPU_ID`. |
| `GTG_DEVICE` | Torch device for VAE / construct helpers (`cuda`, `cuda:0`, `cpu`). Overrides auto-selection in `train_vae` / `resolve_torch_device` (`diffuser/utils/construct_runtime.py`). |
| `GTG_DISTANCE_ON_GPU` | Construct trajectory **distance matrix**: default `1` = use GPU when available; set to `0` to force CPU. |
| `CPU_THREADS` | Cap OpenMP / BLAS / PyTorch CPU threads (e.g. `4`). Same effect as `--cpu_threads N` on `train.py` / `evaluate.py` / `construct_trajectories.py` / `train_vae.py` (`diffuser/cpu_threads.py`). |
| `PYTHONUNBUFFERED` | When `1`, Python stdout/stderr are unbuffered; useful when redirecting logs to files. `run_real_tasks.sh` sets it to `1` by default. |
| `PYTHON` | Python interpreter path (some scripts default to `python3` or a conda env; `search_gtopx_traj_hyperparams.py` also reads this). |

### Repo layout and shell wrappers

| Variable | Purpose |
|----------|---------|
| `PROJECT` | Repo root passed into `run_singletask.sh` / `run_multitask.sh` / `run_real_tasks.sh` (default: script directory). |
| `RESULTS` | Override auto-generated `results/...` run directory (see `run_multitask.sh` / `run_singletask.sh` headers). |
| `REAL_TASK_RESULTS_ROOT` | `run_real_tasks.sh`: base directory for real-task zero-shot / few-shot outputs (default under `results/real_task/`). |
| `TEXT_ENCODER_MODEL` | Absolute path to an offline **sentence-transformers** snapshot when text conditioning is enabled (`run_multitask.sh` can auto-append `--text_encoder_model`). |

Pipeline-only knobs (`USE_RETURNS`, `USE_TEXT_CONDITION`, `TRAIN_EXTRA`, `EVAL_EXTRA_CMD`, `EVAL_ALL`, `START_SEED`, `NUM_SEEDS`, `AUTO_CONTINUE`, `EVAL_ONLY`, `TRAJ_PARAMS_JSON`, `FRAC` / `SIGMA` / `N_TRAJ` / …) are documented in the **header comments** of `run_multitask.sh`, `run_singletask.sh`, `run_real_tasks.sh`, and `scripts/train_eval_sweep_w_text.sh`.

### Proxy model (train + eval query filtering)

| Variable | Purpose |
|----------|---------|
| `PROXY_FILTER` | `1` (default): train the proxy where applicable and **rank/filter** evaluation queries with it. `0`: **do not** train/load proxy; evaluation uses diffusion samples directly (no proxy scoring). Read in `diffuser/utils/proxy_filter.py`. Zero-shot real-task eval forces proxy off regardless; few-shot can set `PROXY_FILTER=0` or `1`. Shell scripts may pass `--proxy_filter` when this variable is set. |

### Weights & Biases (`train.py` / `evaluate.py`)

| Variable | Purpose |
|----------|---------|
| `WANDB_DISABLED` | If `1` / `true` / `yes`, skip `wandb.init`. |
| `WANDB_MODE` | e.g. `offline` for local-only logs (combined with `GTG_WANDB_OFFLINE`). |
| `GTG_WANDB_OFFLINE` | If set to a truthy value, treated like offline wandb (see scripts). |
| `WANDB_INIT_TIMEOUT` | Seconds for online `wandb.init` (default `300`). |

### Real-task transfer (pretrained multitask checkpoint resolution)

| Variable | Purpose |
|----------|---------|
| `GTG_REAL_TASK_PRETRAINED_MT_HEX` | Default **16-hex** multitask hyper segment for pretrained diffusion (default `911054c35daad7e0`). `diffuser/utils/real_task_transfer.py`. |
| `GTG_REAL_TASK_PRETRAINED_MULTITASK_CSV` | Comma-separated **train_tasks** CSV for that pretrained run (default full 9-task dictionary order). |

### Real-world few-shot data and LunarLander oracle

| Variable | Purpose |
|----------|---------|
| `GTG_REAL_WORLD_FEWSHOT_DIR` | Directory containing `real_task_data/meta_dataset.json`; default `<repo>/real_task_data`. |
| `GTG_REAL_WORLD_FEWSHOT_K` | Override few-shot subset size `k` (integer). |
| `GTG_REAL_WORLD_FEWSHOT_MODE` | Override mode: `all` \| `random` \| `worst`. |
| `GTG_REAL_WORLD_FEWSHOT_SEED` | Override RNG seed for few-shot selection. |
| `GTG_LUNAR_ORACLE_N_ENVS` | **LunarLander only**: number of env seeds **per design** averaged in the oracle (default **5** if unset; was 50 historically). Higher = slower but lower variance. `diffuser/utils/real_world_oracle.py`. |

### Checkpoint path suffix overrides (multitask naming)

| Variable | Purpose |
|----------|---------|
| `GTG_MTTEXTONLY_PATH_INFIX` | Default `_mttextonly`; used in hyper directory naming. `diffuser/utils/multitask_canon.py`. |
| `GTG_TEXTCOND_PATH_INFIX` | Default `_textcond`. |
| `GTG_RETCOND_PATH_INFIX` | Default `_retcond` (returns-conditioned runs). |

### Optional SOO / external bench

| Variable | Purpose |
|----------|---------|
| `SOO_BENCH_ROOT` | Root for SOO benchmark data if not using defaults (`diffuser/utils/soo_gtopx.py`). |

### `scripts/analyze_eval_results.py` (tables / sweep)

These tune where aggregated results and text-CFG sweep columns are read from (defaults match `results/` layout).

| Variable | Purpose |
|----------|---------|
| `EVAL_ALL_TASK_FRAC_SIG` | e.g. `all_frac1.0_sigma0.0` for text-conditioned “all tasks” folder names. |
| `DUO_TASK_FRAC_SIG` | Default `frac1.0_sigma0.0` segment for single/multi task paths under `results/`. |
| `DUO_FULL_MULTITASK_PREFIX` | Override full multitask “all tasks” results prefix (includes hyper subdirectory when needed). |
| `DUO_FULL_MULTITASK_HYPER` | Force hyper subdirectory name for full-multitask unified paths. |
| `DUO_NMAX_MULTITASK_PREFIX` | Override prefix for nmax multitask text rows. |
| `EVAL_ALL_IMPROVED_TASK_FRAC_SIG` | Subdir name under `text_conditioned_only/` for `all_improved_*` layout. |
| `DUO_SWEEP_W_PREFIX` | When set, analysis uses `eval_sweep_w_text/...` checkpoint tree for text CFG `w` columns. |
| `DUO_SWEEP_W_VALUE` | Set together with sweep prefix by tooling; documents chosen `w`. |
| `DUO_SWEEP_W_DISABLE` | Truthy values disable sweep-w-specific behavior in some table modes. |
| `SWEEP_W_MODEL_DIR` | Override model directory for sweep-w ablation resolution. |

### Trajectory JSON / multitask slug (`max_short_traj_context`, tables)

Used by `scripts/analyze_eval_results.py` (wide-table alignment) and training/eval when matching `mt_<hex>` folders to `examples/traj_params_per_task_example2.json`.

| Variable | Purpose |
|----------|---------|
| `DUO_MAX_SHORT_TRAJ_JSON` | Path to per-task trajectory hyperparameter JSON (default `examples/traj_params_per_task_example2.json`). |
| `DUO_MAX_SHORT_HORIZON` | Diffusion horizon for multitask trajectory signature / checkpoint hyper dir (default `64`). |
| `DUO_MAX_SHORT_FULL_MT_TASKS` | Comma-separated task list for full-multitask slug (must match construct/train). |

### TF-Bind auxiliary CE during diffusion training (`diffuser/models/diffusion.py`)

Adds **MSE + λ·CE** on TF-Bind logit dimensions when `predict_epsilon=True`. `train.py` passes `train_tasks_list` into the diffusion object so batch rows map to task names.

| Variable | Purpose |
|----------|---------|
| `DUO_DISCRETE_CE_LAMBDA` | Weight **λ** for the auxiliary cross-entropy term (default **`0`** = CE disabled). |
| `DUO_DISCRETE_CE_TASK_NAMES` | Comma-separated task names that receive CE (default **`tfbind8,tfbind10`**). |

### Per-timestep diffusion loss during training (`diffuser/models/diffusion.py`)

When enabled, each training log step (same cadence as `log_freq`) includes extra scalars
``train/t_loss/b{bin}_t{lo}_{hi}``: mean **weighted MSE** (same weighting as the main diffusion loss)
over batch items whose sampled discrete timestep ``t`` falls in ``[lo, hi]`` (``t`` from
``torch.randint(0, n_timesteps, ...)``). **Smaller bin index → smaller ``t``** (less noise injected
in ``q_sample`` for the default beta schedule). Use these curves on wandb vs ``steps`` to see whether
the model is worse near ``t → 0`` (small noise) vs mid/high ``t``.

| Variable | Purpose |
|----------|---------|
| `DUO_LOG_PER_T_LOSS` | Set to `1` / `true` / `yes` / `on` to log per-bin diffusion MSE (default off). |
| `DUO_LOG_PER_T_LOSS_BINS` | Number of bins (default **`20`**, clamped to ``[2, n_timesteps]``). |

### Oracle sampling curves (`visualize.sh`)

Four eval modes (`mt_text`, `mt_task`, `st_text`, `st_duo`) write `eval_<tag>_seed<seed>.log` under `VISUALIZE_RESULTS`. If a log file already exists, that eval is **skipped** unless overridden below.

| Variable | Purpose |
|----------|---------|
| `PYTHON` | Interpreter (default conda path in script; must have PyTorch). |
| `PROJECT` | Repo root (default: directory containing `visualize.sh`). |
| `TRAJ_JSON` | Trajectory JSON for multitask `n_traj/k/eps` (default `examples/traj_params_per_task_example2.json`). |
| `FULL_MT` | Comma-separated `train_tasks` for multitask rows (default 9-task CSV). |
| `FRAC`, `SIGMA`, `HORIZON`, `CTX_LEN` | Passed through to `evaluate.py` (defaults `1.0`, `0.0`, `64`, `32`). |
| `N_TRAJ_MT`, `K_MT`, `EPS_MT` | Multitask trajectory scalars before per-task JSON overrides (defaults `1000`, `50`, `0.05`). |
| `MT_EXPECT_HEX` | Optional expected 16-hex segment inside printed multitask hyper dir (warning only). |
| `SAMPLE_VIZ_STRIDE`, `SAMPLE_VIZ_MAX_QUERIES` | Sample-viz stepping and query cap (defaults `10`, `512`). |
| `NUM_SEEDS`, `START_SEED` | Same semantics as `--multi_seeds` when exported before launch. |
| `VISUALIZE_RESULTS` | Output directory for `visualize.log` and `eval_*.log` (default `results/visualize/task_<TASK>_…`). |
| `DUMP_ROOT` | Multi-seed JSONL root (default `$VISUALIZE_RESULTS/sample_viz_dump`). |
| `WANDB_RUN_GROUP_PREFIX` | Prefix for `WANDB_RUN_GROUP` (default `duo_viz`). |
| `SKIP_MT_TEXT`, `SKIP_MT_TASK`, `SKIP_ST_TEXT`, `SKIP_ST_DUO` | Set to `1` to skip one of the four eval branches. |
| `VISUALIZE_FORCE_EVAL` | Set to `1` to **re-run** evaluate even when `eval_<tag>_seed*.log` already exists. |
| `EXTRA_EVAL_FLAGS` | Extra arguments forwarded to each `evaluate.py` invocation (space-separated). |

### Sample-viz JSONL (`evaluate.py`)

| Variable | Purpose |
|----------|---------|
| `DUO_SAMPLE_VIZ_DUMP_DIR` | Directory for per-step sample-viz JSONL dumps when not set via CLI `sample_viz_dump_jsonl`. |

---


## Multitask checkpoints

Training and evaluation share one directory: `trained_models/multi_<tasks>_frac.../` (no per-eval-task suffix). See `run_multitask.sh` comments for `EVAL_ALL` / `EVAL_ONLY`.

---

## Task text metadata

Short English blurbs per task for optional text conditioning: see `task_metadata/README.md`.

---

## Optional: SOO benchmarks

`thirdparty_benchmark/` and `bash scripts/setup_soo_bench.sh` — details in `thirdparty_benchmark/README.md`.

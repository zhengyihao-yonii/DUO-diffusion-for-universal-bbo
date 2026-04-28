from __future__ import annotations

import glob
import re

import diffuser.utils as utils
from ml_logger import logger
import torch
import math

from diffuser.cpu_threads import apply_torch_cpu_threads_from_env

apply_torch_cpu_threads_from_env()

import numpy as np
import os
import pickle as pkl
import json
from pathlib import Path
from typing import Any, Optional, Tuple

wandb = None
from diffuser.utils.arrays import to_torch

# Top-k oracle mean for eval / viz (fitting diagnostics: top16 vs max8 sample-viz).
_ORACLE_TOPK_MAX8 = 8
_ORACLE_TOPK_DIAG = 16


def _oracle_topk_mean(arr: np.ndarray, k: int) -> float:
    """Mean of the largest ``k`` finite values (or all values if fewer than ``k``)."""
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    kk = k if a.size >= k else int(a.size)
    return float(np.mean(np.sort(a)[-kk:]))
from diffuser.utils.des_bench import DesignBenchFunctionWrapper
from diffuser.models.vae import VAE
from sklearn.preprocessing import StandardScaler
from diffuser.datasets.sequence import MinimalTrajectoryDataset, PointRegretDataset
from diffuser.datasets.real_world_fewshot import (
    REAL_WORLD_FEWSHOT_TASK_SPECS,
    is_real_world_fewshot_task,
)
from diffuser.utils.proxy_filter import resolve_proxy_filter_for_eval
from diffuser.utils.real_world_oracle import oracle_predict
from diffuser.utils.vae_layout import (
    generated_vae_info_filename,
    multitask_generated_candidate_rel_dirs,
    resolve_generated_vae_info_path,
    resolve_multitask_generated_root_for_vae,
    resolve_vae_weights_path_for_eval,
)

# DUO repository root（用于解析 ``vae_info`` 内相对路径，与 cwd 无关）
_EVAL_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _eval_effective_latent_dim(deps: dict[str, Any] | None, config: Any) -> int:
    """
    Effective VAE / trajectory latent width for building the diffusion UNet.

    Anchor Config (e.g. dkitty_config) may not define ``latent_dim``; CLI still passes
    it in ``deps`` (``**vars(args)``). Prefer ``deps`` so zero-shot real_task matches
    a latent64 checkpoint and vae_info_latent64.p.
    """
    if deps and deps.get("latent_dim") is not None:
        return int(deps["latent_dim"])
    return int(getattr(config, "latent_dim", 32))


def _generated_datasets_vae_base(
    *,
    is_multitask: bool,
    train_tasks_list: list[str],
    eval_task: str,
    frac: float,
    sigma: float,
    fixed_dim: int = 128,
    latent_dim: int = 32,
) -> str:
    """
    与 ``construct_trajectories`` / ``data_path`` / ``train_vae`` 一致。
    多任务：优先 ``multi_*_dim{fixed}_latent{lat}``（latent≠32），再回退 ``multi_*_frac_sigma``。
    """
    from diffuser.utils.multitask_canon import canonical_train_tasks_csv

    if not is_multitask:
        return f"./generated_datasets/{eval_task}_frac{frac}_sigma{sigma}"
    csv = canonical_train_tasks_csv(",".join(train_tasks_list))
    return resolve_multitask_generated_root_for_vae(
        train_tasks_csv=csv,
        frac=float(frac),
        sigma=float(sigma),
        fixed_dim=int(fixed_dim),
        latent_dim=int(latent_dim),
    )


def _resolve_multitask_data_dir_for_mixed(
    *,
    train_tasks_list: list[str],
    config_data_path: str,
    sig: str | None,
    latent_dim: int,
    fixed_dim: int,
    frac: float,
    sigma: float,
) -> str:
    """
    解析含 mixed_mt_*.p 的目录：优先 ``Config.data_path`` 父目录，
    再尝试 ``_dim{fixed}_latent{lat}`` 后缀目录（与用户现有 generated_datasets 布局兼容）。
    """
    from diffuser.utils.multitask_canon import canonical_train_tasks_csv
    from diffuser.utils.traj_params import resolve_multitask_mixed_path

    primary = os.path.dirname(config_data_path)
    csv = canonical_train_tasks_csv(",".join(train_tasks_list))
    order: list[str] = [os.path.normpath(primary)]
    for rel in multitask_generated_candidate_rel_dirs(
        train_tasks_csv=csv,
        frac=float(frac),
        sigma=float(sigma),
        fixed_dim=int(fixed_dim),
        latent_dim=int(latent_dim),
    ):
        d = os.path.normpath("./" + rel)
        if d not in order:
            order.append(d)
    for d in order:
        try:
            resolve_multitask_mixed_path(d, sig, int(latent_dim))
            return d
        except FileNotFoundError:
            continue
    return primary


def _real_world_d_best_fewshot_params(
    eval_task: str,
    Config,
    is_multitask: bool,
    deps: dict[str, Any] | None = None,
) -> tuple[int | None, str, int]:
    """
    D(best) 与 construct 的 few-shot 池对齐：优先读 ``vae_info.p`` 中 construct 写入的字段。
    """
    fk: int | None = getattr(Config, "fewshot_k", None)
    fm = str(getattr(Config, "fewshot_mode", "all"))
    fseed: int | None = getattr(Config, "fewshot_seed", None)
    if not is_multitask and is_real_world_fewshot_task(eval_task):
        _ld = int(_eval_effective_latent_dim(deps, Config))
        _base = f"./generated_datasets/{eval_task}_frac{Config.frac}_sigma{Config.sigma}"
        _vip = resolve_generated_vae_info_path(_base, _ld)
        if _vip and os.path.isfile(_vip):
            with open(_vip, "rb") as f:
                _vi = pkl.load(f)
            if isinstance(_vi, dict) and "real_world_fewshot_seed" in _vi:
                fk = _vi.get("real_world_fewshot_k", fk)
                fm = str(_vi.get("real_world_fewshot_mode", fm))
                fseed = int(_vi["real_world_fewshot_seed"])
    if fseed is None:
        fseed = int(getattr(Config, "seed", 1))
    return fk, fm, fseed


def _real_world_proxy_fallback_dirs(logger_prefix: str) -> list[str]:
    """
    Zero-shot 评估的 RUN.prefix 含 ``_realtask_zs``，proxy 实际写在「同轨迹超参、曾跑过 construct/train」
    的目录下（无 ``_realtask_zs``，或为 ``_fewshot_ft``）。按序尝试这些父目录下的 ``proxy_checkpoint``。
    """
    base = logger_prefix.rstrip("/")
    out: list[str] = []
    seen: set[str] = set()

    def _add(p: str) -> None:
        if p not in seen:
            seen.add(p)
            out.append(p)

    _add(os.path.join(base, "proxy_checkpoint"))
    if "_realtask_zs" in base:
        _add(os.path.join(base.replace("_realtask_zs", "", 1), "proxy_checkpoint"))
        _add(os.path.join(base.replace("_realtask_zs", "_fewshot_ft", 1), "proxy_checkpoint"))
    return out


def _resolve_proxy_checkpoint_path(proxy_ckpt_dir, Config):
    """
    与 diffuser.utils.multitask_proxy_paths.multitask_proxy_checkpoint_exists 判定一致：
    存在任意 state_*.pt 即视为「有 checkpoint」；加载时优先 state.pt、state_{proxy_n_train_steps}.pt，
    否则取 state_*.pt 中训练步数最大的文件（避免仅存在 state_1000.pt 时评估仍找不到）。
    """
    if not os.path.isdir(proxy_ckpt_dir):
        return None
    st = os.path.join(proxy_ckpt_dir, "state.pt")
    if os.path.isfile(st):
        return st
    if getattr(Config, "save_checkpoints", False):
        n = int(getattr(Config, "proxy_n_train_steps", 5000))
        p = os.path.join(proxy_ckpt_dir, f"state_{n}.pt")
        if os.path.isfile(p):
            return p
        for alt in (5000, 10000, 1000, 3000):
            p2 = os.path.join(proxy_ckpt_dir, f"state_{alt}.pt")
            if os.path.isfile(p2):
                return p2
        matches = glob.glob(os.path.join(proxy_ckpt_dir, "state_*.pt"))
        if matches:

            def _step(path):
                m = re.search(r"state_(\d+)\.pt$", path)
                return int(m.group(1)) if m else -1

            return max(matches, key=_step)
    return None


def _resolve_diffusion_checkpoint_path(
    ckpt_dir: str,
    Config: Any,
    *,
    ckpt_train_steps: int | None = None,
    train_epochs: int | None = None,
) -> str | None:
    """Delegate to ``resolve_diffusion_state_pt`` (single source of truth)."""
    from diffuser.utils.real_task_transfer import resolve_diffusion_state_pt

    return resolve_diffusion_state_pt(
        ckpt_dir,
        Config,
        ckpt_train_steps=ckpt_train_steps,
        train_epochs=train_epochs,
    )


def _predict_y_arrays_from_queries_np(
    queries: np.ndarray,
    eval_task: str,
    func: DesignBenchFunctionWrapper,
) -> Tuple[np.ndarray, np.ndarray]:
    """与主评估路径一致：Oracle / task.predict → 原始 y 与 [0,1] 归一化 y。"""
    nq = int(queries.shape[0])
    if nq == 0:
        return np.array([]), np.array([])
    if eval_task.startswith("tfbind"):
        q = func.task.to_integers(queries.reshape(nq, -1, 3))
    else:
        q = queries.reshape(nq, -1)
    if is_real_world_fewshot_task(eval_task):
        _phys = int(REAL_WORLD_FEWSHOT_TASK_SPECS[eval_task]["dim"])
        if q.shape[-1] > _phys:
            q = np.asarray(q, dtype=np.float64)[:, :_phys]
        y = oracle_predict(eval_task, np.asarray(q, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).reshape(-1)
    else:
        y = np.asarray(func.task.predict(q), dtype=np.float64).reshape(-1)
    y_norm = (y - func.min) / (func.max - func.min)
    return y, y_norm


def _oracle_viz_stats_from_transition_x(
    x: torch.Tensor,
    *,
    trainer,
    dataset,
    Config,
    eval_task: str,
    context_length: int,
    observation_dim: int,
    original_observation_dim: int,
    latent_dim: int,
    vae,
    scaler,
    func: DesignBenchFunctionWrapper,
    max_queries: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float, float, float]:
    """
    将当前扩散状态 x（归一化轨迹）解码为设计空间，取 context 之后时间步上的点，
    调用 Oracle 得到一批 y，返回 mean/max 与 top-8 均值（原始与归一化；与 sample_viz 一致）。
    """
    samples = x[..., :observation_dim]
    ds_dev = samples.device
    if hasattr(dataset.normalizer, "maxs"):
        dataset.normalizer.maxs = dataset.normalizer.maxs.to(ds_dev)
        dataset.normalizer.mins = dataset.normalizer.mins.to(ds_dev)
    unnormalized_samples = dataset.normalizer.unnormalize(samples)
    batch_size, horizon, _ = unnormalized_samples.shape
    if context_length >= horizon:
        return (float("nan"),) * 6

    if vae is not None:
        with torch.no_grad():
            samples_flat = unnormalized_samples.reshape(-1, latent_dim)
            decoded_flat = vae.decode(samples_flat)
            if scaler is not None:
                n_sf = int(len(scaler.scale_))
                if decoded_flat.shape[1] < n_sf:
                    return (float("nan"),) * 6
                inv = scaler.inverse_transform(
                    decoded_flat[:, :n_sf].detach().cpu().numpy()
                )
                decoded_flat = decoded_flat.clone()
                decoded_flat[:, :n_sf] = torch.from_numpy(inv).to(
                    dtype=decoded_flat.dtype, device=decoded_flat.device
                )
            decoded_flat_truncated = decoded_flat[:, :original_observation_dim]
            decoded_samples = decoded_flat_truncated.reshape(
                batch_size, horizon, original_observation_dim
            )
            tail = decoded_samples[:, context_length:, :]
    else:
        tail = unnormalized_samples[:, context_length:, :]

    queries = tail.reshape(-1, tail.shape[-1]).detach().cpu().numpy()
    nq = queries.shape[0]
    if nq > max_queries:
        sel = rng.choice(nq, size=max_queries, replace=False)
        queries = queries[sel]

    y, y_norm = _predict_y_arrays_from_queries_np(queries, eval_task, func)
    if y.size == 0:
        return (float("nan"),) * 6

    return (
        float(np.mean(y)),
        float(np.max(y)),
        float(np.mean(y_norm)),
        float(np.max(y_norm)),
        _oracle_topk_mean(y, _ORACLE_TOPK_MAX8),
        _oracle_topk_mean(y_norm, _ORACLE_TOPK_MAX8),
    )


def _ensure_single_task_real_world_proxy(
    eval_task,
    Config,
    dataset,
    proxy_dataset,
    state_dict,
    renderer,
):
    """单任务真实任务：canonical 下无 proxy 时，用已加载的扩散权重训练并保存 proxy（对齐 train.ensure_multitask_proxies）。"""
    from diffuser.utils.multitask_proxy_paths import multitask_proxy_prefix

    use_task_branch = False
    observation_dim = dataset.observation_dim
    original_observation_dim = proxy_dataset.original_observation_dim
    action_dim = dataset.action_dim

    if Config.diffusion == "models.GaussianInvDynDiffusion":
        transition_dim = observation_dim
    else:
        transition_dim = observation_dim + action_dim

    proxy_prefix = multitask_proxy_prefix(eval_task, Config)
    os.makedirs(os.path.join(proxy_prefix, "proxy_checkpoint"), exist_ok=True)

    model_config = utils.Config(
        Config.model,
        savepath="model_config.pkl",
        horizon=Config.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        dim=Config.dim,
        returns_condition=Config.returns_condition,
        device=Config.device,
        task_condition=use_task_branch,
        num_tasks=1,
        condition_dropout=getattr(Config, "condition_dropout", 0.25),
        calc_energy=getattr(Config, "calc_energy", False),
        text_condition=getattr(Config, "use_text_condition", False),
        text_embed_input_dim=int(getattr(Config, "text_embed_dim", 384)),
        text_condition_dropout=float(
            getattr(Config, "text_condition_dropout", 0.1)
        ),
    )
    proxy_model_config = utils.Config(
        Config.proxy_model,
        savepath="proxy_model_config.pkl",
        input_dim=original_observation_dim,
        hidden_dim=Config.proxy_hidden_dim,
        output_dim=action_dim,
        n_ensembles=Config.proxy_n_ensembles,
        device=Config.device,
    )
    _diff_kw_rw = dict(
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        device=Config.device,
        condition_guidance_w=Config.condition_guidance_w,
        condition_guidance_w_task=float(getattr(Config, "condition_guidance_w_task", 0.0)),
        condition_guidance_w_text=float(getattr(Config, "condition_guidance_w_text", 0.0)),
        cfg_apply_task=bool(getattr(Config, "cfg_apply_task", True)),
        cfg_apply_text=bool(getattr(Config, "cfg_apply_text", True)),
        sample_with_task_embedding=bool(getattr(Config, "sample_with_task_embedding", True)),
        sample_with_text_embedding=bool(getattr(Config, "sample_with_text_embedding", True)),
    )
    _diff_kw_rw["train_timestep_bias_power"] = float(
        getattr(Config, "train_timestep_bias_power", 0.0)
    )
    _diff_kw_rw["train_loss_min_snr_gamma"] = float(
        getattr(Config, "train_loss_min_snr_gamma", 0.0)
    )
    if Config.diffusion != "models.GaussianInvDynDiffusion":
        _diff_kw_rw["n_sample_timesteps"] = int(
            getattr(Config, "n_sample_timesteps", Config.n_diffusion_steps)
        )
    diffusion_config = utils.Config(
        Config.diffusion,
        savepath="diffusion_config.pkl",
        **_diff_kw_rw,
    )
    Config.batch_size = 128
    trainer_config = utils.Config(
        utils.Trainer,
        savepath="trainer_config.pkl",
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        proxy_train_lr=Config.proxy_learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        proxy_save_freq=Config.proxy_save_freq,
        log_freq=Config.log_freq,
        proxy_log_freq=Config.proxy_log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
        save_checkpoints=Config.save_checkpoints,
        proxy_save_prefix=proxy_prefix,
    )

    model = model_config()
    proxy_model = proxy_model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, proxy_model, dataset, proxy_dataset, renderer)

    trainer.step = state_dict["step"]
    trainer.model.load_state_dict(state_dict["model"])
    trainer.ema_model.load_state_dict(state_dict["ema"])
    print(
        f"[real_world] 训练 proxy → {proxy_prefix}proxy_checkpoint/ "
        f"（{Config.proxy_n_train_steps} steps）",
        flush=True,
    )
    trainer.train_proxy(n_train_steps=Config.proxy_n_train_steps)


def _import_config(task_name):
    if task_name == 'ant':
        from config.ant_config import Config
    elif task_name == 'dkitty':
        from config.dkitty_config import Config
    elif task_name == 'tfbind8':
        from config.tfbind8_config import Config
    elif task_name == 'tfbind10':
        from config.tfbind10_config import Config
    elif task_name == 'superconductor':
        from config.superconductor_config import Config
    elif task_name in ('gtopx2', 'gtopx3', 'gtopx4', 'gtopx6'):
        from config.gtopx_config import Config
    elif task_name in ("lunar_lander", "robot_push", "rover"):
        from config.ant_config import Config
    else:
        print(f"警告: 未知的任务 {task_name}，使用默认 dkitty 配置")
        from config.dkitty_config import Config
    return Config


def _evaluate_single_task(
    eval_task: str,
    deps: dict[str, Any],
    state_dict: Optional[dict[str, Any]],
    Config: Any,
    log_wandb: bool = True,
    save_suffix: str = "",
) -> dict[str, Any]:
    """单次评估：加载 proxy、数据集、VAE，条件扩散采样并在 design-bench 上算 reward。

    Config 必须由 evaluate() 传入（已按 checkpoint 对应任务 _import_config + _update）。
    勿在循环内按 eval_task 再次 import 不同 config.*_config：params_proto 全局 ARGS 会重复注册 --seed 等参数并报错。
    """
    from ml_logger import logger

    Config.eval_task = eval_task
    Config.dataset = eval_task
    if 'train_tasks_list' in deps:
        Config.train_tasks_list = deps['train_tasks_list']
    is_multitask = deps.get('is_multitask', False)
    Config.is_multitask = is_multitask
    train_tasks_list = deps.get('train_tasks_list', [eval_task])
    use_task_branch = is_multitask and not getattr(Config, "multitask_text_only", False)
    use_proxy_filter = resolve_proxy_filter_for_eval(deps)
    setattr(Config, "proxy_filter", use_proxy_filter)
    # 真实任务 + --real_task_zero_shot_eval：固定为无轨迹上下文采样（不读 pkl、不注入 context 帧）
    rw_zs_no_ctx = bool(deps.get("real_task_zero_shot_eval")) and is_real_world_fewshot_task(
        eval_task
    )
    print(
        f"[evaluate] proxy_filter={int(use_proxy_filter)} "
        f"({'proxy 打分筛选 queries' if use_proxy_filter else '关闭：仅扩散采样后 eval'})",
        flush=True,
    )
    if rw_zs_no_ctx:
        print(
            "[evaluate] 真实任务 zero-shot：无轨迹 pkl 条件；"
            "ctx_len=0、不注入 context 帧（仅 task/text 等全局条件）",
            flush=True,
        )

    proxy_state_dict = None
    # 多任务：从单独路径加载该任务的 proxy（proxy_filter=0 时跳过）
    if is_multitask and use_proxy_filter:
        print(f"加载评估任务 {eval_task} 的 proxy model")
        nd = getattr(Config, "traj_n_traj_dict", None)
        if nd is not None and eval_task in nd:
            _n = nd[eval_task]
            _k = getattr(Config, "traj_k_dict", {})[eval_task]
            _e = getattr(Config, "traj_eps_dict", {})[eval_task]
        else:
            _n, _k, _e = Config.n_traj, Config.k, Config.eps
        proxy_model_prefix = f"trained_models/{eval_task}_frac{Config.frac}_sigma{Config.sigma}/{_n}x{Config.horizon}_k{_k}_eps{_e}/seed{Config.seed}/"
        proxy_ckpt_dir = os.path.join(proxy_model_prefix, "proxy_checkpoint")
        proxy_loadpath = _resolve_proxy_checkpoint_path(proxy_ckpt_dir, Config)
        if proxy_loadpath is not None:
            print(f"找到 proxy model: {proxy_loadpath}")
            proxy_state_dict = torch.load(proxy_loadpath, map_location=Config.device)
        else:
            print(f"警告: 未找到 {eval_task} 的 proxy，回退到 logger.prefix 下 proxy")
            fallback_dir = os.path.join(logger.prefix, "proxy_checkpoint")
            proxy_loadpath = _resolve_proxy_checkpoint_path(fallback_dir, Config)
            if proxy_loadpath is None:
                raise FileNotFoundError(
                    f"无法加载 proxy：既不在 {proxy_ckpt_dir}，也不在 {fallback_dir}。"
                    f"多任务需先为各任务训练 proxy（trained_models/<task>_.../seed{Config.seed}/proxy_checkpoint/state_*.pt）。"
                )
            proxy_state_dict = torch.load(proxy_loadpath, map_location=Config.device)

    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)

    # 轨迹数据路径：多任务使用混合 pkl（与 train 一致）
    if rw_zs_no_ctx:
        _ld_zs = int(_eval_effective_latent_dim(deps, Config))
        _fd_zs = int(getattr(Config, "fixed_dim", 128))
        _vae_base_early = _generated_datasets_vae_base(
            is_multitask=is_multitask,
            train_tasks_list=train_tasks_list,
            eval_task=eval_task,
            frac=float(Config.frac),
            sigma=float(Config.sigma),
            fixed_dim=_fd_zs,
            latent_dim=_ld_zs,
        )
        vae_info_early = resolve_generated_vae_info_path(_vae_base_early, _ld_zs)
        latent_obs = int(_eval_effective_latent_dim(deps, Config))
        if vae_info_early and os.path.isfile(vae_info_early):
            with open(vae_info_early, "rb") as f:
                _vi = pkl.load(f)
            latent_obs = int(_vi.get("latent_dim", latent_obs))
        dataset = MinimalTrajectoryDataset(
            Config.horizon,
            observation_dim=latent_obs,
            action_dim=1,
            n_fake=256,
            seed=int(Config.seed),
        )
        logger.print(
            f"[evaluate] MinimalTrajectoryDataset obs_dim={latent_obs}（vae_info 若存在则覆盖；否则用 CLI/config 有效 latent）",
            color="cyan",
        )
    elif is_multitask:
        from diffuser.utils.traj_params import (
            ensure_multitask_mixed_trajectories,
            multitask_mixed_basename,
            resolve_multitask_mixed_path,
        )

        _sig = getattr(Config, "multitask_traj_signature", None)
        _skip_auto = bool(deps.get("skip_auto_construct_trajectories", False))
        _latent = int(_eval_effective_latent_dim(deps, Config))
        _fd_mt = int(getattr(Config, "fixed_dim", 128))
        data_dir = _resolve_multitask_data_dir_for_mixed(
            train_tasks_list=list(train_tasks_list),
            config_data_path=str(Config.data_path),
            sig=_sig,
            latent_dim=_latent,
            fixed_dim=_fd_mt,
            frac=float(Config.frac),
            sigma=float(Config.sigma),
        )
        try:
            ensure_multitask_mixed_trajectories(
                train_tasks_list=list(train_tasks_list),
                frac=float(Config.frac),
                sigma=float(Config.sigma),
                seed=int(Config.seed),
                n_traj=int(Config.n_traj),
                k=int(Config.k),
                eps=float(Config.eps),
                horizon=int(Config.horizon),
                traj_params_json=deps.get("traj_params_json"),
                fixed_dim=_fd_mt,
                skip_auto=_skip_auto,
                latent_dim=_latent,
            )
            mixed_data_path = resolve_multitask_mixed_path(data_dir, _sig, _latent)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{e}\n请先运行 construct_trajectories.py 生成混合轨迹，或去掉 --skip_auto_construct_trajectories。"
            ) from e
        dataset = PointRegretDataset(
            horizon=Config.horizon,
            data_path=mixed_data_path,
            context_length=Config.context_length,
            regret=Config.regret,
            include_returns=Config.include_returns,
            task_name=None,
            task_text_embeds=getattr(Config, "_task_text_embeds_np", None),
            include_task_idx=use_task_branch,
        )
        _ld_expect = int(_eval_effective_latent_dim(deps, Config))
        if int(dataset.observation_dim) != _ld_expect:
            _expect_name = (
                multitask_mixed_basename(str(_sig), _ld_expect) if _sig else "mixed_*.p"
            )
            raise RuntimeError(
                f"[{eval_task}] 混合轨迹 PKL 中 observation 宽度={dataset.observation_dim}，"
                f"与 Config.latent_dim={_ld_expect} 不一致。"
                f"前者由磁盘 mixed 文件决定，后者为 CLI 超参；二者必须相同。"
                f"期望文件名示例: {_expect_name}；若曾误将 32 维 mixed 保存为该名，请删后重建 construct_trajectories.py --latent_dim {_ld_expect}。"
            )
    else:
        if not os.path.isfile(Config.data_path):
            _rw = is_real_world_fewshot_task(eval_task)
            _extra = ""
            if _rw:
                _extra = (
                    " 真实任务须先用多任务 vae_info 跑 construct_trajectories.py（--pretrained_vae_info）；"
                    "run_real_tasks.sh zero_shot 会在缺文件时自动 construct。"
                )
            raise FileNotFoundError(
                f"未找到轨迹文件: {Config.data_path}\n"
                f"请先运行 construct_trajectories.py，参数与当前 evaluate 一致（n_traj/k/eps/horizon/frac/sigma）。{_extra}"
            )
        dataset_config = utils.Config(
            Config.loader,
            savepath='dataset_config.pkl',
            horizon=Config.horizon,
            data_path=Config.data_path,
            context_length=Config.context_length,
            regret=Config.regret,
            include_returns=Config.include_returns,
            task_text_embeds=getattr(Config, "_task_text_embeds_np", None),
        )
        dataset = dataset_config()

    if rw_zs_no_ctx:

        class _ProxyDimOnly:
            __slots__ = ("original_observation_dim",)

            def __init__(self, dim: int):
                self.original_observation_dim = int(dim)

        proxy_dataset_full = _ProxyDimOnly(
            REAL_WORLD_FEWSHOT_TASK_SPECS[eval_task]["dim"],
        )
    else:
        proxy_dataset_config = utils.Config(
            Config.proxy_loader,
            dataset=eval_task,
            frac=Config.frac,
            sigma=Config.sigma,
            soo_seed=int(getattr(Config, 'seed', 1)),
            savepath='proxy_dataset_config.pkl',
        )
        proxy_dataset_full = proxy_dataset_config()
    renderer = Config.renderer
    observation_dim = dataset.observation_dim
    original_observation_dim = proxy_dataset_full.original_observation_dim
    action_dim = dataset.action_dim
    proxy_dataset = proxy_dataset_full if use_proxy_filter else None
    print(f"[{eval_task}] observation_dim={observation_dim}, action_dim={action_dim}")

    if not is_multitask and use_proxy_filter:
        from diffuser.utils.multitask_proxy_paths import multitask_proxy_prefix

        proxy_loadpath = None
        proxy_ckpt_dir = None
        if is_real_world_fewshot_task(eval_task):
            for d in _real_world_proxy_fallback_dirs(logger.prefix):
                proxy_loadpath = _resolve_proxy_checkpoint_path(d, Config)
                if proxy_loadpath is not None:
                    proxy_ckpt_dir = d
                    print(
                        f"[real_world] proxy 从 fallback 加载: {proxy_loadpath}",
                        flush=True,
                    )
                    break
            if proxy_loadpath is None:
                canon_dir = os.path.join(
                    multitask_proxy_prefix(eval_task, Config), "proxy_checkpoint"
                )
                proxy_loadpath = _resolve_proxy_checkpoint_path(canon_dir, Config)
                if proxy_loadpath is not None:
                    proxy_ckpt_dir = canon_dir
                    print(
                        f"[real_world] proxy 从 canonical 路径加载: {proxy_loadpath}",
                        flush=True,
                    )
        if proxy_loadpath is None:
            proxy_ckpt_dir = os.path.join(logger.prefix, "proxy_checkpoint")
            proxy_loadpath = _resolve_proxy_checkpoint_path(proxy_ckpt_dir, Config)
        if proxy_loadpath is None and is_real_world_fewshot_task(eval_task):
            _ensure_single_task_real_world_proxy(
                eval_task,
                Config,
                dataset,
                proxy_dataset,
                state_dict,
                renderer,
            )
            canon_dir = os.path.join(
                multitask_proxy_prefix(eval_task, Config), "proxy_checkpoint"
            )
            proxy_loadpath = _resolve_proxy_checkpoint_path(canon_dir, Config)
            proxy_ckpt_dir = canon_dir
        if proxy_loadpath is None:
            raise FileNotFoundError(
                f"无法加载 proxy：{proxy_ckpt_dir} 下无 state.pt 或 state_*.pt"
                + (
                    "（真实任务可依赖自动训练；若仍失败请检查数据与权限）"
                    if is_real_world_fewshot_task(eval_task)
                    else ""
                )
            )
        proxy_state_dict = torch.load(proxy_loadpath, map_location=Config.device)

    # 与 ZipDataset 一致的离线训练子集上 y 的最优值（真实任务 few-shot：池内 max(y)，见 vae_info 元数据）
    from diffuser.utils.offline_train_best import offline_training_best_y

    _rw_fk, _rw_fm, _rw_fs = _real_world_d_best_fewshot_params(
        eval_task, Config, is_multitask, deps=deps
    )
    _y_tr_best = offline_training_best_y(
        eval_task,
        frac=float(Config.frac),
        sigma=float(Config.sigma),
        seed=int(getattr(Config, "seed", 1)),
        real_world_fewshot_k=_rw_fk,
        real_world_fewshot_mode=_rw_fm,
        real_world_fewshot_seed=_rw_fs,
    )
    logger.print(
        f"[{eval_task}] offline_train_best_y: {_y_tr_best}",
        color="green",
    )
    if is_real_world_fewshot_task(eval_task):
        from diffuser.utils.offline_train_best import (
            offline_full_dataset_best_y,
            offline_full_dataset_y_bounds,
        )

        _y_lo_full, _y_hi_full = offline_full_dataset_y_bounds(
            eval_task,
            frac=float(Config.frac),
            sigma=float(Config.sigma),
            seed=int(getattr(Config, "seed", 1)),
        )
        _y_all_best = offline_full_dataset_best_y(
            eval_task,
            frac=float(Config.frac),
            sigma=float(Config.sigma),
            seed=int(getattr(Config, "seed", 1)),
        )
        logger.print(
            f"[{eval_task}] real_world_fewshot_pool: k={_rw_fk!r} mode={_rw_fm} seed={_rw_fs}",
            color="green",
        )
        logger.print(
            f"[{eval_task}] offline_dataset_y_bounds_full: {_y_lo_full}/{_y_hi_full}",
            color="green",
        )
        logger.print(
            f"[{eval_task}] offline_dataset_best_y_all: {_y_all_best}",
            color="green",
        )

    vae = None
    latent_dim = observation_dim
    vae_input_output_dim = getattr(Config, 'fixed_dim', 128)

    _ld_vae = int(_eval_effective_latent_dim(deps, Config))
    _fd_vae = int(getattr(Config, "fixed_dim", 128))
    _vae_base = _generated_datasets_vae_base(
        is_multitask=is_multitask,
        train_tasks_list=train_tasks_list,
        eval_task=eval_task,
        frac=float(Config.frac),
        sigma=float(Config.sigma),
        fixed_dim=_fd_vae,
        latent_dim=_ld_vae,
    )
    vae_info_path = resolve_generated_vae_info_path(_vae_base, _ld_vae)

    scaler = None
    if vae_info_path and os.path.exists(vae_info_path):
        try:
            with open(vae_info_path, 'rb') as f:
                vae_info = pkl.load(f)
            _sig_mt = getattr(Config, "multitask_traj_signature", None)
            _raw_vp = vae_info.get("vae_path") or vae_info.get("model_path")
            _latent_from_info = int(vae_info.get("latent_dim", _ld_vae))
            latent_dim = _latent_from_info
            vae_path_resolved, _vae_tried = resolve_vae_weights_path_for_eval(
                raw_path=_raw_vp if _raw_vp else None,
                vae_info_path=vae_info_path,
                latent_dim=_latent_from_info,
                project_root=str(_EVAL_PROJECT_ROOT),
                multitask_traj_signature=_sig_mt,
            )
            if vae_path_resolved is None:
                raise FileNotFoundError(
                    "无法在下列路径中找到 VAE 权重文件（末尾为 vae_latent*.pt）；"
                    f"tried={_vae_tried}"
                )
            _intended_norm: str | None = None
            if _raw_vp:
                _intended_norm = (
                    os.path.normpath(_raw_vp)
                    if os.path.isabs(_raw_vp)
                    else os.path.normpath(
                        os.path.join(str(_EVAL_PROJECT_ROOT), _raw_vp)
                    )
                )
            if (
                _intended_norm is not None
                and os.path.normpath(vae_path_resolved) != _intended_norm
            ):
                logger.print(
                    f"[evaluate] VAE 权重从回退路径加载（mt 目录段或拷贝位置与 vae_info 记录不一致）: "
                    f"{vae_path_resolved}",
                    color="yellow",
                )
            vae_path = vae_path_resolved
            vae = VAE(input_dim=vae_input_output_dim, latent_dim=latent_dim)
            vae.load_state_dict(torch.load(vae_path, map_location=Config.device))
            vae.to(Config.device)
            vae.eval()
            model_save_dir = os.path.dirname(vae_path)
            aux_scaler = vae_info.get("fewshot_real_world_scaler_path")
            if aux_scaler and os.path.isfile(aux_scaler):
                scaler_path = aux_scaler
            else:
                scaler_path = os.path.join(model_save_dir, f"scaler_{eval_task}.p")
            if os.path.exists(scaler_path):
                scaler = StandardScaler()
                scaler_dict = pkl.load(open(scaler_path, "rb"))
                scaler.mean_ = scaler_dict['mean']
                scaler.scale_ = scaler_dict['scale']
                scaler.n_features_in_ = len(scaler.mean_)
        except Exception as e:
            print(f"加载 VAE 出错: {e}")
            vae = None
    elif not vae_info_path or not os.path.exists(vae_info_path):
        print(
            f"[{eval_task}] 未找到 VAE 元数据: 尝试路径前缀 {_vae_base!r}，"
            f"latent_dim={_ld_vae}（期望 {generated_vae_info_filename(_ld_vae)}）",
            flush=True,
        )

    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        transition_dim = observation_dim
    else:
        transition_dim = observation_dim + action_dim

    num_tasks = len(train_tasks_list) if is_multitask else 1
    model_config = utils.Config(
        Config.model,
        savepath='model_config.pkl',
        horizon=Config.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        dim=Config.dim,
        returns_condition=Config.returns_condition,
        device=Config.device,
        task_condition=use_task_branch,
        num_tasks=num_tasks,
        condition_dropout=getattr(Config, 'condition_dropout', 0.25),
        calc_energy=getattr(Config, 'calc_energy', False),
        text_condition=getattr(Config, "use_text_condition", False),
        text_embed_input_dim=int(getattr(Config, "text_embed_dim", 384)),
        text_condition_dropout=float(
            getattr(Config, "text_condition_dropout", 0.1)
        ),
    )

    _diff_kw_eval = dict(
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        device=Config.device,
        condition_guidance_w=Config.condition_guidance_w,
        condition_guidance_w_task=float(getattr(Config, "condition_guidance_w_task", 0.0)),
        condition_guidance_w_text=float(getattr(Config, "condition_guidance_w_text", 0.0)),
        cfg_apply_task=bool(getattr(Config, "cfg_apply_task", True)),
        cfg_apply_text=bool(getattr(Config, "cfg_apply_text", True)),
        sample_with_task_embedding=bool(getattr(Config, "sample_with_task_embedding", True)),
        sample_with_text_embedding=bool(getattr(Config, "sample_with_text_embedding", True)),
    )
    _diff_kw_eval["train_timestep_bias_power"] = float(
        getattr(Config, "train_timestep_bias_power", 0.0)
    )
    _diff_kw_eval["train_loss_min_snr_gamma"] = float(
        getattr(Config, "train_loss_min_snr_gamma", 0.0)
    )
    if Config.diffusion != "models.GaussianInvDynDiffusion":
        _diff_kw_eval["n_sample_timesteps"] = int(
            getattr(Config, "n_sample_timesteps", Config.n_diffusion_steps)
        )
    diffusion_config = utils.Config(
        Config.diffusion,
        savepath='diffusion_config.pkl',
        **_diff_kw_eval,
    )

    Config.batch_size = 128
    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        proxy_train_lr=Config.proxy_learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        proxy_save_freq=Config.proxy_save_freq,
        log_freq=Config.log_freq,
        proxy_log_freq=Config.proxy_log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
    )

    model = model_config()
    if use_proxy_filter:
        proxy_model_config = utils.Config(
            Config.proxy_model,
            savepath='proxy_model_config.pkl',
            input_dim=original_observation_dim,
            hidden_dim=Config.proxy_hidden_dim,
            output_dim=action_dim,
            n_ensembles=Config.proxy_n_ensembles,
            device=Config.device,
        )
        proxy_model = proxy_model_config()
    else:
        proxy_model = None
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, proxy_model, dataset, proxy_dataset, renderer)
    logger.print(utils.report_parameters(model), color='green')

    # 随机权重：跳过 checkpoint，用当前网络结构的默认初始化（并同步 EMA），用于验证推理链
    if state_dict is None:
        if not bool(deps.get("random_diffusion_weights")):
            raise RuntimeError("state_dict is None but random_diffusion_weights is not set")
        trainer.step = 0
        trainer.reset_parameters()
        logger.print(
            "[evaluate] diffusion: random init (no checkpoint load); inference sanity check",
            color="yellow",
        )
    else:
        trainer.step = state_dict["step"]
        trainer.model.load_state_dict(state_dict["model"])
        trainer.ema_model.load_state_dict(state_dict["ema"])
    if use_proxy_filter:
        trainer.proxy_step = proxy_state_dict['step']
        trainer.proxy_model.load_state_dict(proxy_state_dict['model'])
    if vae is not None:
        trainer.vae = vae

    device = Config.device
    context_length = getattr(Config, 'ctx_len', deps.get('ctx_len', 32))
    num_queries = 128
    num_eval = 1

    task_label_idx = 0
    if is_multitask and hasattr(dataset, "tasks_list") and dataset.tasks_list:
        if eval_task in dataset.tasks_list:
            task_label_idx = dataset.tasks_list.index(eval_task)
    elif is_multitask and train_tasks_list and eval_task in train_tasks_list:
        task_label_idx = train_tasks_list.index(eval_task)

    _sv_tag = str(deps.get("sample_viz_tag") or deps.get("sample_viz_run_tag") or "viz")
    _sv_stride = int(deps.get("sample_viz_stride", 10))
    _sv_maxq = int(deps.get("sample_viz_max_queries", 512))

    _sv_dump_dir = deps.get("sample_viz_dump_jsonl") or os.environ.get(
        "DUO_SAMPLE_VIZ_DUMP_DIR", ""
    )
    _sv_dump_dir = str(_sv_dump_dir).strip() if _sv_dump_dir else ""

    sample_viz_active = Config.diffusion == "models.GaussianDiffusion" and (
        (
            bool(deps.get("sample_viz_wandb", False))
            and log_wandb
            and wandb is not None
        )
        or bool(_sv_dump_dir)
    )
    if (
        bool(deps.get("sample_viz_wandb", False))
        or bool(_sv_dump_dir)
    ) and Config.diffusion != "models.GaussianDiffusion":
        logger.print(
            f"[sample_viz] 仅支持 GaussianDiffusion，当前为 {Config.diffusion}，跳过",
            color="yellow",
        )
        sample_viz_active = False

    if bool(deps.get("sample_viz_wandb", False)) and not (
        log_wandb and wandb is not None
    ) and not _sv_dump_dir:
        logger.print(
            "[sample_viz] 无 wandb 且未设置 --sample_viz_dump_jsonl，跳过曲线",
            color="yellow",
        )
        sample_viz_active = False

    func_viz = None
    viz_rng = None
    viz_step = [0]

    if sample_viz_active:
        func_viz = DesignBenchFunctionWrapper(
            eval_task, normalise=True, soo_seed=int(getattr(Config, "seed", 1))
        )
        viz_rng = np.random.default_rng(int(getattr(Config, "seed", 0)))

        _dump_base = Path(_sv_dump_dir) if _sv_dump_dir else None
        if _dump_base is not None:
            _dump_base.mkdir(parents=True, exist_ok=True)
        _wiz = (
            bool(deps.get("sample_viz_wandb", False))
            and log_wandb
            and wandb is not None
        )

        def _sample_viz_cb(timestep_index, step_ordinal, total_steps, x):
            m, mx, nm, nmx, m8, nm8 = _oracle_viz_stats_from_transition_x(
                x,
                trainer=trainer,
                dataset=trainer.dataset,
                Config=Config,
                eval_task=eval_task,
                context_length=int(context_length),
                observation_dim=int(observation_dim),
                original_observation_dim=int(original_observation_dim),
                latent_dim=int(latent_dim),
                vae=vae,
                scaler=scaler,
                func=func_viz,
                max_queries=_sv_maxq,
                rng=viz_rng,
            )
            _prog = float(step_ordinal) / max(1, int(total_steps) - 1)
            _step_i = int(viz_step[0])
            if _wiz:
                wandb.log(
                    {
                        "sample_viz_step": _step_i,
                        f"sample_viz/{_sv_tag}/mean_y": m,
                        f"sample_viz/{_sv_tag}/max_y": mx,
                        f"sample_viz/{_sv_tag}/top8_mean": m8,
                        f"sample_viz/{_sv_tag}/mean_y_norm": nm,
                        f"sample_viz/{_sv_tag}/max_y_norm": nmx,
                        f"sample_viz/{_sv_tag}/top8_mean_norm": nm8,
                        f"sample_viz/{_sv_tag}/t_index": int(timestep_index),
                        f"sample_viz/{_sv_tag}/denoise_progress": _prog,
                    },
                )
            if _dump_base is not None:
                _dj = (
                    _dump_base
                    / f"{_sv_tag}_seed{int(getattr(Config, 'seed', 0))}.jsonl"
                )
                rec = {
                    "viz_step": _step_i,
                    "mean_y": m,
                    "max_y": mx,
                    "top8_mean": m8,
                    "mean_y_norm": nm,
                    "max_y_norm": nmx,
                    "top8_mean_norm": nm8,
                    "t_index": int(timestep_index),
                    "denoise_progress": _prog,
                }
                with open(_dj, "a", encoding="utf-8") as _jf:
                    _jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            viz_step[0] += 1

        extra_sample_kwargs = {
            "step_callback": _sample_viz_cb,
            "step_callback_stride": _sv_stride,
        }
    else:
        extra_sample_kwargs = {}

    contexts = []
    queries = []
    for e in range(num_eval):
        if rw_zs_no_ctx:
            _bs = int(Config.batch_size)
            conditions = {
                "ctx_len": torch.zeros(_bs, dtype=torch.long, device=device),
            }
            if use_task_branch:
                conditions["task_idx"] = torch.full(
                    (_bs,), task_label_idx, dtype=torch.long, device=device
                )
            if getattr(Config, "use_text_condition", False) and hasattr(
                Config, "_task_text_embeds_np"
            ):
                _ti = task_label_idx if is_multitask else 0
                _te = torch.from_numpy(Config._task_text_embeds_np[_ti]).float().to(device)
                conditions["text_embed"] = _te.unsqueeze(0).expand(_bs, -1)
        else:
            batch = next(trainer.dataloader)
            _bs = int(batch.trajectories.shape[0])
            conditions = {
                i: to_torch(
                    batch.trajectories[:, i + Config.horizon - context_length],
                    device=device,
                )
                for i in range(context_length)
            }
            conditions["ctx_len"] = to_torch(np.ones(_bs,), device=device) * context_length
            if use_task_branch:
                conditions["task_idx"] = torch.full(
                    (_bs,), task_label_idx, dtype=torch.long, device=device
                )
            if getattr(Config, "use_text_condition", False) and hasattr(
                Config, "_task_text_embeds_np"
            ):
                _ti = task_label_idx if is_multitask else 0
                _te = torch.from_numpy(Config._task_text_embeds_np[_ti]).float().to(device)
                conditions["text_embed"] = _te.unsqueeze(0).expand(_bs, -1)

        # ------------------------------------------------------------------
        # 条件于「目标函数值 y」的两种含义（请勿混淆）：
        #
        # (A) 轨迹状态里的 y 通道（默认实验：含「只标签 / 标签+文本」等多任务设置）
        #     batch.trajectories[..., -1] 为归一化后的 objective（construct 中每任务已映到
        #     [0,1]；PointRegretDataset 再按 [0,1] 做 normalizer）。context 条件通过上面
        #     conditions[0..ctx_len-1] 注入前 context_length 步，其中已包含各步的 y。
        #
        # (B) 显式 returns 条件（经典「再条件于标量 return / test_ret」的 GTG 分支）
        #     仅当 TemporalUnet.returns_condition=True（Config.returns_condition）时，
        #     下面的 `returns` 才会进入 returns_mlp；否则传入仍计算但网络不读入（与多任务
        #     是否加 task/text 无关）。若需 (B)，训练侧还需 include_returns=True 等一致配置。
        #
        # Config.alpha：仅当 (B) 开启时作为标量条件强度；默认 gtopx 配置下 returns_condition=False，
        # 文件名里仍带 alpha 仅为记录 CLI，不改变默认 UNet 前向。
        # ------------------------------------------------------------------
        returns = torch.ones(1, ).to(device=device).unsqueeze(0) * Config.alpha
        returns = returns.repeat(_bs, 1)

        logger.print(
            "[eval] conditional_sample (diffusion) starting…",
            color="cyan",
        )
        if sample_viz_active:
            _dst = []
            if _wiz:
                _dst.append("wandb")
            if _dump_base is not None:
                _dst.append(f"dump={_dump_base}")
            logger.print(
                f"[sample_viz] stride={_sv_stride} tag={_sv_tag} max_queries={_sv_maxq} → "
                + ", ".join(_dst),
                color="cyan",
            )
        samples, time = trainer.ema_model.conditional_sample(
            conditions, values=None, returns=returns, **extra_sample_kwargs
        )
        samples = samples[..., :observation_dim]

        if vae is not None:
            with torch.no_grad():
                samples_device = samples.device
                if hasattr(trainer.dataset.normalizer, 'maxs'):
                    trainer.dataset.normalizer.maxs = trainer.dataset.normalizer.maxs.to(samples_device)
                    trainer.dataset.normalizer.mins = trainer.dataset.normalizer.mins.to(samples_device)
                unnormalized_samples = trainer.dataset.normalizer.unnormalize(samples)
                batch_size, horizon, _ = unnormalized_samples.shape
                samples_flat = unnormalized_samples.reshape(-1, latent_dim)
                decoded_flat = vae.decode(samples_flat)
                if scaler is not None:
                    # scaler 在 VAE 训练时按 fixed_dim（与 scaler.pkl 一致，常为 128）拟合；
                    # 须在截断到真实任务物理维（如 robot_push=14）之前 inverse_transform。
                    n_sf = int(len(scaler.scale_))
                    if decoded_flat.shape[1] < n_sf:
                        raise ValueError(
                            f"解码维数 {decoded_flat.shape[1]} < scaler 维数 {n_sf}，无法 inverse_transform"
                        )
                    inv = scaler.inverse_transform(
                        decoded_flat[:, :n_sf].detach().cpu().numpy()
                    )
                    decoded_flat = decoded_flat.clone()
                    decoded_flat[:, :n_sf] = torch.from_numpy(inv).to(
                        dtype=decoded_flat.dtype, device=decoded_flat.device
                    )
                decoded_flat_truncated = decoded_flat[:, :original_observation_dim]
                decoded_samples = decoded_flat_truncated.reshape(
                    batch_size, horizon, original_observation_dim
                )
                samples = decoded_samples

        tail = samples[:, context_length:]
        # 不用 proxy 时：不再“取前 N 个点”，而是优先取每条轨迹末尾的若干点，
        # 使 batch_size * k ≈ num_queries（通常 128）。若超过则后续随机采样到 num_queries。
        if not use_proxy_filter:
            bs = int(tail.shape[0])
            tl = int(tail.shape[1])
            k_need = int(math.ceil(float(num_queries) / max(1, bs)))
            k = max(1, min(tl, k_need))
            tail = tail[:, -k:, :]
        queries.append(tail)
        contexts.append(samples[:, :context_length])

    queries = torch.cat(queries, dim=0).reshape(-1, original_observation_dim if vae is not None else observation_dim)
    contexts = torch.cat(contexts, dim=0).reshape(-1, original_observation_dim if vae is not None else observation_dim).cpu().numpy()
    queries_cpu = queries.cpu()

    if use_proxy_filter:
        if vae is not None:
            queries_norm = trainer.proxy_dataset.normalizer.normalize(queries_cpu.to('cpu'))
        else:
            if int(observation_dim) != int(original_observation_dim):
                _ld_err = int(_eval_effective_latent_dim(deps, Config))
                raise RuntimeError(
                    f"[{eval_task}] proxy 需要物理维 {original_observation_dim}，"
                    f"但轨迹/PKL 给出的扩散状态维 observation_dim={observation_dim}（每步向量宽度），"
                    f"且未加载 VAE 解码。latent_dim(有效)={_ld_err}；"
                    f"若 observation_dim 与之不符，说明 mixed PKL 与 latent_dim 不匹配。"
                    f"请确认 {generated_vae_info_filename(_ld_err)} 存在于 generated_datasets/multi_<canonical>_frac…/ 且 vae_path 可读。"
                )
            queries_unnorm = trainer.dataset.normalizer.unnormalize(queries_cpu)
            if torch.is_tensor(queries_unnorm):
                _qin = queries_unnorm.detach().clone().to(
                    dtype=torch.float32, device="cpu"
                )
            else:
                _qin = torch.tensor(
                    np.asarray(queries_unnorm),
                    dtype=torch.float32,
                    device="cpu",
                )
            queries_norm = trainer.proxy_dataset.normalizer.normalize(_qin)
        queries_norm = queries_norm.to(trainer.device)
        queries_proxy_score = trainer.proxy_model(queries_norm).flatten()
        queries = queries[torch.argsort(queries_proxy_score)[-num_queries:]].cpu()
    else:
        # 从“每条轨迹末尾 k 个点”构成的候选池中随机选 num_queries 个（若不足则全取）
        q = queries_cpu
        n = int(q.shape[0])
        if n > num_queries:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(getattr(Config, "seed", 0)))
            idx = torch.randperm(n, generator=gen)[: int(num_queries)]
            queries = q[idx]
        else:
            queries = q

    if vae is None:
        queries = dataset.normalizer.unnormalize(queries).numpy()
    else:
        queries = queries.numpy()

    nq = int(queries.shape[0])
    func = DesignBenchFunctionWrapper(
        eval_task, normalise=True, soo_seed=int(getattr(Config, 'seed', 1))
    )
    if eval_task.startswith("tfbind"):
        queries = func.task.to_integers(queries.reshape(nq, -1, 3))
    else:
        queries = queries.reshape(nq, -1)
    # ZipDataset 将 real-world 设计填充到 fixed_length=128，proxy 用满维；Oracle 仍按物理维（如 rover=60）
    if is_real_world_fewshot_task(eval_task):
        _phys = int(REAL_WORLD_FEWSHOT_TASK_SPECS[eval_task]["dim"])
        if queries.shape[-1] > _phys:
            queries = np.asarray(queries, dtype=np.float64)[:, :_phys]
        logger.print(
            f"[{eval_task}] oracle_predict: {nq} designs (NumPy/SciPy sim, CPU-only; may take minutes)",
            color="cyan",
        )
        y = oracle_predict(eval_task, np.asarray(queries, dtype=np.float64))
    else:
        y = func.task.predict(queries)
    if is_real_world_fewshot_task(eval_task):
        from diffuser.utils.offline_train_best import offline_full_dataset_y_bounds

        y_lo_f, y_hi_f = offline_full_dataset_y_bounds(
            eval_task,
            frac=float(Config.frac),
            sigma=float(Config.sigma),
            seed=int(getattr(Config, "seed", 1)),
        )
        span = float(y_hi_f) - float(y_lo_f)
        y_arr = np.asarray(y, dtype=np.float64)
        if span <= 1e-20:
            y_norm = np.zeros_like(y_arr, dtype=np.float64)
        else:
            y_norm = (y_arr - y_lo_f) / span
    else:
        y_norm = (y - func.min) / (func.max - func.min)

    y_vec = np.asarray(y, dtype=np.float64).reshape(-1)
    y_norm_vec = np.asarray(y_norm, dtype=np.float64).reshape(-1)
    top16_mean = _oracle_topk_mean(y_vec, _ORACLE_TOPK_DIAG)
    ntop16_mean = _oracle_topk_mean(y_norm_vec, _ORACLE_TOPK_DIAG)

    logger.print(
        f"[{eval_task}] max_ep_reward: {np.max(y)}, median: {np.median(y)}, mean: {np.mean(y)}",
        color='green',
    )
    logger.print(
        f"[{eval_task}] top16_mean (raw oracle, fitting diag): {top16_mean}",
        color='green',
    )
    logger.print(
        f"[{eval_task}] nmax_ep_reward: {np.max(y_norm)}, nmedian: {np.median(y_norm)}, nmean: {np.mean(y_norm)}",
        color='green',
    )
    logger.print(
        f"[{eval_task}] ntop16_mean (normalized oracle, fitting diag): {ntop16_mean}",
        color='green',
    )
    logger.log_metrics_summary({
        f"max_ep_reward_{eval_task}": np.max(y),
        f"median_ep_reward_{eval_task}": np.median(y),
        f"mean_ep_reward_{eval_task}": np.mean(y),
        f"top16_mean_ep_reward_{eval_task}": top16_mean,
        f"nmax_ep_reward_{eval_task}": np.max(y_norm),
        f"nmedian_ep_reward_{eval_task}": np.median(y_norm),
        f"nmean_ep_reward_{eval_task}": np.mean(y_norm),
        f"ntop16_mean_ep_reward_{eval_task}": ntop16_mean,
    })

    if log_wandb and wandb is not None:
        wandb.log({
            f"max_ep_reward/{eval_task}": np.max(y),
            f"median_ep_reward/{eval_task}": np.median(y),
            f"mean_ep_reward/{eval_task}": np.mean(y),
            f"top16_mean_ep_reward/{eval_task}": top16_mean,
            f"nmax_ep_reward/{eval_task}": np.max(y_norm),
            f"nmedian_ep_reward/{eval_task}": np.median(y_norm),
            f"nmean_ep_reward/{eval_task}": np.mean(y_norm),
            f"ntop16_mean_ep_reward/{eval_task}": ntop16_mean,
            "eval_task": eval_task,
            "n_traj": Config.n_traj,
            "horizon": Config.horizon,
            "k": Config.k,
            "eps": Config.eps,
            "seed": Config.seed,
        })

    tag = save_suffix or eval_task
    # 与 train 共用同一 checkpoint；扫 w_text 时若不区分文件名会互相覆盖。
    _wtag = ""
    if getattr(Config, "use_text_condition", False) or getattr(
        Config, "multitask_text_only", False
    ):
        _wt = float(getattr(Config, "condition_guidance_w_text", 0.0))
        _wtag = f"_wtext{_wt:g}"
    _nctx = "_nctx" if rw_zs_no_ctx else ""
    _stem = (
        f"performance_{tag}_{Config.n_train_steps}_{trainer.batch_size}x"
        f"{Config.horizon - context_length}_alpha{Config.alpha}{_wtag}{_nctx}"
    )
    np.savez_compressed(os.path.join(logger.prefix, _stem), y=y, y_norm=y_norm, time=time)
    _stem_s = (
        f"samples_{tag}_{Config.n_train_steps}_{trainer.batch_size}x"
        f"{Config.horizon - context_length}_alpha{Config.alpha}{_wtag}{_nctx}"
    )
    np.savez_compressed(os.path.join(logger.prefix, _stem_s), queries=queries)
    return {
        'eval_task': eval_task,
        'max': float(np.max(y)),
        'median': float(np.median(y)),
        'mean': float(np.mean(y)),
        'top16_mean': float(top16_mean),
        'nmax': float(np.max(y_norm)),
        'nmedian': float(np.median(y_norm)),
        'nmean': float(np.mean(y_norm)),
        'ntop16_mean': float(ntop16_mean),
    }


def evaluate(**deps):
    global wandb
    from ml_logger import logger, RUN

    try:
        import wandb as _wandb

        wandb = _wandb
    except Exception as e:
        print(f"[wandb] import 失败，继续评估（不同步 wandb）: {e}", flush=True)
        wandb = None

    RUN._update(deps)
    print(deps)

    from diffuser.utils.multitask_canon import canonical_train_tasks_csv

    train_tasks = deps.get('train_tasks', deps.get('task', 'dkitty'))
    train_tasks = str(train_tasks)
    if ',' in train_tasks:
        train_tasks = canonical_train_tasks_csv(train_tasks)
        deps['train_tasks'] = train_tasks
    train_tasks_list = [t.strip() for t in train_tasks.split(',') if t.strip()]
    deps['train_tasks_list'] = train_tasks_list
    is_multitask = len(train_tasks_list) > 1
    deps['is_multitask'] = is_multitask

    eval_all_tasks = deps.get('eval_all_tasks', False)
    default_eval = deps.get('eval_task', train_tasks_list[0])
    if isinstance(default_eval, str) and ',' in default_eval:
        default_eval = default_eval.split(',')[0].strip()
    tasks_to_eval = train_tasks_list if (is_multitask and eval_all_tasks) else [default_eval]

    # 多任务 checkpoint 与「评谁」无关：配置锚点用字典序首任务（与 train.py 一致）
    ck_task = train_tasks_list[0]
    Config = _import_config(ck_task)
    Config._update(deps)
    for _tk in (
        "multitask_traj_signature",
        "traj_n_traj_dict",
        "traj_k_dict",
        "traj_eps_dict",
    ):
        if _tk in deps and deps[_tk] is not None:
            setattr(Config, _tk, deps[_tk])
    _te_dep = deps.get("train_epochs")
    if _te_dep is not None and str(_te_dep).strip() != "":
        te = int(_te_dep)
        if te >= 1:
            spe = int(getattr(Config, "n_steps_per_epoch", 100) or 100)
            Config.n_train_steps = te * spe
    Config.eval_task = ck_task
    Config.train_tasks_list = train_tasks_list
    Config.is_multitask = is_multitask

    if getattr(Config, "multitask_text_only", False) and not getattr(
        Config, "use_text_condition", False
    ):
        Config.use_text_condition = True

    if getattr(Config, "use_text_condition", False):
        from diffuser.utils.task_text_embedding import build_task_text_embedding_matrix

        root = Path(__file__).resolve().parent.parent
        meta = root / getattr(Config, "task_metadata_dir", "task_metadata")
        mat, dim = build_task_text_embedding_matrix(
            train_tasks_list,
            metadata_dir=meta,
            model_name=getattr(
                Config, "text_encoder_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
        )
        Config.text_embed_dim = int(dim)
        Config._task_text_embeds_np = mat

    logger.log_params(Config=vars(Config), RUN=vars(RUN))
    # 不覆盖 Config.device：与 train 一致，尊重 --device 及 CUDA_VISIBLE_DEVICES 下的默认 cuda

    run_name = (
        f"multi_eval_{'_'.join(tasks_to_eval)}_{Config.n_traj}x{Config.horizon}"
        if len(tasks_to_eval) > 1
        else f"{tasks_to_eval[0]}_{Config.n_traj}x{Config.horizon}_k{Config.k}_eps{Config.eps}_seed{Config.seed}"
    )
    if wandb is not None:
        if os.environ.get("WANDB_DISABLED", "").strip().lower() in ("1", "true", "yes"):
            print("[wandb] WANDB_DISABLED=1，跳过 wandb.init", flush=True)
            wandb = None
        else:
            _offline = (
                os.environ.get("WANDB_MODE", "").strip().lower() == "offline"
                or os.environ.get("GTG_WANDB_OFFLINE", "").strip().lower()
                in ("1", "true", "yes")
            )
            try:
                if _offline:
                    print(
                        "[wandb] 离线模式（GTG_WANDB_OFFLINE / WANDB_MODE=offline），日志仅本地",
                        flush=True,
                    )
                    _st = wandb.Settings(mode="offline")
                else:
                    try:
                        _to = int(os.environ.get("WANDB_INIT_TIMEOUT", "300"))
                    except ValueError:
                        _to = 300
                    _st = wandb.Settings(init_timeout=_to)
                _wgroup = os.environ.get("WANDB_RUN_GROUP", "").strip() or "evaluation"
                wandb.init(
                    project="decdiff-opt",
                    config=Config,
                    name=run_name,
                    group=_wgroup,
                    settings=_st,
                )
                if hasattr(wandb, "define_metric"):
                    wandb.define_metric("sample_viz_step")
                    wandb.define_metric("sample_viz/*", step_metric="sample_viz_step")
            except Exception as e:
                print(
                    f"[wandb] init 失败（评估仍继续，仅不同步云端）: {e}",
                    flush=True,
                )
                wandb = None

    if bool(deps.get("random_diffusion_weights")):
        if deps.get("load_diffusion_checkpoint"):
            raise ValueError(
                "random_diffusion_weights 与 load_diffusion_checkpoint 不能同时使用"
            )
        print(
            "[evaluate] random_diffusion_weights=1：跳过扩散 checkpoint，使用随机初始化权重",
            flush=True,
        )
        state_dict = None
        deps["_diffusion_checkpoint_loadpath"] = "<random_init>"
    else:
        _ld = deps.get("load_diffusion_checkpoint")
        if _ld and os.path.isfile(_ld):
            loadpath = _ld
        else:
            ckpt_dir = deps.get("diffusion_checkpoint_dir") or os.path.join(
                logger.prefix, "checkpoint"
            )
            _cts = deps.get("diffusion_ckpt_train_steps")
            if _cts is not None and str(_cts).strip() != "":
                ckpt_train_steps = int(_cts)
            else:
                ckpt_train_steps = None
            _te = deps.get("train_epochs")
            if _te is not None and str(_te).strip() != "":
                train_epochs = int(_te)
            else:
                train_epochs = None
            loadpath = _resolve_diffusion_checkpoint_path(
                ckpt_dir,
                Config,
                ckpt_train_steps=ckpt_train_steps,
                train_epochs=train_epochs,
            )
        if loadpath is None:
            ckpt_dir = deps.get("diffusion_checkpoint_dir") or os.path.join(
                logger.prefix, "checkpoint"
            )
            _hint = ""
            _cts2 = deps.get("diffusion_ckpt_train_steps")
            if _cts2 is not None and str(_cts2).strip() != "":
                _hint = (
                    f"（已指定 --diffusion_ckpt_train_steps={_cts2}，但缺少 "
                    f"state_{int(_cts2)}.pt）"
                )
            elif train_epochs is not None:
                spe = int(getattr(Config, "n_steps_per_epoch", 100) or 100)
                _hint = (
                    f"（已指定 train_epochs={train_epochs}×n_steps_per_epoch={spe}="
                    f"{int(train_epochs) * spe}，但缺少对应 state_*.pt；可改用 "
                    f"--diffusion_ckpt_train_steps 或检查 checkpoint 目录）"
                )
            raise FileNotFoundError(
                "未找到扩散 checkpoint：请设置 --load_diffusion_checkpoint，"
                f"或检查目录 {ckpt_dir} 下的 state_*.pt / state.pt{_hint}"
            )
        print(f"[evaluate] 加载扩散权重: {loadpath}", flush=True)
        state_dict = torch.load(loadpath, map_location=Config.device)
        deps["_diffusion_checkpoint_loadpath"] = loadpath

    results = []
    for i, eval_task in enumerate(tasks_to_eval):
        suffix = f"{eval_task}" if len(tasks_to_eval) > 1 else ""
        r = _evaluate_single_task(
            eval_task, deps, state_dict, Config,
            log_wandb=True,
            save_suffix=suffix,
        )
        results.append(r)

    if len(results) > 1:
        print("=== 多任务评估汇总（绝对值 / 归一化 [0,1]；top16 / ntop16 仅作拟合诊断）===")
        hdr = (
            f"{'task':<18} {'max':>10} {'median':>10} {'mean':>10} {'top16':>10} | "
            f"{'nmax':>10} {'nmedian':>10} {'nmean':>10} {'nt16':>10}"
        )
        print(hdr)
        print("-" * len(hdr))
        for r in results:
            print(
                f"{r['eval_task']:<18} {r['max']:10.4f} {r['median']:10.4f} {r['mean']:10.4f} "
                f"{r['top16_mean']:10.4f} | {r['nmax']:10.4f} {r['nmedian']:10.4f} {r['nmean']:10.4f} "
                f"{r['ntop16_mean']:10.4f}"
            )
        _wandb_summary = {}
        for r in results:
            _wandb_summary[f"summary/max_{r['eval_task']}"] = r['max']
            _wandb_summary[f"summary/nmax_{r['eval_task']}"] = r['nmax']
            _wandb_summary[f"summary/ntop16_mean_{r['eval_task']}"] = r['ntop16_mean']
        if wandb is not None:
            wandb.log(_wandb_summary)
    _json_out = deps.get("eval_summary_json_out")
    if _json_out and results:
        import json
        import re as _re_json
        from pathlib import Path as _Path

        _p = _Path(str(_json_out))
        _p.parent.mkdir(parents=True, exist_ok=True)
        _loadp = deps.get("_diffusion_checkpoint_loadpath")
        _m = _re_json.search(r"state_(\d+)\.pt$", str(_loadp or ""))
        _ckpt_step = int(_m.group(1)) if _m else None
        spe = int(getattr(Config, "n_steps_per_epoch", 100) or 100)
        _equiv_epochs = (
            _ckpt_step // spe if _ckpt_step is not None and spe > 0 else None
        )
        payload: dict[str, object] = {
            "is_multitask": bool(len(results) > 1),
            "train_tasks": [r["eval_task"] for r in results],
            "train_epochs": deps.get("train_epochs"),
            "diffusion_ckpt_train_steps": deps.get("diffusion_ckpt_train_steps"),
            "diffusion_checkpoint_train_steps_from_file": _ckpt_step,
            "eval_equiv_train_epochs": _equiv_epochs,
            "n_steps_per_epoch": spe,
            "condition_guidance_w_text": float(
                getattr(Config, "condition_guidance_w_text", 0.0)
            ),
            "diffusion_checkpoint_loadpath": _loadp,
            "tasks": {
                r["eval_task"]: {
                    k: float(r[k])
                    for k in (
                        "max",
                        "median",
                        "mean",
                        "top16_mean",
                        "nmax",
                        "nmedian",
                        "nmean",
                        "ntop16_mean",
                    )
                    if k in r
                }
                for r in results
            },
        }
        _p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[evaluate] wrote eval summary json: {_p}", flush=True)

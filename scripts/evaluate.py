from __future__ import annotations

import glob
import re

import diffuser.utils as utils
from ml_logger import logger
import torch

from diffuser.cpu_threads import apply_torch_cpu_threads_from_env

apply_torch_cpu_threads_from_env()

import numpy as np
import os
import pickle as pkl
import json
from pathlib import Path
from typing import Tuple

wandb = None
from diffuser.utils.arrays import to_torch
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


def _resolve_diffusion_checkpoint_path(ckpt_dir, Config):
    """
    与训练时实际落盘步数一致：优先 state.pt，再 state_{Config.n_train_steps}.pt；
    若不存在（例如训练用了 --train_epochs 覆盖了 n_train_steps，而 config 仍为默认值），
    则取 checkpoint 目录下 state_<步数>.pt 中步数最大的文件。
    """
    if not os.path.isdir(ckpt_dir):
        return None
    st = os.path.join(ckpt_dir, "state.pt")
    if os.path.isfile(st):
        return st
    if getattr(Config, "save_checkpoints", False):
        n = int(getattr(Config, "n_train_steps", 0))
        p = os.path.join(ckpt_dir, f"state_{n}.pt")
        if os.path.isfile(p):
            return p
        matches = glob.glob(os.path.join(ckpt_dir, "state_*.pt"))
        if matches:

            def _step(path):
                m = re.search(r"state_(\d+)\.pt$", path)
                return int(m.group(1)) if m else -1

            return max(matches, key=_step)
    return None


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
) -> Tuple[float, float, float, float]:
    """
    将当前扩散状态 x（归一化轨迹）解码为设计空间，取 context 之后时间步上的点，
    调用 Oracle 得到一批 y，返回 mean/max（原始与归一化）。
    """
    samples = x[..., :observation_dim]
    ds_dev = samples.device
    if hasattr(dataset.normalizer, "maxs"):
        dataset.normalizer.maxs = dataset.normalizer.maxs.to(ds_dev)
        dataset.normalizer.mins = dataset.normalizer.mins.to(ds_dev)
    unnormalized_samples = dataset.normalizer.unnormalize(samples)
    batch_size, horizon, _ = unnormalized_samples.shape
    if context_length >= horizon:
        return (float("nan"),) * 4

    if vae is not None:
        with torch.no_grad():
            samples_flat = unnormalized_samples.reshape(-1, latent_dim)
            decoded_flat = vae.decode(samples_flat)
            if scaler is not None:
                n_sf = int(len(scaler.scale_))
                if decoded_flat.shape[1] < n_sf:
                    return (float("nan"),) * 4
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
        return (float("nan"),) * 4
    return (
        float(np.mean(y)),
        float(np.max(y)),
        float(np.mean(y_norm)),
        float(np.max(y_norm)),
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
    diffusion_config = utils.Config(
        Config.diffusion,
        savepath="diffusion_config.pkl",
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


def _evaluate_single_task(eval_task, deps, state_dict, Config, log_wandb=True, save_suffix=""):
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
        if is_multitask:
            train_tasks_str = "_".join(train_tasks_list)
            vae_info_early = (
                f"./generated_datasets/multi_{train_tasks_str}_frac{Config.frac}_sigma{Config.sigma}/vae_info.p"
            )
        else:
            vae_info_early = (
                f"./generated_datasets/{eval_task}_frac{Config.frac}_sigma{Config.sigma}/vae_info.p"
            )
        latent_obs = int(deps.get("latent_observation_dim") or 32)
        if os.path.isfile(vae_info_early):
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
            f"[evaluate] MinimalTrajectoryDataset obs_dim={latent_obs}（来自 vae_info 或默认 32）",
            color="cyan",
        )
    elif is_multitask:
        data_dir = os.path.dirname(Config.data_path)
        from diffuser.utils.traj_params import (
            ensure_multitask_mixed_trajectories,
            resolve_multitask_mixed_path,
        )

        _sig = getattr(Config, "multitask_traj_signature", None)
        _skip_auto = bool(deps.get("skip_auto_construct_trajectories", False))
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
                fixed_dim=int(getattr(Config, "fixed_dim", 128)),
                skip_auto=_skip_auto,
            )
            mixed_data_path = resolve_multitask_mixed_path(data_dir, _sig)
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

    # 与 ZipDataset 一致的离线训练子集上 y 的最优值（非全局最优；亦非上面 DesignBench 仅用于归一化的全库 min/max）
    from diffuser.utils.offline_train_best import offline_training_best_y

    _y_tr_best = offline_training_best_y(
        eval_task,
        frac=float(Config.frac),
        sigma=float(Config.sigma),
        seed=int(getattr(Config, "seed", 1)),
    )
    logger.print(
        f"[{eval_task}] offline_train_best_y: {_y_tr_best}",
        color="green",
    )

    vae = None
    latent_dim = observation_dim
    vae_input_output_dim = getattr(Config, 'fixed_dim', 128)

    if is_multitask:
        train_tasks_str = '_'.join(train_tasks_list)
        vae_info_path = f"./generated_datasets/multi_{train_tasks_str}_frac{Config.frac}_sigma{Config.sigma}/vae_info.p"
    else:
        vae_info_path = f"./generated_datasets/{eval_task}_frac{Config.frac}_sigma{Config.sigma}/vae_info.p"

    scaler = None
    if os.path.exists(vae_info_path):
        try:
            with open(vae_info_path, 'rb') as f:
                vae_info = pkl.load(f)
            vae_path = vae_info.get('vae_path')
            latent_dim = vae_info.get('latent_dim', observation_dim)
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

    diffusion_config = utils.Config(
        Config.diffusion,
        savepath='diffusion_config.pkl',
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

    trainer.step = state_dict['step']
    trainer.model.load_state_dict(state_dict['model'])
    trainer.ema_model.load_state_dict(state_dict['ema'])
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
            m, mx, nm, nmx = _oracle_viz_stats_from_transition_x(
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
                        f"sample_viz/{_sv_tag}/mean_y": m,
                        f"sample_viz/{_sv_tag}/max_y": mx,
                        f"sample_viz/{_sv_tag}/mean_y_norm": nm,
                        f"sample_viz/{_sv_tag}/max_y_norm": nmx,
                        f"sample_viz/{_sv_tag}/t_index": int(timestep_index),
                        f"sample_viz/{_sv_tag}/denoise_progress": _prog,
                    },
                    step=_step_i,
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
                    "mean_y_norm": nm,
                    "max_y_norm": nmx,
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

        queries.append(samples[:, context_length:])
        contexts.append(samples[:, :context_length])

    queries = torch.cat(queries, dim=0).reshape(-1, original_observation_dim if vae is not None else observation_dim)
    contexts = torch.cat(contexts, dim=0).reshape(-1, original_observation_dim if vae is not None else observation_dim).cpu().numpy()
    queries_cpu = queries.cpu()

    if use_proxy_filter:
        if vae is not None:
            queries_norm = trainer.proxy_dataset.normalizer.normalize(queries_cpu.to('cpu'))
        else:
            queries_unnorm = trainer.dataset.normalizer.unnormalize(queries_cpu)
            queries_norm = trainer.proxy_dataset.normalizer.normalize(
                torch.tensor(queries_unnorm, device='cpu')
            )
        queries_norm = queries_norm.to(trainer.device)
        queries_proxy_score = trainer.proxy_model(queries_norm).flatten()
        queries = queries[torch.argsort(queries_proxy_score)[-num_queries:]].cpu()
    else:
        n_take = min(num_queries, queries.shape[0])
        queries = queries[:n_take].cpu()

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
    y_norm = (y - func.min) / (func.max - func.min)

    logger.print(
        f"[{eval_task}] max_ep_reward: {np.max(y)}, median: {np.median(y)}, mean: {np.mean(y)}",
        color='green',
    )
    logger.print(
        f"[{eval_task}] nmax_ep_reward: {np.max(y_norm)}, nmedian: {np.median(y_norm)}, nmean: {np.mean(y_norm)}",
        color='green',
    )
    logger.log_metrics_summary({
        f"max_ep_reward_{eval_task}": np.max(y),
        f"median_ep_reward_{eval_task}": np.median(y),
        f"mean_ep_reward_{eval_task}": np.mean(y),
        f"nmax_ep_reward_{eval_task}": np.max(y_norm),
        f"nmedian_ep_reward_{eval_task}": np.median(y_norm),
        f"nmean_ep_reward_{eval_task}": np.mean(y_norm),
    })

    if log_wandb and wandb is not None:
        wandb.log({
            f"max_ep_reward/{eval_task}": np.max(y),
            f"median_ep_reward/{eval_task}": np.median(y),
            f"mean_ep_reward/{eval_task}": np.mean(y),
            f"nmax_ep_reward/{eval_task}": np.max(y_norm),
            f"nmedian_ep_reward/{eval_task}": np.median(y_norm),
            f"nmean_ep_reward/{eval_task}": np.mean(y_norm),
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
        'nmax': float(np.max(y_norm)),
        'nmedian': float(np.median(y_norm)),
        'nmean': float(np.mean(y_norm)),
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
                wandb.init(
                    project="decdiff-opt",
                    config=Config,
                    name=run_name,
                    group="evaluation",
                    settings=_st,
                )
            except Exception as e:
                print(
                    f"[wandb] init 失败（评估仍继续，仅不同步云端）: {e}",
                    flush=True,
                )
                wandb = None

    _ld = deps.get("load_diffusion_checkpoint")
    if _ld and os.path.isfile(_ld):
        loadpath = _ld
    else:
        ckpt_dir = deps.get("diffusion_checkpoint_dir") or os.path.join(
            logger.prefix, "checkpoint"
        )
        loadpath = _resolve_diffusion_checkpoint_path(ckpt_dir, Config)
    if loadpath is None:
        raise FileNotFoundError(
            f"未找到扩散 checkpoint：请设置 --load_diffusion_checkpoint 或检查目录 "
            f"{deps.get('diffusion_checkpoint_dir') or os.path.join(logger.prefix, 'checkpoint')}"
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
        print("=== 多任务评估汇总（绝对值 / 归一化 [0,1]）===")
        hdr = f"{'task':<18} {'max':>10} {'median':>10} {'mean':>10} | {'nmax':>10} {'nmedian':>10} {'nmean':>10}"
        print(hdr)
        print("-" * len(hdr))
        for r in results:
            print(
                f"{r['eval_task']:<18} {r['max']:10.4f} {r['median']:10.4f} {r['mean']:10.4f} | "
                f"{r['nmax']:10.4f} {r['nmedian']:10.4f} {r['nmean']:10.4f}"
            )
        _wandb_summary = {}
        for r in results:
            _wandb_summary[f"summary/max_{r['eval_task']}"] = r['max']
            _wandb_summary[f"summary/nmax_{r['eval_task']}"] = r['nmax']
        if wandb is not None:
            wandb.log(_wandb_summary)

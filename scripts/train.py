import diffuser.utils as utils
import diffuser.models as models
import torch
import torch.nn.functional as F

from tqdm import tqdm
import pickle as pkl

# 由 main() 成功 import wandb 后赋值；失败则为 None（旧 typing_extensions 下 wandb 无法 import，训练仍应能跑）
wandb = None
import os
from pathlib import Path
from typing import Optional
from diffuser.models.vae import VAE
from diffuser.utils.training import Trainer
from diffuser.cpu_threads import apply_torch_cpu_threads_from_env
from diffuser.utils.multitask_proxy_paths import (
    multitask_proxy_checkpoint_exists,
    multitask_proxy_prefix,
)
from diffuser.utils.multitask_canon import (
    canonical_train_tasks_csv,
    multitask_path_token,
)
from diffuser.utils.vae_layout import (
    raw_train_pkl_path_from_latent_path,
    vae_state_pt_filename,
    vae_train_dir_suffix,
)
from diffuser.utils.traj_params import multitask_vae_dir_token
from diffuser.utils.proxy_filter import proxy_filter_enabled_for_train

apply_torch_cpu_threads_from_env()


def _wandb_float_tag(value: float) -> str:
    """Compact non-negative float for run names (no dots)."""
    return f"{float(value):.6g}".replace(".", "p").replace("-", "m")


def _env_truthy(key: str) -> bool:
    v = os.environ.get(key, "").strip().lower()
    return v in ("1", "true", "yes")


def _wandb_training_recipe_slug(cfg) -> str:
    """
    Human-distinguishable tags for Weights & Biases run name
    (training recipe: latent, loss shaping, text modes, CE, logging env knobs).
    """
    parts: list[str] = []
    z = int(getattr(cfg, "latent_dim", 32))
    if z != 32:
        parts.append(f"z{z}")

    tbp = float(getattr(cfg, "train_timestep_bias_power", 0.0))
    if tbp != 0.0:
        parts.append(f"ts{_wandb_float_tag(tbp)}")

    ms = float(getattr(cfg, "train_loss_min_snr_gamma", 0.0))
    if ms != 0.0:
        parts.append(f"snr{_wandb_float_tag(ms)}")

    hlr = float(getattr(cfg, "train_half_lr_mult", 1.0))
    if hlr != 1.0:
        parts.append(f"halftbias{_wandb_float_tag(hlr)}")

    if getattr(cfg, "use_text_condition", False):
        parts.append("txt")
    if getattr(cfg, "multitask_text_only", False):
        parts.append("mttxt")
    if getattr(cfg, "real_task_text_only_finetune", False) or getattr(
        cfg, "fewshot_text_only_finetune", False
    ):
        parts.append("rtft")

    _mt = bool(getattr(cfg, "is_multitask", False))
    _mto = bool(getattr(cfg, "multitask_text_only", False))
    _rft = bool(
        getattr(cfg, "real_task_text_only_finetune", False)
        or getattr(cfg, "fewshot_text_only_finetune", False)
    )
    if _mt and not _mto and not _rft:
        parts.append("mt_taskidx")

    if int(getattr(cfg, "proxy_filter", 1)) == 0:
        parts.append("nopx")

    if getattr(cfg, "returns_condition", False):
        parts.append("ret")

    rs = (getattr(cfg, "run_suffix", "") or "").strip()
    if rs:
        safe = rs.lstrip("_").replace("/", "_").replace(" ", "")
        if safe:
            parts.append(safe)

    if _env_truthy("DUO_LOG_PER_T_LOSS"):
        parts.append("logt")
        try:
            nb = int(os.environ.get("DUO_LOG_PER_T_LOSS_BINS", "20"))
        except ValueError:
            nb = 20
        if nb != 20:
            parts.append(f"tb{nb}")

    raw_lam = os.environ.get("DUO_DISCRETE_CE_LAMBDA", "").strip()
    if raw_lam and "_ce" not in rs:
        try:
            lam = float(raw_lam)
        except ValueError:
            lam = 0.0
        if lam > 0.0:
            parts.append(f"dce{_wandb_float_tag(lam)}")

    return "_".join(parts)


def _maybe_build_task_text_embeddings(Config):
    """If use_text_condition, build [K, D] numpy table from task_metadata/*.txt (cached)."""
    if not getattr(Config, "use_text_condition", False):
        return
    from diffuser.utils.task_text_embedding import build_task_text_embedding_matrix

    root = Path(__file__).resolve().parent.parent
    meta = root / getattr(Config, "task_metadata_dir", "task_metadata")
    mat, dim = build_task_text_embedding_matrix(
        Config.train_tasks_list,
        metadata_dir=meta,
        model_name=getattr(
            Config, "text_encoder_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
    Config.text_embed_dim = int(dim)
    Config._task_text_embeds_np = mat
    print(
        f"[task text] use_text_condition=True, embed_dim={dim}, metadata_dir={meta}"
    )


def ensure_multitask_proxies(diffusion, dataset, renderer, Config, action_dim, logger):
    """多任务训练前：为 train_tasks_list 中每个任务检查 proxy；不存在则训练并保存。"""
    for task_name in Config.train_tasks_list:
        prefix = multitask_proxy_prefix(task_name, Config)
        if multitask_proxy_checkpoint_exists(prefix, Config):
            logger.print(f"[multitask proxy] 已存在，跳过: {task_name}")
            continue
        logger.print(f"[multitask proxy] 开始训练任务 {task_name} 的 proxy → {prefix}")
        proxy_ds = utils.Config(
            Config.proxy_loader,
            dataset=task_name,
            frac=Config.frac,
            sigma=Config.sigma,
            soo_seed=int(getattr(Config, 'seed', 1)),
            savepath=f"proxy_dataset_{task_name}.pkl",
        )()
        orig_dim = proxy_ds.original_observation_dim
        proxy_m = utils.Config(
            Config.proxy_model,
            savepath=f"proxy_model_{task_name}.pkl",
            input_dim=orig_dim,
            hidden_dim=Config.proxy_hidden_dim,
            output_dim=action_dim,
            n_ensembles=Config.proxy_n_ensembles,
            device=Config.device,
        )()
        proxy_trainer = Trainer(
            diffusion,
            proxy_m,
            dataset,
            proxy_ds,
            renderer,
            ema_decay=Config.ema_decay,
            train_batch_size=Config.batch_size,
            train_lr=Config.learning_rate,
            proxy_train_lr=Config.proxy_learning_rate,
            gradient_accumulate_every=Config.gradient_accumulate_every,
            log_freq=Config.log_freq,
            proxy_log_freq=Config.proxy_log_freq,
            sample_freq=Config.sample_freq,
            save_freq=Config.save_freq,
            proxy_save_freq=Config.proxy_save_freq,
            label_freq=int(Config.n_train_steps // Config.n_saves),
            save_parallel=Config.save_parallel,
            n_reference=Config.n_reference,
            bucket=Config.bucket,
            train_device=Config.device,
            save_checkpoints=Config.save_checkpoints,
            proxy_save_prefix=prefix,
        )
        proxy_trainer.train_proxy(n_train_steps=Config.proxy_n_train_steps)
        logger.print(f"[multitask proxy] 任务 {task_name} proxy 训练完成")

def train_multitask_vae(
    tasks_list,
    latent_dim: int = 32,
    fixed_dim: int = 128,
    frac: float = 1.0,
    sigma: float = 0.0,
    seed: int = 0,
    device: str = "cuda",
    *,
    num_epochs: int = 100,
    force_retrain: bool = False,
    multitask_traj_signature: Optional[str] = None,
) -> VAE:
    """
    与 ``construct_trajectories`` 一致：多任务 VAE 仅通过 ``train_vae.main`` 训练，
    产出 ``dataset_info.p`` / ``scaler_*.p`` / ``vae_info.p`` 及带验证集的 loss 日志。
    """
    from train_vae import main as train_vae_main

    _csv = canonical_train_tasks_csv(
        ",".join(str(t).strip() for t in tasks_list if str(t).strip())
    )

    class VAEArgs:
        """与 ``construct_trajectories.VAEArgs``（多任务）对齐的占位 namespace。"""

    ns = VAEArgs()
    ns.tasks = _csv
    ns.task = None
    ns.frac = float(frac)
    ns.sigma = float(sigma)
    ns.fixed_dim = int(fixed_dim)
    ns.force_retrain = bool(force_retrain)
    ns.latent_dim = int(latent_dim)
    ns.d_model = 256
    ns.nhead = 4
    ns.num_layers = 4
    ns.dropout = 0.1
    ns.batch_size = 64
    ns.val_split = 0.1
    ns.lr = 1e-4
    ns.weight_decay = 1e-5
    ns.num_epochs = int(num_epochs)
    ns.kl_weight = 0.1
    ns.seed = int(seed)
    ns.device = str(device)
    ns.pretrained_vae_info = None
    ns.multitask_traj_signature = (
        str(multitask_traj_signature) if multitask_traj_signature is not None else None
    )

    print(
        f"[multitask VAE] 调用 train_vae.main（与 construct 相同），tasks={_csv}, "
        f"latent_dim={latent_dim}, num_epochs={num_epochs}, "
        f"multitask_traj_signature={ns.multitask_traj_signature!r}"
    )
    vae, _scalers, _save_dir = train_vae_main(ns)
    return vae

def main(**deps):
    global wandb
    from ml_logger import logger, RUN
    import os

    try:
        import wandb as _wandb

        wandb = _wandb
    except Exception as e:
        print(f"[wandb] import 失败，继续训练（不同步 wandb）: {e}", flush=True)
        wandb = None

    train_epochs = deps.pop("train_epochs", None)
    _retrain = bool(deps.pop("retrain", False))

    RUN._update(deps)
    print(deps)

    # 默认按离散 t 分桶记录扩散 MSE（diffusion._per_t_bin_weighted_mse_metrics）；可用环境变量关闭或改 bins
    os.environ.setdefault("DUO_LOG_PER_T_LOSS", "1")
    os.environ.setdefault("DUO_LOG_PER_T_LOSS_BINS", "20")

    if 'train_tasks' in deps and ',' in str(deps.get('train_tasks', '')):
        deps['train_tasks'] = canonical_train_tasks_csv(deps['train_tasks'])

    # 确定使用哪个配置文件（多任务：与 eval 无关，用 canonical 后 train_tasks 的逗号首项，即字典序第一个任务）
    task_to_use = deps.get('eval_task', deps.get('task', 'dkitty'))
    if 'train_tasks' in deps and ',' in str(deps.get('train_tasks', '')):
        task_to_use = deps['train_tasks'].split(',')[0].strip()

    if task_to_use == 'ant':
        from config.ant_config import Config
    elif task_to_use == 'dkitty':
        from config.dkitty_config import Config
    elif task_to_use == 'tfbind8':
        from config.tfbind8_config import Config
    elif task_to_use == 'tfbind10':
        from config.tfbind10_config import Config  
    elif task_to_use == 'superconductor':
        from config.superconductor_config import Config
    elif task_to_use in ('gtopx2', 'gtopx3', 'gtopx4', 'gtopx6'):
        from config.gtopx_config import Config
    else:
        from config.dkitty_config import Config
    
    # 更新配置
    Config._update(deps)
    if train_epochs is not None:
        if int(train_epochs) < 1:
            raise ValueError("train_epochs 须为 >= 1 的整数")
        Config.n_train_steps = int(train_epochs) * int(Config.n_steps_per_epoch)
    for _tk in (
        "multitask_traj_signature",
        "traj_n_traj_dict",
        "traj_k_dict",
        "traj_eps_dict",
    ):
        if _tk in deps and deps[_tk] is not None:
            setattr(Config, _tk, deps[_tk])

    # 检查是否为多任务模式
    if 'train_tasks' in deps and ',' in deps['train_tasks']:
        Config.is_multitask = True
        # 将训练任务列表转换为数组（与路径 multi_<字典序>_ 一致）
        Config.train_tasks_list = [
            t.strip() for t in deps['train_tasks'].split(',') if t.strip()
        ]
        print(f"📊 多任务训练模式启用 📊")
        print(f"📋 训练任务列表: {Config.train_tasks_list}")
        print(f"🎯 Config 锚点任务（与 checkpoint 路径无关）: {task_to_use}")
        print(f"🔍 任务数量: {len(Config.train_tasks_list)}")
    else:
        Config.is_multitask = False
        # 单任务模式下，确保train_tasks_list是列表格式
        if 'train_tasks' in deps:
            Config.train_tasks_list = [deps['train_tasks']]
        else:
            Config.train_tasks_list = [Config.dataset]
        print(f"📊 单任务训练模式启用 📊")
        print(f"📋 训练任务: {Config.train_tasks_list[0]}")

    _real_task_ft = getattr(Config, "real_task_text_only_finetune", False) or getattr(
        Config, "fewshot_text_only_finetune", False
    )
    if _real_task_ft and train_epochs is not None:
        _te_int = int(train_epochs)
        # 微调常见 ~100 epoch；若误把「预训练轮数」当作 --train_epochs，会得到 1400/1500 等
        if _te_int >= 500:
            print(
                f"⚠️ real_task 微调：--train_epochs={_te_int} 很大（n_train_steps={int(Config.n_train_steps)}）。"
                f"若你只想微调约 100 epoch、仅多任务预训练用 1400+，请对 train 传更小的 --train_epochs，"
                f"预训练步数只应体现在 load 的 state_*.pt，而不是本项。",
                flush=True,
            )
    if _real_task_ft:
        if not getattr(Config, "use_text_condition", False):
            Config.use_text_condition = True
            print("⚠️ real_task_text_only_finetune：已自动设置 use_text_condition=True")
        print(
            "📌 real-task：单任务轨迹 + 仅文本条件（与 multitask_text_only 同架构，task_condition=False）"
        )
    elif getattr(Config, "multitask_text_only", False):
        if not Config.is_multitask:
            raise ValueError("multitask_text_only 仅适用于多任务（train_tasks 含多个任务）")
        if not getattr(Config, "use_text_condition", False):
            Config.use_text_condition = True
            print("⚠️ multitask_text_only：已自动设置 use_text_condition=True")
        print(
            "📌 multitask_text_only：多任务数据 + 仅 text 分支（task_condition=False，batch 无 task_idx）"
        )

    use_task_branch = Config.is_multitask and not getattr(
        Config, "multitask_text_only", False
    ) and not _real_task_ft

    use_proxy_filter = proxy_filter_enabled_for_train(deps)
    setattr(Config, "proxy_filter", use_proxy_filter)
    print(
        f"[train] proxy_filter={int(use_proxy_filter)} "
        f"({'训练 proxy 并在评估中用于筛选' if use_proxy_filter else '关闭：不训 proxy，评估时仅扩散采样后 eval'})",
        flush=True,
    )

    # gtopx 等共用 gtopx_config 时类默认 dataset 仍为 gtopx2；必须与真实任务名一致，
    # 否则 ZipDataset / load_gtopx_offline_arrays 维数错误，evaluate 会与 checkpoint 不一致。
    if not getattr(Config, "is_multitask", False) and Config.train_tasks_list:
        Config.dataset = Config.train_tasks_list[0]

    # logger.remove('*.pkl')
    # logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))
    logger.log_text("""
                    charts:
                    - yKey: loss
                      xKey: steps
                    - yKey: a0_loss
                      xKey: steps
                    """, filename=".charts.yml", dedent=True, overwrite=True)

    torch.backends.cudnn.benchmark = False
    utils.set_seed(Config.seed)
    # -----------------------------------------------------------------------------#
    # ---------------------------------- dataset ----------------------------------#
    # -----------------------------------------------------------------------------#
    
    # 构建自定义的run名称，包含任务和参数信息
    if Config.is_multitask:
        _sig = getattr(Config, "multitask_traj_signature", None)
        _s = _sig or f"{Config.n_traj}x{Config.horizon}_k{Config.k}_eps{Config.eps}"
        run_name = f"multitask_{'_'.join(Config.train_tasks_list)}_{_s}_seed{Config.seed}"
    else:
        run_name = f"{Config.train_tasks_list[0]}_{Config.n_traj}x{Config.horizon}_k{Config.k}_eps{Config.eps}_seed{Config.seed}"

    _recipe = _wandb_training_recipe_slug(Config)
    if _recipe:
        run_name = f"{run_name}_{_recipe}"

    if wandb is not None:
        import os as _os

        if _os.environ.get("WANDB_DISABLED", "").strip().lower() in ("1", "true", "yes"):
            print("[wandb] WANDB_DISABLED=1，跳过 wandb.init", flush=True)
            wandb = None
        else:
            _offline = (
                _os.environ.get("WANDB_MODE", "").strip().lower() == "offline"
                or _os.environ.get("GTG_WANDB_OFFLINE", "").strip().lower()
                in ("1", "true", "yes")
            )
            try:
                if _offline:
                    print(
                        "[wandb] 离线模式（GTG_WANDB_OFFLINE=1 或 WANDB_MODE=offline），日志仅本地",
                        flush=True,
                    )
                    _st = wandb.Settings(mode="offline")
                else:
                    try:
                        _to = int(_os.environ.get("WANDB_INIT_TIMEOUT", "300"))
                    except ValueError:
                        _to = 300
                    _st = wandb.Settings(init_timeout=_to)
                _wandb_kw = {}
                _wrid = _os.environ.get("WANDB_RUN_ID", "").strip()
                if _wrid:
                    _wandb_kw["id"] = _wrid
                _wres = _os.environ.get("WANDB_RESUME", "").strip().lower()
                if _wres in ("allow", "must", "auto", "true", "1"):
                    _wandb_kw["resume"] = (
                        "allow" if _wres in ("true", "1", "auto") else _wres
                    )
                if _wandb_kw:
                    print(f"[wandb] resume kwargs: {_wandb_kw}", flush=True)
                wandb.init(
                    project="decdiff-opt",
                    config=Config,
                    name=run_name,
                    settings=_st,
                    **_wandb_kw,
                )
            except Exception as e:
                print(f"[wandb] init 失败（训练仍继续）: {e}", flush=True)
                wandb = None

    if wandb is not None:
        # 更新wandb配置，添加多任务相关信息
        if Config.is_multitask:
            wandb.config.update(
                {
                    'is_multitask': True,
                    'train_tasks': Config.train_tasks_list,
                    'eval_task': Config.eval_task,
                    'num_tasks': len(Config.train_tasks_list),
                    'multitask_text_only': getattr(Config, "multitask_text_only", False),
                },
                allow_val_change=True,
            )
        else:
            wandb.config.update(
                {
                    'is_multitask': False,
                    'train_task': Config.train_tasks_list[0],
                    'eval_task': Config.train_tasks_list[0],
                },
                allow_val_change=True,
            )

        from diffuser.utils.training import configure_wandb_step_axes

        configure_wandb_step_axes(
            include_proxy_axis=not Config.is_multitask and use_proxy_filter,
        )

    _maybe_build_task_text_embeddings(Config)

    # 多任务模式下，创建混合数据集
    if hasattr(Config, 'is_multitask') and Config.is_multitask:
        print(f"创建多任务数据集: {Config.train_tasks_list}")
        
        # 加载混合轨迹文件
        from diffuser.datasets.sequence import PointRegretDataset
        import os
        
        # 构建完整的混合轨迹文件路径
        data_dir = os.path.dirname(Config.data_path)
        from diffuser.utils.traj_params import (
            ensure_multitask_mixed_trajectories,
            resolve_multitask_mixed_path,
        )

        _sig = getattr(Config, "multitask_traj_signature", None)
        _skip_auto = bool(deps.get("skip_auto_construct_trajectories", False))
        _latent = int(getattr(Config, "latent_dim", 32))
        ensure_multitask_mixed_trajectories(
            train_tasks_list=list(Config.train_tasks_list),
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
            latent_dim=_latent,
        )
        mixed_data_path = resolve_multitask_mixed_path(data_dir, _sig, _latent)
        
        # 直接加载混合轨迹文件
        dataset = PointRegretDataset(
            horizon=Config.horizon,
            data_path=mixed_data_path,
            context_length=Config.context_length,
            regret=Config.regret,
            include_returns=Config.include_returns,
            task_name=None,  # 混合文件包含所有任务的信息
            task_text_embeds=getattr(Config, "_task_text_embeds_np", None),
            include_task_idx=use_task_branch,
        )
        print(f"加载混合轨迹文件: {mixed_data_path}")
        print(f"混合数据集大小: {len(dataset)}")
        

        
        # 对于proxy数据集，在多任务模式下我们不再使用它（因为我们移除了proxy model训练）
        proxy_dataset = None
        print("多任务模式下，proxy_dataset设置为None，因为我们不再训练proxy model")
    else:
        # 单任务模式，保持原有逻辑
        dataset_config = utils.Config(
            Config.loader,
            savepath='dataset_config.pkl',
            horizon=Config.horizon,
            data_path=Config.data_path,
            context_length=Config.context_length,
            regret=Config.regret,
            include_returns=Config.include_returns,
            task_name=Config.dataset,
            task_text_embeds=getattr(Config, "_task_text_embeds_np", None),
        )
        
        dataset = dataset_config()
        if use_proxy_filter:
            proxy_dataset_config = utils.Config(
                Config.proxy_loader,
                dataset=Config.dataset,
                frac=Config.frac,
                sigma=Config.sigma,
                soo_seed=int(getattr(Config, 'seed', 1)),
                savepath='proxy_dataset_config.pkl',
            )
            proxy_dataset = proxy_dataset_config()
        else:
            proxy_dataset = None

    # render_config = utils.Config(
    #     Config.renderer,
    #     savepath='render_config.pkl',
    #     env=Config.dataset,
    # )

    # renderer = render_config()
    renderer = Config.renderer
    observation_dim = dataset.observation_dim
    
    # 处理多任务模式下proxy_dataset为None的情况
    if proxy_dataset is not None:
        original_observation_dim = proxy_dataset.original_observation_dim
    else:
        # 在多任务模式下，我们需要从dataset获取original_observation_dim
        # 假设dataset有original_observation_dim属性
        original_observation_dim = getattr(dataset, 'original_observation_dim', observation_dim)
    
    action_dim = dataset.action_dim

    # -----------------------------------------------------------------------------#
    # ------------------------------ model & trainer ------------------------------#
    # -----------------------------------------------------------------------------#
    _ts_bias = float(getattr(Config, "train_timestep_bias_power", 0.0))
    _min_snr = float(getattr(Config, "train_loss_min_snr_gamma", 0.0))
    if _ts_bias > 0.0 or _min_snr > 0.0:
        print(
            f"[diffusion train opt] train_timestep_bias_power={_ts_bias}, "
            f"train_loss_min_snr_gamma={_min_snr}",
            flush=True,
        )

    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        model_config = utils.Config(
            Config.model,
            savepath='model_config.pkl',
            horizon=Config.horizon,
            transition_dim=observation_dim,
            cond_dim=observation_dim,
            dim_mults=Config.dim_mults,
            returns_condition=Config.returns_condition,
            dim=Config.dim,
            condition_dropout=Config.condition_dropout,
            calc_energy=Config.calc_energy,
            device=Config.device,
            task_condition=use_task_branch,
            num_tasks=len(Config.train_tasks_list) if Config.is_multitask else 1,
            text_condition=getattr(Config, "use_text_condition", False),
            text_embed_input_dim=int(
                getattr(Config, "text_embed_dim", 384)
            ),
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
            hidden_dim=Config.hidden_dim,
            ar_inv=Config.ar_inv,
            train_only_inv=Config.train_only_inv,
            ## loss weighting
            action_weight=Config.action_weight,
            loss_weights=Config.loss_weights,
            loss_discount=Config.loss_discount,
            returns_condition=Config.returns_condition,
            condition_guidance_w=Config.condition_guidance_w,
            condition_guidance_w_task=float(getattr(Config, "condition_guidance_w_task", 0.0)),
            condition_guidance_w_text=float(getattr(Config, "condition_guidance_w_text", 0.0)),
            cfg_apply_task=bool(getattr(Config, "cfg_apply_task", True)),
            cfg_apply_text=bool(getattr(Config, "cfg_apply_text", True)),
            sample_with_task_embedding=bool(getattr(Config, "sample_with_task_embedding", True)),
            sample_with_text_embedding=bool(getattr(Config, "sample_with_text_embedding", True)),
            train_tasks_list=tuple(Config.train_tasks_list),
            device=Config.device,
            train_timestep_bias_power=float(
                getattr(Config, "train_timestep_bias_power", 0.0)
            ),
            train_loss_min_snr_gamma=float(
                getattr(Config, "train_loss_min_snr_gamma", 0.0)
            ),
            train_half_timestep_bias_frac=float(
                getattr(Config, "train_half_timestep_bias_frac", 0.7)
            ),
            train_half_lr_mult=float(getattr(Config, "train_half_lr_mult", 1.0)),
        )
        
        # 单任务模式下，为GaussianInvDynDiffusion模型添加proxy_model_config
        if not Config.is_multitask and use_proxy_filter:
            proxy_model_config = utils.Config(
                Config.proxy_model,
                savepath='proxy_model_config.pkl',
                input_dim=original_observation_dim,
                hidden_dim=Config.proxy_hidden_dim,
                output_dim=action_dim,  # 单任务模式下与dfgo-main保持一致，设置为action_dim
                n_ensembles=Config.proxy_n_ensembles,
                device=Config.device,
            )
    else:
        model_config = utils.Config(
            Config.model,
            savepath='model_config.pkl',
            horizon=Config.horizon,
            transition_dim=observation_dim + action_dim,
            cond_dim=observation_dim,
            dim_mults=Config.dim_mults,
            returns_condition=Config.returns_condition,
            dim=Config.dim,
            condition_dropout=Config.condition_dropout,
            calc_energy=Config.calc_energy,
            device=Config.device,
            task_condition=use_task_branch,
            num_tasks=len(Config.train_tasks_list) if Config.is_multitask else 1,
            text_condition=getattr(Config, "use_text_condition", False),
            text_embed_input_dim=int(
                getattr(Config, "text_embed_dim", 384)
            ),
            text_condition_dropout=float(
                getattr(Config, "text_condition_dropout", 0.1)
            ),
        )
            
        # 非GaussianInvDynDiffusion模型的proxy_model_config定义
        if not Config.is_multitask and use_proxy_filter:
            proxy_model_config = utils.Config(
                Config.proxy_model,
                savepath='proxy_model_config.pkl',
                input_dim=original_observation_dim,
                hidden_dim=Config.proxy_hidden_dim,
                output_dim=action_dim,  # 单任务模式下与dfgo-main保持一致，设置为action_dim
                n_ensembles=Config.proxy_n_ensembles,
                device=Config.device,
            )

        diffusion_config = utils.Config(
            Config.diffusion,
            savepath='diffusion_config.pkl',
            horizon=Config.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=Config.n_diffusion_steps,
            n_sample_timesteps=int(
                getattr(Config, "n_sample_timesteps", Config.n_diffusion_steps)
            ),
            loss_type=Config.loss_type,
            clip_denoised=Config.clip_denoised,
            predict_epsilon=Config.predict_epsilon,
            ## loss weighting
            action_weight=Config.action_weight,
            loss_weights=Config.loss_weights,
            loss_discount=Config.loss_discount,
            returns_condition=Config.returns_condition,
            condition_guidance_w=Config.condition_guidance_w,
            condition_guidance_w_task=float(getattr(Config, "condition_guidance_w_task", 0.0)),
            condition_guidance_w_text=float(getattr(Config, "condition_guidance_w_text", 0.0)),
            cfg_apply_task=bool(getattr(Config, "cfg_apply_task", True)),
            cfg_apply_text=bool(getattr(Config, "cfg_apply_text", True)),
            sample_with_task_embedding=bool(getattr(Config, "sample_with_task_embedding", True)),
            sample_with_text_embedding=bool(getattr(Config, "sample_with_text_embedding", True)),
            train_tasks_list=tuple(Config.train_tasks_list),
            device=Config.device,
            train_timestep_bias_power=float(
                getattr(Config, "train_timestep_bias_power", 0.0)
            ),
            train_loss_min_snr_gamma=float(
                getattr(Config, "train_loss_min_snr_gamma", 0.0)
            ),
            train_half_timestep_bias_frac=float(
                getattr(Config, "train_half_timestep_bias_frac", 0.7)
            ),
            train_half_lr_mult=float(getattr(Config, "train_half_lr_mult", 1.0)),
        )

    _ld_path = getattr(Config, "load_diffusion_checkpoint", None)
    if _ld_path in ("", None):
        _ld_path = None
    _ld_epoch = getattr(Config, "load_diffusion_checkpoint_epoch", None)
    if (
        _ld_path is None
        and (not _retrain)
        and _ld_epoch is None
        and bool(getattr(Config, "save_checkpoints", True))
    ):
        import re as _re_ckpt

        def _resume_step_from_ckpt_path(p: str) -> int:
            m = _re_ckpt.search(r"state_(\d+)\.pt$", p)
            return int(m.group(1)) if m else 0

        _ck_dir = os.path.join(str(logger.prefix), "checkpoint")
        if os.path.isdir(_ck_dir):
            from diffuser.utils.real_task_transfer import resolve_diffusion_state_pt

            _cand = resolve_diffusion_state_pt(_ck_dir, Config)
            if _cand and os.path.isfile(_cand):
                _rs = _resume_step_from_ckpt_path(_cand)
                _goal = int(getattr(Config, "n_train_steps", 0) or 0)
                _ld_path = _cand
                setattr(Config, "load_diffusion_checkpoint", _cand)
                if _goal > 0 and _rs < _goal:
                    print(
                        f"[train] resume（默认）：加载 {_cand}（文件名 step≈{_rs}），"
                        f"继续训练至 n_train_steps={_goal}；同目录从零重训请加 --retrain",
                        flush=True,
                    )
                elif _goal > 0:
                    print(
                        f"[train] resume（默认）：加载 {_cand}（step≈{_rs}），"
                        f"已达或超过 n_train_steps={_goal}，扩散循环将跳过。",
                        flush=True,
                    )
                else:
                    print(
                        f"[train] resume（默认）：加载 {_cand}（step≈{_rs}）。",
                        flush=True,
                    )
    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        proxy_train_lr=Config.proxy_learning_rate,  # 添加proxy模型的学习率参数
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
        save_checkpoints=Config.save_checkpoints,
        load_checkpoint_path=_ld_path,
        load_checkpoint=_ld_epoch,
    )

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    # 加载VAE模型用于从隐空间解码到原始空间
    # 原始观测维度已经在前面设置
    
    # 确定VAE模型路径
    fixed_dim = getattr(Config, 'fixed_dim', 128)
    _latent = int(getattr(Config, "latent_dim", 32))
    _vae_pt = vae_state_pt_filename(_latent)
    _vae_sub = vae_train_dir_suffix(_latent)
    if Config.is_multitask:
        # 与 train_vae.main / construct 一致：multi_<字典序 token>，且与轨迹签名同 mt_<hex>（有签名时）
        train_tasks_str = multitask_path_token(
            canonical_train_tasks_csv(",".join(Config.train_tasks_list))
        )
        _mt_sig = getattr(Config, "multitask_traj_signature", None)
        _mt_tok = multitask_vae_dir_token(_mt_sig) if _mt_sig else ""
        vae_model_path = (
            f"./trained_models/vae/multi_{train_tasks_str}_frac{Config.frac}_sigma{Config.sigma}"
            f"_dim{fixed_dim}{_mt_tok}{_vae_sub}/{_vae_pt}"
        )
        print(f"多任务模式，VAE模型路径: {vae_model_path}")
    else:
        task_name = Config.train_tasks_list[0]
        vae_model_path = (
            f"./trained_models/vae/{task_name}_frac{Config.frac}_sigma{Config.sigma}_dim{fixed_dim}{_vae_sub}/{_vae_pt}"
        )
        print(f"单任务模式，VAE模型路径: {vae_model_path}")
    
    # 确保模型目录存在
    os.makedirs(os.path.dirname(vae_model_path), exist_ok=True)
    
    # 创建VAE模型实例（input_dim 与 train_vae / 轨迹构建一致）
    vae = VAE(input_dim=fixed_dim, latent_dim=_latent)
    vae.to(Config.device)
    
    # 尝试加载现有模型
    if os.path.exists(vae_model_path):
        print(f"加载VAE模型: {vae_model_path}")
        vae.load_state_dict(torch.load(vae_model_path, map_location=Config.device))
        vae.eval()
    else:
        print(f"未找到VAE模型，需要训练新模型: {vae_model_path}")
        # 根据模式选择不同的训练方法
        if Config.is_multitask:
            print(f"开始在多个数据集上训练VAE: {Config.train_tasks_list}")
            vae = train_multitask_vae(
                tasks_list=Config.train_tasks_list,
                latent_dim=_latent,
                fixed_dim=int(fixed_dim),
                frac=float(Config.frac),
                sigma=float(Config.sigma),
                seed=int(getattr(Config, "seed", 0)),
                device=str(Config.device),
                num_epochs=100,
                force_retrain=False,
                multitask_traj_signature=getattr(
                    Config, "multitask_traj_signature", None
                ),
            )
            # train_vae.main 已写入与 vae_model_path 同目录的权重与 scaler / vae_info
            vae.eval()
            print(f"多任务VAE已由 train_vae 写入: {os.path.dirname(os.path.abspath(vae_model_path))}")
        else:
            # 单任务VAE训练逻辑，使用原有数据集训练VAE
            print(f"单任务模式: 使用{Config.dataset}数据集训练VAE")
            
            # 加载原始观测数据用于训练VAE
            original_data_path = raw_train_pkl_path_from_latent_path(Config.data_path)
            
            if os.path.exists(original_data_path):
                print(f"加载原始观测数据: {original_data_path}")
                with open(original_data_path, 'rb') as f:
                    trajectories = pkl.load(f)
                
                # 提取所有观测数据
                observations = []
                for traj in trajectories:
                    observations.append(traj['obs'])
                observations = torch.cat(observations, dim=0)
                print(f"观测数据形状: {observations.shape}")
                
                # 训练VAE
                from torch.utils.data import TensorDataset, DataLoader
                dataset = TensorDataset(observations)
                dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
                
                optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
                vae.train()
                
                print("开始训练单任务VAE...")
                for epoch in tqdm(range(100)):
                    total_loss = 0
                    for batch in dataloader:
                        x = batch[0].to(Config.device)
                        recon_x, mu, logvar, _z = vae(x)
                        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
                        kl_loss = -0.5 * torch.mean(
                            torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                        )
                        loss = recon_loss + 0.1 * kl_loss
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                    
                    avg_loss = total_loss / len(dataloader)
                    if (epoch + 1) % 10 == 0:
                        print(f"VAE训练 Epoch {epoch+1}/100, Loss: {avg_loss:.6f}")
                
                # 保存训练好的VAE模型
                torch.save(vae.state_dict(), vae_model_path)
                print(f"单任务VAE模型已保存至: {vae_model_path}")
            else:
                print(f"未找到原始观测数据: {original_data_path}")
                print("使用预训练VAE或随机初始化VAE")
                vae.eval()

    model = model_config()
    if not Config.is_multitask and use_proxy_filter:
        proxy_model = proxy_model_config()
    else:
        proxy_model = None

    diffusion = diffusion_config(model)

    if Config.is_multitask and use_proxy_filter:
        ensure_multitask_proxies(diffusion, dataset, renderer, Config, action_dim, logger)

    # 创建训练器（load_diffusion_checkpoint → Trainer.load_from_path：加载 model/ema/step，优化器仍从新 Adam 起，在预训练权重上继续更新）
    trainer = trainer_config(diffusion, proxy_model, dataset, proxy_dataset, renderer)
    # Provide total training steps for optional two-stage schedule.
    setattr(trainer, "_total_train_steps", int(getattr(Config, "n_train_steps", 0)))
    trainer.vae = vae  # 添加VAE属性到训练器
    if _ld_path:
        logger.print(
            f"[train] 已从预训练 checkpoint 加载扩散权重（微调）：{_ld_path} | trainer.step={trainer.step}",
            color="cyan",
        )

    # 多任务预训练等外源 state_*.pt 内可能带 step=5e4；本 run 的 n_train_steps 仅 1e4 时
    # 不重置会满足 step>=n_train_total 而整段跳过扩散、wandb 只有 proxy 无 loss。
    # 从**本 run** `.../seed*/checkpoint/` 续训时目录相同，不重置以保留 resume 语义。
    _rft = bool(
        getattr(Config, "real_task_text_only_finetune", False)
        or getattr(Config, "fewshot_text_only_finetune", False)
    )
    if _rft and _ld_path:
        _own_ckpt = os.path.normpath(
            os.path.join(str(logger.prefix), "checkpoint")
        )
        _ld_dir = os.path.normpath(
            os.path.dirname(os.path.abspath(str(_ld_path)))
        )
        if _ld_dir != _own_ckpt:
            _ps = int(getattr(trainer, "step", 0) or 0)
            trainer.step = 0
            trainer.reset_parameters()
            logger.print(
                f"[train] real_task 微调：外源 checkpoint 的 step={_ps} 已置 0，"
                f"本 run 将按 n_train_steps={int(getattr(Config, 'n_train_steps', 0) or 0)} 训练扩散并写入 wandb（loss / per-t）。",
                color="yellow",
            )

    # -----------------------------------------------------------------------------#
    # ------------------------ test forward & backward pass -----------------------#
    # -----------------------------------------------------------------------------#

    utils.report_parameters(model)

    logger.print('Testing forward...', end=' ', flush=True)
    batch = utils.batchify(dataset[0], Config.device)
    loss, _ = diffusion.loss(*batch)
    loss.backward()
    logger.print('✓')

    # -----------------------------------------------------------------------------#
    # --------------------------------- main loop ---------------------------------#
    # -----------------------------------------------------------------------------#
    
    # 单任务且 proxy_filter：训练 proxy；多任务 proxy 在 ensure_multitask_proxies
    if not Config.is_multitask and use_proxy_filter:
        logger.print("🚀 开始训练代理模型... 🚀")
        logger.print(f"📈 代理模型训练参数:")
        logger.print(f"   - 批量大小: {Config.batch_size}")
        logger.print(f"   - 学习率: {Config.proxy_learning_rate}")
        logger.print(f"   - 总训练步数: {Config.proxy_n_train_steps}")
        logger.print(f"   - 设备: {Config.device}")
        
        # 训练代理模型
        trainer.train_proxy(n_train_steps=Config.proxy_n_train_steps)
    
    # 训练扩散模型（支持 resume：trainer.step 可能已由 checkpoint 非零）
    n_train_total = int(Config.n_train_steps)
    spe = int(Config.n_steps_per_epoch)
    total_epochs = max(1, n_train_total // spe)
    if int(trainer.step) >= n_train_total:
        logger.print(
            f"[train] trainer.step={trainer.step} 已达 n_train_steps={n_train_total}，"
            f"跳过扩散训练循环。",
            color="yellow",
        )
    else:
        logger.print("🚀 开始训练扩散模型... 🚀")
        logger.print(f"📈 扩散模型训练参数:")
        logger.print(f"   - 批量大小: {Config.batch_size}")
        logger.print(f"   - 学习率: {Config.learning_rate}")
        logger.print(f"   - 总训练步数: {Config.n_train_steps}")
        logger.print(f"   - 设备: {Config.device}")
        logger.print(f"   - 训练模式: {'多任务' if Config.is_multitask else '单任务'}")

        if wandb is not None:
            wandb.log(
                {
                    "finetune_step": int(trainer.step),
                    "training_mode": "multitask"
                    if Config.is_multitask
                    else "singletask",
                }
            )

        logger.print(
            f"🔄 训练循环（目标总步数 {n_train_total}，每 epoch {spe} 步，"
            f"起始 trainer.step={trainer.step}，epoch 上界 {total_epochs}）"
        )
        while trainer.step < n_train_total:
            cur_epoch = int(trainer.step // spe)
            chunk = min(spe, n_train_total - int(trainer.step))
            logger.print(f"📊 Epoch {cur_epoch} / {total_epochs} | {logger.prefix}")
            trainer.train(n_train_steps=chunk)
            logger.print(f"✅ Epoch {cur_epoch} 段训练完成（chunk={chunk}）")
            if wandb is not None:
                wandb.log(
                    {
                        "finetune_step": int(trainer.step),
                        "epoch": cur_epoch,
                        "total_epochs": total_epochs,
                    }
                )
    
    # 训练完成后的总结
    logger.print("🎉 扩散模型训练完成! 🎉")
    logger.print(f"📊 训练统计:")
    logger.print(f"   - 完成步数: {trainer.step}")
    
    # 记录训练完成信息到wandb
    if wandb is not None:
        wandb.log(
            {
                "finetune_step": int(trainer.step),
                "training_completed": True,
                "final_step": trainer.step,
            }
        )

    # save_freq 可能大于总步数（如微调 4000 步而 save_freq=5000），训练循环内可能从未触发 save
    logger.print("💾 保存最终扩散 checkpoint（含 EMA）…", flush=True)
    trainer.save()


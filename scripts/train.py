import diffuser.utils as utils
import diffuser.models as models
import torch

from tqdm import tqdm
import pickle as pkl

# 由 main() 成功 import wandb 后赋值；失败则为 None（旧 typing_extensions 下 wandb 无法 import，训练仍应能跑）
wandb = None
import os
import glob
from pathlib import Path
from diffuser.models.vae import VAE
from torch.utils.data import DataLoader, ConcatDataset
from diffuser.utils.training import Trainer
from diffuser.cpu_threads import (
    apply_torch_cpu_threads_from_env,
    dataloader_num_workers_cap,
)

apply_torch_cpu_threads_from_env()


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


def multitask_proxy_prefix(task_name, Config):
    """与 evaluate 中单任务 proxy 路径一致。"""
    nd = getattr(Config, "traj_n_traj_dict", None)
    if nd is not None and task_name in nd:
        n = nd[task_name]
        k = getattr(Config, "traj_k_dict", {})[task_name]
        e = getattr(Config, "traj_eps_dict", {})[task_name]
    else:
        n, k, e = Config.n_traj, Config.k, Config.eps
    return (
        f"trained_models/{task_name}_frac{Config.frac}_sigma{Config.sigma}/"
        f"{n}x{Config.horizon}_k{k}_eps{e}/seed{Config.seed}/"
    )


def multitask_proxy_checkpoint_exists(prefix, Config):
    base = os.path.join(prefix, "proxy_checkpoint")
    if os.path.isfile(os.path.join(base, "state.pt")):
        return True
    if getattr(Config, "save_checkpoints", False):
        p = os.path.join(base, f"state_{Config.proxy_n_train_steps}.pt")
        if os.path.isfile(p):
            return True
        if glob.glob(os.path.join(base, "state_*.pt")):
            return True
    return False


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

def train_multitask_vae(tasks_list, latent_dim=32, batch_size=64, n_epochs=100, lr=1e-4, device='cuda'):
    """
    在多个数据集上训练统一的VAE模型
    
    参数:
    - tasks_list: 训练数据集列表
    - latent_dim: VAE的隐空间维度
    - batch_size: 批量大小
    - n_epochs: 训练轮数
    - lr: 学习率
    - device: 训练设备
    
    返回:
    - 训练好的VAE模型
    """
    print(f"开始在以下数据集上训练多任务VAE: {tasks_list}")
    
    # 创建VAE模型（与 train_vae.py 一致：固定 128 维输入）
    vae = VAE(input_dim=128, latent_dim=latent_dim)
    vae.to(device)
    
    # 准备多任务数据集
    datasets = []
    for task in tasks_list:
        print(f"加载数据集: {task}")
        # 为每个任务创建数据加载器配置
        task_dataset_config = utils.Config(
            'datasets.ZipDataset',  # 使用ZipDataset作为数据加载器
            dataset=task,
            frac=1.0,  # 使用全部数据
            sigma=0.0,  # 默认噪声
        )
        task_dataset = task_dataset_config()
        datasets.append(task_dataset)
    
    # 合并所有数据集
    combined_dataset = ConcatDataset(datasets)
    print(f"合并后的数据集大小: {len(combined_dataset)}")
    
    # 创建数据加载器
    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dataloader_num_workers_cap(4),
    )
    
    # 设置优化器
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    
    # 训练循环
    vae.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{n_epochs}'):
            # 假设数据集中的每个元素都是(x, y)格式，我们只需要x进行VAE训练
            x = batch[0].to(device)
            
            # 前向传播
            recon_x, mu, logvar = vae(x)
            
            # 计算VAE损失 (重建损失 + KL散度)
            loss = vae.loss_function(recon_x, x, mu, logvar)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 打印每轮的平均损失
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
        
        # 记录到wandb
        if wandb is not None:
            wandb.log({'vae_epoch': epoch + 1, 'vae_loss': avg_loss})
    
    vae.eval()
    print("多任务VAE训练完成")
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

    RUN._update(deps)
    print(deps)

    from diffuser.utils.multitask_canon import canonical_train_tasks_csv

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

    if getattr(Config, "multitask_text_only", False):
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
    
    if wandb is not None:
        wandb.init(project='decdiff-opt', config=Config, name=run_name)

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

    _maybe_build_task_text_embeddings(Config)

    # 多任务模式下，创建混合数据集
    if hasattr(Config, 'is_multitask') and Config.is_multitask:
        print(f"创建多任务数据集: {Config.train_tasks_list}")
        
        # 加载混合轨迹文件
        from diffuser.datasets.sequence import PointRegretDataset
        import os
        
        # 构建完整的混合轨迹文件路径
        data_dir = os.path.dirname(Config.data_path)
        from diffuser.utils.traj_params import resolve_multitask_mixed_path

        _sig = getattr(Config, "multitask_traj_signature", None)
        mixed_data_path = resolve_multitask_mixed_path(data_dir, _sig)
        
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
        
        proxy_dataset_config = utils.Config(
            Config.proxy_loader,
            dataset=Config.dataset,
            frac=Config.frac,
            sigma=Config.sigma,
            soo_seed=int(getattr(Config, 'seed', 1)),
            savepath='proxy_dataset_config.pkl',
        )
        
        dataset = dataset_config()
        proxy_dataset = proxy_dataset_config()

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
            device=Config.device,
        )
        
        # 单任务模式下，为GaussianInvDynDiffusion模型添加proxy_model_config
        if not Config.is_multitask:
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
        if not Config.is_multitask:
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
            device=Config.device,
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
    )

    # -----------------------------------------------------------------------------#
    # -------------------------------- instantiate --------------------------------#
    # -----------------------------------------------------------------------------#

    # 加载VAE模型用于从隐空间解码到原始空间
    # 原始观测维度已经在前面设置
    
    # 确定VAE模型路径
    fixed_dim = getattr(Config, 'fixed_dim', 128)
    if Config.is_multitask:
        # 与 train_vae.py / construct_trajectories 保存目录一致（含 _dim{fixed_dim}）
        train_tasks_str = '_'.join(Config.train_tasks_list)
        vae_model_path = f"./trained_models/vae/multi_{train_tasks_str}_frac{Config.frac}_sigma{Config.sigma}_dim{fixed_dim}/vae_latent32.pt"
        print(f"多任务模式，VAE模型路径: {vae_model_path}")
    else:
        task_name = Config.train_tasks_list[0]
        vae_model_path = f"./trained_models/vae/{task_name}_frac{Config.frac}_sigma{Config.sigma}_dim{fixed_dim}/vae_latent32.pt"
        print(f"单任务模式，VAE模型路径: {vae_model_path}")
    
    # 确保模型目录存在
    os.makedirs(os.path.dirname(vae_model_path), exist_ok=True)
    
    # 创建VAE模型实例（input_dim 与 train_vae / 轨迹构建一致）
    vae = VAE(input_dim=fixed_dim, latent_dim=32)
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
                latent_dim=32,
                batch_size=64,
                n_epochs=100,
                lr=1e-4,
                device=Config.device,
            )
            # 保存训练好的VAE模型
            torch.save(vae.state_dict(), vae_model_path)
            print(f"多任务VAE模型已保存至: {vae_model_path}")
        else:
            # 单任务VAE训练逻辑，使用原有数据集训练VAE
            print(f"单任务模式: 使用{Config.dataset}数据集训练VAE")
            
            # 加载原始观测数据用于训练VAE
            original_data_path = Config.data_path.replace('_vae_latent32_train.p', '_train.p')
            
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
                        recon_x, mu, logvar = vae(x)
                        loss = vae.loss_function(recon_x, x, mu, logvar)
                        
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
    proxy_model = proxy_model_config() if not Config.is_multitask else None

    diffusion = diffusion_config(model)

    if Config.is_multitask:
        ensure_multitask_proxies(diffusion, dataset, renderer, Config, action_dim, logger)

    # 创建训练器
    trainer = trainer_config(diffusion, proxy_model, dataset, proxy_dataset, renderer)
    trainer.vae = vae  # 添加VAE属性到训练器

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
    
    # 只有在单任务模式下才训练proxy model，多任务模式下不需要
    if not Config.is_multitask:
        logger.print("🚀 开始训练代理模型... 🚀")
        logger.print(f"📈 代理模型训练参数:")
        logger.print(f"   - 批量大小: {Config.batch_size}")
        logger.print(f"   - 学习率: {Config.proxy_learning_rate}")
        logger.print(f"   - 总训练步数: {Config.proxy_n_train_steps}")
        logger.print(f"   - 设备: {Config.device}")
        
        # 训练代理模型
        trainer.train_proxy(n_train_steps=Config.proxy_n_train_steps)
    
    # 训练扩散模型
    logger.print("🚀 开始训练扩散模型... 🚀")
    logger.print(f"📈 扩散模型训练参数:")
    logger.print(f"   - 批量大小: {Config.batch_size}")
    logger.print(f"   - 学习率: {Config.learning_rate}")
    logger.print(f"   - 总训练步数: {Config.n_train_steps}")
    logger.print(f"   - 设备: {Config.device}")
    logger.print(f"   - 训练模式: {'多任务' if Config.is_multitask else '单任务'}")
    
    # 将模式信息添加到wandb
    if wandb is not None:
        wandb.log({'training_mode': 'multitask' if Config.is_multitask else 'singletask'})
    
    n_epochs = int(Config.n_train_steps // Config.n_steps_per_epoch)
    logger.print(f"🔄 开始训练循环，共{n_epochs}个epoch")
    
    for i in range(n_epochs):
        logger.print(f'📊 Epoch {i} / {n_epochs} | {logger.prefix}')
        trainer.train(n_train_steps=Config.n_steps_per_epoch)
        logger.print(f'✅ Epoch {i} 训练完成')
        
        # 记录epoch信息到wandb
        if wandb is not None:
            wandb.log({'epoch': i, 'total_epochs': n_epochs})
    
    # 训练完成后的总结
    logger.print("🎉 扩散模型训练完成! 🎉")
    logger.print(f"📊 训练统计:")
    logger.print(f"   - 完成步数: {trainer.step}")
    
    # 记录训练完成信息到wandb
    if wandb is not None:
        wandb.log({'training_completed': True, 'final_step': trainer.step})


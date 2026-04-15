import os
import sys

if __name__ == "__main__":
    from diffuser.cpu_threads import maybe_apply_from_argv_and_env

    maybe_apply_from_argv_and_env()

import numpy as np
import torch
import diffuser.numpy_design_bench_compat  # noqa: F401

if __name__ == "__main__":
    from diffuser.cpu_threads import apply_torch_cpu_threads_from_env

    apply_torch_cpu_threads_from_env()
from diffuser.models.vae import VAE, train_vae, create_vae_dataloaders
from sklearn.preprocessing import StandardScaler
import argparse

import design_bench
from tqdm import tqdm
import pickle as pkl
from diffuser.datasets.real_world_fewshot import (
    REAL_WORLD_FEWSHOT_TASK_SPECS,
    is_real_world_fewshot_task,
    load_real_world_for_pipeline,
)
from diffuser.utils.soo_gtopx import (
    TASKNAME_TO_VAR_NUM,
    load_gtopx_offline_arrays,
    is_gtopx_task,
)
from diffuser.utils.construct_runtime import resolve_torch_device

# 任务映射
TASKNAME2FULL = {
    'dkitty': design_bench.datasets.continuous.dkitty_morphology_dataset.DKittyMorphologyDataset,
    'ant': design_bench.datasets.continuous.ant_morphology_dataset.AntMorphologyDataset,
    'tfbind8': design_bench.datasets.discrete.tf_bind_8_dataset.TFBind8Dataset,
    'tfbind10': design_bench.datasets.discrete.tf_bind_10_dataset.TFBind10Dataset,
    'superconductor': design_bench.datasets.continuous.superconductor_dataset.SuperconductorDataset,
}

TASKNAME2TASK = {
    'dkitty': 'DKittyMorphology-Exact-v0',
    'ant': 'AntMorphology-Exact-v0',
    'tfbind8': 'TFBind8-Exact-v0',
    'tfbind10': 'TFBind10-Exact-v0',
    'superconductor': 'Superconductor-RandomForest-v0',
}

TASKNAME2MAX_SAMPLES = {
    'dkitty': 10004,
    'ant': 10004,
    'tfbind8': 32898,
    'tfbind10': 50000,
    'superconductor': 17014,
}

# 获取不同任务的原始观测维度
def get_original_observation_dim(task_name):
    """获取不同任务的原始观测维度"""
    dim_map = {
        'dkitty': 56,  # 位置x,y,z、姿态roll,pitch,yaw及其速度
        'ant': 60,     # Ant任务的原始维度
        'tfbind8': 8,  # TFBind8的原始维度
        'tfbind10': 10,# TFBind10的原始维度
        'superconductor': 86,  # 超导体任务的原始维度
        'lunar_lander': 12,
        'robot_push': 14,
        'rover': 60,
    }
    if is_gtopx_task(task_name):
        return TASKNAME_TO_VAR_NUM[task_name]
    return dim_map.get(task_name, 128)  # 默认返回128

def main(args):
    # 设备：--device > 环境变量 GTG_DEVICE > 自动 cuda/cpu（construct 传入的 VAEArgs 无 device 时走后两者）
    explicit = getattr(args, "device", None)
    if isinstance(explicit, str):
        explicit = explicit.strip() or None
    device = resolve_torch_device(explicit)
    print(f"使用设备: {device}（可用 GTG_DEVICE 或 train_vae --device 指定）")
    
    # 创建trained_models目录
    os.makedirs("trained_models/vae", exist_ok=True)
    
    # 确定是单任务模式还是多任务模式
    is_multitask = args.tasks is not None
    
    soo_seed = int(getattr(args, 'seed', 0))

    if is_multitask:
        # 多任务模式（与 construct_trajectories / multi_* 路径一致：字典序）
        task_list = sorted(t.strip() for t in args.tasks.split(",") if t.strip())
        args.tasks = ",".join(task_list)
        print(f"多任务模式，加载任务列表: {task_list}")
        rw_in_mt = [t for t in task_list if is_real_world_fewshot_task(t)]
        if rw_in_mt:
            raise ValueError(
                "few-shot 与 real-world 实验仅支持单任务；多任务列表含 real-world（%s）已禁止。"
                "请只跑一个 real-world 任务，并用 --pretrained_vae_info 在 few-shot 数据上微调 VAE。"
                % ", ".join(rw_in_mt)
            )

        all_data = []
        for task_name in tqdm(task_list, desc="加载任务数据"):
            print(f"加载任务: {task_name}")
            if is_gtopx_task(task_name):
                x_np, _, _, _, _ = load_gtopx_offline_arrays(
                    task_name, frac=args.frac, sigma=0.0, seed=soo_seed
                )
                data_x = torch.from_numpy(x_np).float()
            elif is_real_world_fewshot_task(task_name):
                raise ValueError("多任务不应包含 real-world（已在前面拒绝）")
            else:
                task = design_bench.make(TASKNAME2TASK[task_name],
                                       dataset_kwargs=dict(
                                       max_samples=int(TASKNAME2MAX_SAMPLES[task_name] * args.frac),
                                       distribution=None,
                                       min_percentile=0)
                                   )
                
                if task_name.startswith("tfbind"):
                    task.map_to_logits()
                
                # 获取数据并转换为tensor
                data_x = torch.from_numpy(task.x.reshape(task.x.shape[0], -1)).float()
            all_data.append(data_x)
        
        # 使用固定维度进行统一处理
        input_dim = args.fixed_dim
        print(f"使用固定输入维度: {input_dim}")
        
        # 生成多任务模型保存路径
        task_str = "multi_" + "_".join(task_list)
        
    else:
        # 单任务模式
        if not args.task:
            raise ValueError("必须指定 --task 或 --tasks 参数")
        
        print(f"单任务模式，加载任务: {args.task}")
        if (
            is_real_world_fewshot_task(args.task)
            and not getattr(args, "pretrained_vae_info", None)
        ):
            raise ValueError(
                "real-world 任务需在多任务预训练权重上微调：请指定 --pretrained_vae_info 指向 "
                "仅含 Design-Bench 等任务的 vae_info.p（例如 multi_* 目录下的 vae_info.p）。"
            )
        if is_gtopx_task(args.task):
            x_np, _, _, _, _ = load_gtopx_offline_arrays(
                args.task, frac=args.frac, sigma=0.0, seed=soo_seed
            )
            data_x = torch.from_numpy(x_np).float()
        elif is_real_world_fewshot_task(args.task):
            proc, _, _ = load_real_world_for_pipeline(
                args.task,
                fixed_length=args.fixed_dim,
                frac=args.frac,
                sigma=args.sigma,
                fewshot_k=getattr(args, "fewshot_k", None),
                fewshot_mode=getattr(args, "fewshot_mode", "all"),
                fewshot_seed=int(getattr(args, "fewshot_seed", soo_seed)),
            )
            data_x = torch.from_numpy(np.asarray(proc, dtype=np.float32)).float()
        else:
            task = design_bench.make(TASKNAME2TASK[args.task],
                                    dataset_kwargs=dict(
                                    max_samples=int(TASKNAME2MAX_SAMPLES[args.task] * args.frac),
                                    distribution=None,
                                    min_percentile=0)
                                )
            
            if args.task.startswith("tfbind"):
                    task.map_to_logits()
            
            # 获取数据
            data_x = torch.from_numpy(task.x.reshape(task.x.shape[0], -1)).float()
        all_data = [data_x]
        
        # 使用固定维度
        input_dim = args.fixed_dim
        print(f"使用固定输入维度: {input_dim}")
        
        # 生成单任务模型保存路径
        task_str = args.task

    finetune_rw = False
    pretrained_vae_info_dict = None
    if (
        not is_multitask
        and getattr(args, "task", None)
        and is_real_world_fewshot_task(args.task)
        and getattr(args, "pretrained_vae_info", None)
    ):
        finetune_rw = True
        with open(args.pretrained_vae_info, "rb") as f:
            pretrained_vae_info_dict = pkl.load(f)
        args.latent_dim = int(pretrained_vae_info_dict.get("latent_dim", args.latent_dim))
        _pfd = int(pretrained_vae_info_dict.get("fixed_dim", input_dim))
        if _pfd != input_dim:
            print(
                f"注意: 预训练 vae_info 中 fixed_dim={_pfd}，当前 input_dim={input_dim}；"
                "请与预训练时 --fixed_dim 一致。"
            )

    # 设置VAE模型参数
    vae_config = {
        'input_dim': input_dim,
        'latent_dim': args.latent_dim,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    }

    _rw_suffix = "_rwft" if finetune_rw else ""
    # 生成模型保存路径
    model_save_dir = f"trained_models/vae/{task_str}_frac{args.frac}_sigma{args.sigma}_dim{input_dim}{_rw_suffix}"
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"vae_latent{args.latent_dim}.pt")
    
    # 检查是否已存在训练好的模型
    if os.path.exists(model_path) and not args.force_retrain:
        print(f"VAE模型已存在，加载预训练模型: {model_path}")
        # 创建模型实例
        vae = VAE(**vae_config)
        vae.load_state_dict(torch.load(model_path, map_location=device))
        vae.to(device)
        
        # 加载缩放器
        if is_multitask:
            # 多任务模式，加载每个任务的scaler
            scalers = []
            for task_name in task_list:
                scaler_path = os.path.join(model_save_dir, f"scaler_{task_name}.p")
                if not os.path.exists(scaler_path):
                    raise FileNotFoundError(f"未找到任务 {task_name} 的scaler文件: {scaler_path}")
                scaler_dict = pkl.load(open(scaler_path, "rb"))
                scaler = StandardScaler()
                scaler.mean_ = scaler_dict['mean']
                scaler.scale_ = scaler_dict['scale']
                scaler.var_ = scaler_dict['var']
                scaler.n_samples_seen_ = scaler_dict['n_samples_seen']
                scaler.n_features_in_ = len(scaler.mean_)
                scalers.append(scaler)
            # 对于多任务，返回scalers列表
            scaler = scalers
        else:
            # 单任务模式，加载单个scaler
            scaler_path = os.path.join(model_save_dir, f"scaler_{task_str}.p")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"未找到任务 {task_str} 的scaler文件: {scaler_path}")
            scaler_dict = pkl.load(open(scaler_path, "rb"))
            scaler = StandardScaler()
            scaler.mean_ = scaler_dict['mean']
            scaler.scale_ = scaler_dict['scale']
            scaler.var_ = scaler_dict['var']
            scaler.n_samples_seen_ = scaler_dict['n_samples_seen']
            scaler.n_features_in_ = len(scaler.mean_)
        
        print("模型和缩放器加载完成！")
        return vae, scaler, model_save_dir
    
    print(f"准备训练VAE模型，输入维度: {input_dim}，隐空间维度: {args.latent_dim}")
    
    # 准备所有任务的数据，为每个任务使用独立的scaler
    all_processed_data = []
    scalers = []
    
    # 处理每个数据集，使用独立的scaler
    for i, data in enumerate(tqdm(all_data, desc="处理数据集")):
        x_np = data.numpy()
        
        # 为当前任务创建独立的scaler
        task_scaler = StandardScaler()
        x_scaled = task_scaler.fit_transform(x_np)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        
        # 数据填充或截断以适应固定输入维度
        if x_tensor.size(1) < input_dim:
            x_processed = torch.zeros(x_tensor.size(0), input_dim, dtype=torch.float32)
            x_processed[:, :x_tensor.size(1)] = x_tensor
        elif x_tensor.size(1) > input_dim:
            x_processed = x_tensor[:, :input_dim]
        else:
            x_processed = x_tensor
        
        all_processed_data.append(x_processed)
        scalers.append(task_scaler)
    
    # 合并所有处理后的数据
    x_processed = torch.cat(all_processed_data, dim=0)
    print(f"合并后的数据集大小: {x_processed.shape}")
    
    # 记录数据集信息，包含每个任务的原始维度和scaler信息
    dataset_info = {
        'is_multitask': is_multitask,
        'tasks': task_list if is_multitask else [task_str],
        'total_samples': x_processed.shape[0],
        'fixed_dim': input_dim,
        'individual_sizes': [data.shape[0] for data in all_data],
        'original_dims': [data.shape[1] for data in all_data]  # 记录每个任务的原始维度
    }
    print(f"数据集信息: {dataset_info}")
    # 保存数据集信息
    pkl.dump(dataset_info, open(os.path.join(model_save_dir, "dataset_info.p"), "wb"))
    
    # 保存每个任务的scaler
    for i, (task_name, scaler) in enumerate(zip(dataset_info['tasks'], scalers)):
        scaler_path = os.path.join(model_save_dir, f"scaler_{task_name}.p")
        scaler_dict = {
            'mean': scaler.mean_,
            'scale': scaler.scale_,
            'var': scaler.var_,
            'n_samples_seen': scaler.n_samples_seen_,
            'n_features_in_': scaler.n_features_in_
        }
        pkl.dump(scaler_dict, open(scaler_path, "wb"))
        print(f"scaler参数已保存到: {scaler_path}")
    
    # 创建数据加载器（few-shot 微调时用较小 batch，避免 train_loader 为空）
    n_s = int(x_processed.shape[0])
    vs = float(args.val_split)
    if finetune_rw and n_s < 32:
        vs = max(vs, min(0.25, 2.0 / max(n_s, 1)))
    eff_batch = min(int(args.batch_size), max(2, n_s - max(1, int(n_s * vs))))
    train_loader, val_loader = create_vae_dataloaders(
        x_processed,
        batch_size=eff_batch,
        val_split=vs,
    )

    # 创建VAE模型
    vae = VAE(**vae_config)
    vae.to(device)
    if finetune_rw and pretrained_vae_info_dict:
        pt_path = pretrained_vae_info_dict.get("vae_path")
        if not pt_path or not os.path.isfile(pt_path):
            raise FileNotFoundError(f"预训练 VAE 权重不存在: {pt_path}")
        vae.load_state_dict(torch.load(pt_path, map_location=device))
        print(f"已从 {pt_path} 加载权重，在 few-shot 数据上微调（lr={getattr(args, 'finetune_lr', 3e-5)}）")

    opt_lr = float(getattr(args, "finetune_lr", args.lr)) if finetune_rw else args.lr
    n_ep = int(getattr(args, "finetune_epochs", args.num_epochs)) if finetune_rw else args.num_epochs
    _fkl = getattr(args, "finetune_kl_weight", None)
    kl_w = (
        float(_fkl if _fkl is not None else args.kl_weight)
        if finetune_rw
        else float(args.kl_weight)
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=opt_lr,
        weight_decay=args.weight_decay,
    )

    # 训练VAE
    phase = "微调" if finetune_rw else "训练"
    print(f"开始{phase}VAE模型，共{n_ep}轮（lr={opt_lr}, kl_weight={kl_w}）")
    train_losses, val_losses = train_vae(
        vae,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=n_ep,
        kl_weight=kl_w,
    )
    
    # 保存模型
    torch.save(vae.state_dict(), model_path)
    print(f"VAE模型已保存到: {model_path}")
    
    # 对于单任务模式，保持向后兼容，同时保存单个scaler文件
    if not is_multitask:
        # 保存单个scaler的兼容文件
        scaler_dict = {
            'mean': scalers[0].mean_,
            'scale': scalers[0].scale_,
            'var': scalers[0].var_,
            'n_samples_seen': scalers[0].n_samples_seen_
        }
        pkl.dump(scaler_dict, open(os.path.join(model_save_dir, f"scaler_{task_str}.p"), "wb"))
    
    # 保存VAE配置
    vae_info = {
        'config': vae_config,
        'input_dim': input_dim,
        'model_path': model_path,
        'vae_path': model_path,
        'dataset_info': dataset_info,
    }
    if finetune_rw:
        vae_info['pretrained_vae_info_source'] = os.path.abspath(args.pretrained_vae_info)
    pkl.dump(vae_info, open(os.path.join(model_save_dir, "vae_info.p"), "wb"))
    
    print("VAE训练完成！")
    # 根据是否是多任务模式返回不同格式的scaler
    if is_multitask:
        return vae, scalers, model_save_dir
    else:
        return vae, scalers[0], model_save_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练VAE模型用于降维")
    
    # 任务参数
    _all_tasks = sorted(
        set(TASKNAME2TASK.keys())
        | set(TASKNAME_TO_VAR_NUM.keys())
        | set(REAL_WORLD_FEWSHOT_TASK_SPECS.keys())
    )
    parser.add_argument('--task', type=str, choices=_all_tasks, help='单任务名称')
    parser.add_argument('--seed', type=int, default=0, help='SOO GTOPX 离线采样随机种子（与 construct_trajectories --seed 对齐）')
    parser.add_argument('--tasks', type=str, help='多任务列表，用逗号分隔，例如: dkitty,ant,tfbind8')
    parser.add_argument('--frac', type=float, default=1.0, help='使用数据集的比例')
    parser.add_argument('--sigma', type=float, default=0.0, help='噪声标准差')
    parser.add_argument('--force_retrain', action='store_true', help='强制重新训练')
    parser.add_argument('--fixed_dim', type=int, default=128, help='固定的输入维度，用于统一不同数据集的长度')
    parser.add_argument(
        '--fewshot_k',
        type=int,
        default=None,
        help='real-world：few-shot 点数（与 construct 一致；None=全量）',
    )
    parser.add_argument(
        '--fewshot_mode',
        type=str,
        default='all',
        choices=('all', 'random', 'worst'),
        help='real-world：few-shot 采样方式',
    )

    # VAE模型参数
    parser.add_argument('--latent_dim', type=int, default=32, help='隐空间维度')
    parser.add_argument('--d_model', type=int, default=256, help='Transformer模型维度')
    parser.add_argument('--nhead', type=int, default=4, help='注意力头数量')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout概率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--val_split', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--kl_weight', type=float, default=0.1, help='KL散度权重')
    parser.add_argument(
        '--pretrained_vae_info',
        type=str,
        default=None,
        help='real-world：多任务预训练 vae_info.p，用于 few-shot 微调',
    )
    parser.add_argument('--finetune_epochs', type=int, default=50, help='real-world 微调轮数')
    parser.add_argument('--finetune_lr', type=float, default=3e-5, help='real-world 微调学习率')
    parser.add_argument(
        '--finetune_kl_weight',
        type=float,
        default=None,
        help='real-world 微调时的 KL 权重；默认与 --kl_weight 相同',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='训练 VAE 的设备：cuda / cuda:0 / cpu；默认空表示用环境变量 GTG_DEVICE 或自动选择',
    )

    args = parser.parse_args()
    main(args)
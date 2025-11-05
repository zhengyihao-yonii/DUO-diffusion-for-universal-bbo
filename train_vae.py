import os
import sys
import torch
import numpy as np
from diffuser.models.vae import VAE, train_vae, create_vae_dataloaders
from sklearn.preprocessing import StandardScaler
import argparse
import design_bench
from tqdm import tqdm
import pickle as pkl

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

TASKNAME2MAX_SAMPLES ={
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
    }
    return dim_map.get(task_name, 128)  # 默认返回128

def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建trained_models目录
    os.makedirs("trained_models/vae", exist_ok=True)
    
    # 加载任务数据
    print(f"加载任务: {args.task}")
    task = design_bench.make(TASKNAME2TASK[args.task],
                            dataset_kwargs=dict(
                            max_samples=int(TASKNAME2MAX_SAMPLES[args.task] * args.frac),
                            distribution=None,
                            min_percentile=0)
                        )
    
    # 获取数据
    data_x = task.x
    
    # 为不同任务设置合适的输入维度
    original_dim = get_original_observation_dim(args.task)
    input_dim = max(original_dim, 128)  # 确保至少128维，或使用原始维度
    
    # 设置VAE模型参数
    vae_config = {
        'input_dim': input_dim,
        'latent_dim': args.latent_dim,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    }
    
    # 生成模型保存路径
    model_save_dir = f"trained_models/vae/{args.task}_frac{args.frac}_sigma{args.sigma}"
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
        scaler = StandardScaler()
        scaler.mean_ = np.load(os.path.join(model_save_dir, "scaler_mean.npy"))
        scaler.scale_ = np.load(os.path.join(model_save_dir, "scaler_scale.npy"))
        # 设置n_features_in_属性，确保scaler.transform()正常工作
        scaler.n_features_in_ = len(scaler.mean_)
        
        print("模型和缩放器加载完成！")
        return vae, scaler, model_save_dir
    
    print(f"准备训练VAE模型，输入维度: {input_dim}，隐空间维度: {args.latent_dim}")
    
    # 准备数据
    x_np = data_x
    # 标准化数据
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_np)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    
    # 数据填充或截断以适应固定输入维度
    if x_tensor.size(1) < input_dim:
        x_processed = torch.zeros(x_tensor.size(0), input_dim, dtype=torch.float32)
        x_processed[:, :x_tensor.size(1)] = x_tensor
    elif x_tensor.size(1) > input_dim:
        x_processed = x_tensor[:, :input_dim]
    else:
        x_processed = x_tensor
    
    # 创建数据加载器
    train_loader, val_loader = create_vae_dataloaders(
        x_processed, 
        batch_size=args.batch_size, 
        val_split=args.val_split
    )
    
    # 创建VAE模型
    vae = VAE(**vae_config)
    vae.to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        vae.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # 训练VAE
    print(f"开始训练VAE模型，共{args.num_epochs}轮")
    train_losses, val_losses = train_vae(
        vae, 
        train_loader, 
        val_loader, 
        optimizer, 
        device, 
        num_epochs=args.num_epochs,
        kl_weight=args.kl_weight
    )
    
    # 保存模型
    torch.save(vae.state_dict(), model_path)
    print(f"VAE模型已保存到: {model_path}")
    
    # 保存缩放器参数
    np.save(os.path.join(model_save_dir, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(model_save_dir, "scaler_scale.npy"), scaler.scale_)
    
    # 保存VAE配置
    vae_info = {
        'config': vae_config,
        'original_dim': original_dim,
        'model_path': model_path
    }
    pkl.dump(vae_info, open(os.path.join(model_save_dir, "vae_info.p"), "wb"))
    
    print("VAE训练完成！")
    return vae, scaler, model_save_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练VAE模型用于降维")
    
    # 任务参数
    parser.add_argument('--task', type=str, choices=list(TASKNAME2TASK.keys()), default='dkitty')
    parser.add_argument('--frac', type=float, default=1.0, help='使用数据集的比例')
    parser.add_argument('--sigma', type=float, default=0.0, help='噪声标准差')
    parser.add_argument('--force_retrain', action='store_true', help='强制重新训练')
    
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
    
    args = parser.parse_args()
    main(args)
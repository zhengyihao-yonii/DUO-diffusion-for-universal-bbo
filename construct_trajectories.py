import os
import sys
import random
import argparse
from tqdm import tqdm
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pickle as pkl
from tqdm import tqdm
from diffuser.utils import set_seed
from diffuser.models.vae import VAE
import torch.nn.functional as F
from train_vae import main as train_vae_main, get_original_observation_dim

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    import design_bench

    from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
    from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
    from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
    from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
    from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset
    
import torch
import numpy as np

TASKNAME2FULL = {
    'dkitty': DKittyMorphologyDataset,
    'ant': AntMorphologyDataset,
    'tfbind8': TFBind8Dataset,
    'tfbind10': TFBind10Dataset,
    'superconductor': SuperconductorDataset,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, choices=list(TASKNAME2TASK.keys()), default='dkitty')
    parser.add_argument('--frac', type=float, default=1.0)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)

    task = design_bench.make(TASKNAME2TASK[args.task],
                                dataset_kwargs=dict(
                                max_samples=int(TASKNAME2MAX_SAMPLES[args.task] * args.frac),
                                distribution=None,
                                min_percentile=0)
                            )
    fully_observed_task = TASKNAME2FULL[args.task]()

    data_x = task.x
    data_y = task.y
    
    print("bigger dataset min max", fully_observed_task.y.min(), fully_observed_task.y.max())
    print("smaller dataset min max", data_y.min(), data_y.max())
    
    # normalise
    data_y = (data_y - fully_observed_task.y.min()) / (fully_observed_task.y.max() - fully_observed_task.y.min())
    data_y = np.clip(data_y + np.random.randn(*data_y.shape) * args.sigma, 0.0, 1.0)
    print("data_x shape", data_y.shape)
    data_y = data_y.squeeze(-1)
    print("data_y shape", data_y.shape)
    
    data_x = torch.tensor(data_x)
    data_y = torch.tensor(data_y)
    
    N = data_x.shape[0]
    D = data_x.shape[1]
    
    # VAE配置
    latent_dim = 32  # 隐空间维度
    
    # 创建命令行参数对象以调用train_vae_main
    class VAEArgs:
        def __init__(self):
            self.task = args.task
            self.frac = args.frac
            self.sigma = args.sigma
            self.force_retrain = False
            self.latent_dim = latent_dim
            self.d_model = 256
            self.nhead = 4
            self.num_layers = 4
            self.dropout = 0.1
            self.batch_size = 64
            self.val_split = 0.1
            self.lr = 1e-4
            self.weight_decay = 1e-5
            self.num_epochs = 100
            self.kl_weight = 0.1
    
    vae_args = VAEArgs()
    
    def reduce_dimension(vae, data, scaler=None):
        """使用VAE降维数据"""
        vae.eval()
        device = next(vae.parameters()).device
        
        # 标准化数据
        if scaler is not None:
            x_np = data.numpy()
            x_scaled = scaler.transform(x_np)
            x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        else:
            x_tensor = data.clone().float()
        
        # 处理大型数据集，分批进行降维
        batch_size = 1024
        latent_representations = []
        
        with torch.no_grad():
            for i in range(0, len(x_tensor), batch_size):
                batch = x_tensor[i:i+batch_size].to(device)
                
                # 使用get_latent方法获取隐变量表示
                z = vae.get_latent(batch)
                latent_representations.append(z.cpu())
        
        # 合并所有批次的结果
        return torch.cat(latent_representations, dim=0)
    
    # 训练或加载VAE模型
    print(f"训练或加载VAE模型，任务: {args.task}")
    vae, scaler, model_save_dir = train_vae_main(vae_args)
    
    # 获取原始观测维度
    original_observation_dim = get_original_observation_dim(args.task)
    print(f"原始观测维度: {original_observation_dim}")
    
    # 降维
    print("使用VAE进行降维...")
    points_latent = reduce_dimension(vae, data_x, scaler)
    print(f"降维后特征维度: {points_latent.shape[1]}")
    
    # 使用降维后的特征替换原始特征
    points = points_latent
    values = data_y 
    
    # Configs for each task
    traj_len = 64
    num_trajectories = {
        "tfbind8": 1000,
        "tfbind10": 1000,
        "superconductor": 4000,
        "ant": 4000,
        "dkitty": 4000,
    }

    k = {
        "tfbind8": 50,
        "tfbind10": 50,
        "superconductor": 20,
        "ant": 20,
        "dkitty": 20,
    }

    eps = {
        "tfbind8": 0.05,
        "tfbind10": 0.05,
        "superconductor": 0.05,
        "ant": 0.05,
        "dkitty": 0.01,
    }
    
    # Pre-compute Distance Matrix (使用降维后的特征)
    os.makedirs(f"./generated_datasets/{args.task}_frac{args.frac}_sigma{args.sigma}", exist_ok=True)
    distance_file = f"./generated_datasets/{args.task}_frac{args.frac}_sigma{args.sigma}/distance_vae.p"
    if os.path.exists(distance_file):
        distances = pkl.load(open(distance_file, "rb"))
    else:
        distances = torch.zeros((N, N))
        # 在降维空间中使用欧氏距离
        for i in tqdm(range(N)):
            distances[i, :] = torch.sqrt(torch.sum((points[i] - points)**2, dim=1))
        pkl.dump(distances, open(distance_file, "wb"))
        
    
    # Construct Trajectories
    start_percentile = np.percentile(values, 20)
    start_candidates_idx = np.arange(N)[values >= start_percentile]
    
    trajectories = []
    for i in tqdm(range(num_trajectories[args.task])):
        idx_list = []
        trajectory = []
        
        starting_idx = start_candidates_idx[np.random.randint(0, len(start_candidates_idx))]
        # 使用降维后的表示作为起始点
        starting_point = points[starting_idx]
        starting_value = values[[starting_idx]]
        
        idx_list.append(starting_idx)
        trajectory.append(np.concatenate([starting_point, starting_value], axis=0))
        
        for j in range(traj_len-1):
            distances[starting_idx, np.array(idx_list)] = 1000.0
            candidate_idxs = np.arange(N)[values >= starting_value-eps[args.task]]

            if len(candidate_idxs) <= 1:
                candidate_idxs = np.argsort(distances[starting_idx])[:k[args.task]]
                candidate_idx = candidate_idxs[np.argmax(values[candidate_idxs])].item()
            else:
                candidate_idxs = candidate_idxs[np.argsort(distances[starting_idx, candidate_idxs])[:k[args.task]]]
                candidate_idx = np.random.choice(candidate_idxs)
            
            # 使用降维后的表示作为候选点
            candidate_point = points[candidate_idx]
            candidate_value = values[[candidate_idx]]
            trajectory.append(np.concatenate([candidate_point, candidate_value], axis=0))
            
            starting_idx = candidate_idx
            starting_point = candidate_point
            starting_value = max(starting_value, candidate_value)
            idx_list.append(starting_idx)

        trajectory = np.stack(trajectory, axis=0)
        trajectories.append(trajectory)
    trajectories = torch.from_numpy(np.stack(trajectories, axis=0)).float()
    print(trajectories.shape)
    
    our_data = trajectories[..., :-1]
    our_data_vals = trajectories[..., -1]
    pr = 1.0 - our_data_vals
    cumulative_regret_to_go = torch.flip(torch.cumsum(torch.flip(pr, [1]), 1), [1])
    timesteps = torch.arange(traj_len).repeat(num_trajectories[args.task], 1)

    # 保存VAE相关信息（包含原始观测维度）
    vae_info = {
        'latent_dim': latent_dim,
        'vae_path': os.path.join(model_save_dir, f"vae_latent{latent_dim}.pt"),
        'original_observation_dim': original_observation_dim
    }
    pkl.dump(vae_info, open(f"./generated_datasets/{args.task}_frac{args.frac}_sigma{args.sigma}/vae_info.p", "wb"))

    obj = [our_data, our_data_vals, pr, cumulative_regret_to_go, timesteps]
    pkl.dump(obj, open(f"./generated_datasets/{args.task}_frac{args.frac}_sigma{args.sigma}/{num_trajectories[args.task]}x{traj_len}_k{k[args.task]}_eps{eps[args.task]}_vae_latent{latent_dim}_train.p", "wb"))
    print(f"使用VAE降维后的轨迹数据已保存")

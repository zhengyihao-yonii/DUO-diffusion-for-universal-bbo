import os
import sys
import random
import argparse
from tqdm import tqdm
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pickle as pkl
from diffuser.utils import set_seed
from diffuser.models.vae import VAE
import torch.nn.functional as F
from train_vae import main as train_vae_main
from sklearn.preprocessing import StandardScaler
from dataset_utils import MultiDatasetLoader, save_dataset_info, load_dataset_info

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

from diffuser.datasets.sequence import TASKNAME2FULL, TASKNAME2TASK, TASKNAME2MAX_SAMPLES, SUPPORTED_TASKS
from diffuser.utils.soo_gtopx import TASKNAME_TO_VAR_NUM, load_gtopx_offline_arrays, is_gtopx_task
from diffuser.utils.construct_runtime import pairwise_l2_distance_matrix

def preprocess_data_for_vae(data_x, fixed_dim):
    """
    预处理数据，统一长度填充或截断
    
    参数:
    - data_x: 输入数据张量
    - fixed_dim: 固定的目标维度
    
    返回:
    - 处理后的数据张量
    """
    # 如果数据维度小于固定维度，则填充零
    if data_x.size(1) < fixed_dim:
        # 创建零张量并填充原始数据
        padded_data = torch.zeros(data_x.size(0), fixed_dim, dtype=data_x.dtype)
        padded_data[:, :data_x.size(1)] = data_x
        return padded_data
    # 如果数据维度大于固定维度，则截断
    elif data_x.size(1) > fixed_dim:
        return data_x[:, :fixed_dim]
    # 维度匹配，直接返回
    else:
        return data_x

def reduce_dimension(vae, data_x, scaler, fixed_dim=None, task_start_indices=None, dataset_info=None):
    """
    使用VAE对数据进行降维，支持单任务和多任务模式
    
    参数:
    - vae: 训练好的VAE模型
    - data_x: 原始数据
    - scaler: 标准化器，对于多任务是字典形式 {task_name: scaler}
    - fixed_dim: 固定的输入维度（可选）
    - task_start_indices: 任务起始索引字典（可选）
    - dataset_info: 数据集信息字典（可选）
    
    返回:
    - points_latent: 降维后的数据
    """
    vae.eval()
    device = next(vae.parameters()).device
    data_x = data_x.clone().float()
    
    # 标准化数据
    data_x_np = data_x.numpy()
    
    # 多任务模式：对每个任务使用对应的scaler
    if isinstance(scaler, list) and task_start_indices and dataset_info:
        # 将list类型的scaler转换为dict类型，键为任务名称
        scaler_dict = {}
        for task_name, task_scaler in zip(dataset_info.keys(), scaler):
            scaler_dict[task_name] = task_scaler
        
        for task_name in task_start_indices:
            start_idx, end_idx = task_start_indices[task_name]
            # 与 VAE 训练时一致：优先使用 scaler 的特征数（如 tfbind 经 map_to_logits 后与 TASKNAME2DIM 不同）
            sc = scaler_dict[task_name]
            n_feat = getattr(sc, "n_features_in_", None)
            if n_feat is None and hasattr(sc, "mean_") and sc.mean_ is not None:
                n_feat = len(sc.mean_)
            original_dim = int(n_feat) if n_feat else dataset_info[task_name]["original_dim"]
            # 仅使用原始维度的数据进行标准化
            task_x_original = data_x_np[start_idx:end_idx, :original_dim]
            # 使用该任务的scaler进行标准化
            task_x_scaled = sc.transform(task_x_original)
            # 将标准化后的数据放回原位置的前original_dim列
            data_x_np[start_idx:end_idx, :original_dim] = task_x_scaled
    elif scaler is not None:
        # 单任务模式：对所有数据使用同一个scaler
        data_x_np = scaler.transform(data_x_np)
    
    # 转换回张量
    data_x = torch.tensor(data_x_np, dtype=torch.float32)
    
    # 处理大型数据集，分批进行降维
    batch_size = 1024
    latent_representations = []
    
    with torch.no_grad():
        for i in range(0, len(data_x), batch_size):
            batch = data_x[i:i+batch_size].to(device)
            
            # 使用get_latent方法获取隐变量表示
            z = vae.get_latent(batch)
            latent_representations.append(z.cpu())
    
    # 合并所有批次的结果
    return torch.cat(latent_representations, dim=0)

def construct_trajectories(tasks_list, frac=1.0, sigma=0.0, seed=0, n_traj=None, k=None, eps=None, fixed_dim=128, horizon=64):
    """
    构建轨迹，支持单任务和多任务模式
    
    参数:
    - tasks_list: 任务列表
    - frac: 每个任务使用的数据比例
    - sigma: 噪声水平
    - seed: 随机种子
    - n_traj: 每任务生成的轨迹数量（可选）
    - k: 每步选择的候选点数量（可选）
    - eps: 允许的目标值下降范围（可选）
    - fixed_dim: 固定的输入维度，用于统一不同数据集
    - horizon: 每条轨迹长度（时间步数），需与后续扩散训练 Config.horizon 一致
    """
    set_seed(seed)
    traj_len = horizon
    
    # Configs for each task (与dfgo-main保持一致)
    # 默认参数配置
    default_num_trajectories = {
        "tfbind8": 1000,
        "tfbind10": 1000,
        "superconductor": 4000,
        "ant": 4000,
        "dkitty": 4000,
        "gtopx2": 2000,
        "gtopx3": 2000,
        "gtopx4": 2000,
        "gtopx6": 2000,
    }

    default_k = {
        "tfbind8": 50,
        "tfbind10": 50,
        "superconductor": 20,
        "ant": 20,
        "dkitty": 20,
        "gtopx2": 20,
        "gtopx3": 20,
        "gtopx4": 20,
        "gtopx6": 20,
    }

    default_eps = {
        "tfbind8": 0.05,
        "tfbind10": 0.05,
        "superconductor": 0.05,
        "ant": 0.05,
        "dkitty": 0.01,
        "gtopx2": 0.05,
        "gtopx3": 0.05,
        "gtopx4": 0.05,
        "gtopx6": 0.05,
    }
    
    # 统一设置参数，符合用户要求
    # 处理n_traj参数：如果是整数，转换为字典格式
    if n_traj is None:
        n_traj = {task: default_num_trajectories.get(task, 1000) for task in tasks_list}
    elif isinstance(n_traj, int):
        # 如果用户传递的是整数，为每个任务设置相同的轨迹数量
        n_traj = {task: n_traj for task in tasks_list}
    
    # 处理k参数：如果是整数，转换为字典格式
    if k is None:
        k = {task: default_k.get(task, 20) for task in tasks_list}
    elif isinstance(k, int):
        k = {task: k for task in tasks_list}
    
    # 处理eps参数：如果是浮点数，转换为字典格式
    if eps is None:
        eps = {task: default_eps.get(task, 0.05) for task in tasks_list}
    elif isinstance(eps, float):
        eps = {task: eps for task in tasks_list}
    
    # 根据任务数量选择不同的数据加载方式
    is_multitask = len(tasks_list) > 1

    # 与 train/evaluate 中 generated_datasets 路径一致；USE_RETURNS 只改 checkpoint，不改此目录
    if is_multitask:
        tasks_str = "_".join(tasks_list)
        output_dir_early = f"./generated_datasets/multi_{tasks_str}_frac{frac}_sigma{sigma}"
    else:
        output_dir_early = f"./generated_datasets/{tasks_list[0]}_frac{frac}_sigma{sigma}"

    # 产物已存在则跳过数据加载、VAE、降维（避免仅加 USE_RETURNS 仍整段重跑 Step 1）
    if is_multitask:
        mixed_path = os.path.join(output_dir_early, "mixed_trajectories_train.p")
        if os.path.isfile(mixed_path):
            print(
                f"已存在多任务混合轨迹（训练实际加载此文件），跳过数据加载与 VAE：{mixed_path}"
            )
            return output_dir_early, None
    else:
        tn = tasks_list[0]
        task_pkl = os.path.join(
            output_dir_early,
            f"{tn}_{n_traj[tn]}x{traj_len}_k{k[tn]}_eps{eps[tn]}_vae_latent32_train.p",
        )
        if os.path.isfile(task_pkl):
            print(f"已存在轨迹文件，跳过数据加载与 VAE：{task_pkl}")
            return output_dir_early, None

    # 数据加载和预处理
    if is_multitask:
        print(f"开始加载多任务数据，任务列表: {tasks_list}，固定输入维度: {fixed_dim}")
        
        # 使用MultiDatasetLoader加载多数据集
        multi_loader = MultiDatasetLoader(task_list=tasks_list, fixed_length=fixed_dim)
        all_x, all_y, task_start_indices, dataset_info = multi_loader.get_combined_data(frac=frac, sigma=sigma)
        
        # 转换为张量，确保与单任务模式一致
        data_x = torch.tensor(all_x)
        data_y = torch.tensor(all_y)
        # 确保data_y是1维的，与dfgo-main保持一致
        data_y = data_y.squeeze(-1)
        
        # 根据任务数量创建VAEArgs对象
        class VAEArgs:
            def __init__(self):
                if len(tasks_list) > 1:
                    # 多任务模式
                    self.tasks = ','.join(tasks_list)
                    self.task = None
                else:
                    # 单任务模式
                    self.tasks = None
                    self.task = tasks_list[0]
                
                self.frac = frac
                self.sigma = sigma
                self.fixed_dim = fixed_dim
                self.force_retrain = False
                self.latent_dim = 32  # 固定隐空间维度
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
                self.seed = seed
        
        vae_args = VAEArgs()
        
        # 记录任务维度信息
        task_dims_info = {}
        for task_name in tasks_list:
            task_dims_info[task_name] = {
                'original_dim': dataset_info[task_name]['original_dim'],
                'fixed_dim': fixed_dim
            }
        print(f"任务维度信息: {task_dims_info}")
        
    else:
        # 单任务模式
        task_name = tasks_list[0]
        print(f"开始加载单任务数据: {task_name}")
        
        if is_gtopx_task(task_name):
            data_x_np, data_y_np, y_full_min, y_full_max, _ = load_gtopx_offline_arrays(
                task_name, frac=frac, sigma=sigma, seed=seed
            )
            data_x = torch.tensor(data_x_np, dtype=torch.float32)
            data_y = torch.tensor(data_y_np, dtype=torch.float32)
            print("GTOPX 参考 y 范围 [全样本分位]", y_full_min, y_full_max)
            print("离线集 y 归一化后 min/max", float(data_y.min()), float(data_y.max()))
        else:
            # 加载单个 Design-Bench 数据集
            task = design_bench.make(TASKNAME2TASK[task_name],
                                    dataset_kwargs=dict(
                                    max_samples=int(TASKNAME2MAX_SAMPLES[task_name] * frac),
                                    distribution=None,
                                    min_percentile=0)
                                )
            fully_observed_task = TASKNAME2FULL[task_name]()

            if task_name.startswith("tfbind"):
                task.map_to_logits()
            
            # 预处理数据
            data_x = task.x.reshape(task.x.shape[0], -1).astype(np.float32)
            data_y = task.y
            
            print("bigger dataset min max", fully_observed_task.y.min(), fully_observed_task.y.max())
            print("smaller dataset min max", data_y.min(), data_y.max())
            
            # 归一化y值
            data_y = (data_y - fully_observed_task.y.min()) / (fully_observed_task.y.max() - fully_observed_task.y.min())
            data_y = np.clip(data_y + np.random.randn(*data_y.shape) * sigma, 0.0, 1.0)
            # 与dfgo-main保持一致：先squeeze，再转换为张量
            data_y = data_y.squeeze(-1)
            data_x = torch.tensor(data_x)
            data_y = torch.tensor(data_y)
        
        # 创建VAE参数
        class VAEArgs:
            def __init__(self):
                self.task = task_name
                self.tasks = None  # 单任务模式下tasks为None
                self.frac = frac
                self.sigma = sigma
                self.fixed_dim = fixed_dim
                self.force_retrain = False
                self.latent_dim = 32
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
                self.seed = seed
        
        vae_args = VAEArgs()
        
        # 单任务的任务维度信息
        odim = TASKNAME_TO_VAR_NUM[task_name] if is_gtopx_task(task_name) else data_x.shape[1]
        task_dims_info = {
            task_name: {
                'original_dim': odim,
                'fixed_dim': fixed_dim
            }
        }
        # 单任务的任务起始索引
        task_start_indices = {
            task_name: (0, len(data_x))
        }
    
    print(f"数据加载完成，数据集大小: {len(data_x)} 点")
    
    # 训练或加载VAE模型
    print(f"训练或加载{'多任务' if is_multitask else '单任务'}VAE模型")
    try:
        vae, scaler, model_save_dir = train_vae_main(vae_args)
        print(f"成功训练/加载VAE模型: {model_save_dir}")
    except Exception as e:
        print(f"训练VAE模型时出错: {e}")
        raise
    
    # 保存维度信息
    dims_info_path = os.path.join(model_save_dir, "task_dims_info.p")
    save_dataset_info(task_dims_info, dims_info_path)
    print(f"任务维度信息已保存至: {dims_info_path}")
    
    # 使用VAE进行降维...
    print("使用VAE进行降维...")
    points_latent = reduce_dimension(vae, data_x, scaler, fixed_dim, task_start_indices, task_dims_info)
    print(f"降维后特征维度: {points_latent.shape[1]}")
    
    # 使用降维后的数据点和对应的值进行轨迹构建
    points = points_latent
    values = data_y
    N = points.shape[0]
    
    # 构建输出目录
    if is_multitask:
        tasks_str = '_'.join(tasks_list)
        output_dir = f"./generated_datasets/multi_{tasks_str}_frac{frac}_sigma{sigma}"
    else:
        output_dir = f"./generated_datasets/{tasks_list[0]}_frac{frac}_sigma{sigma}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查轨迹文件是否已存在
    all_files_exist = True
    for task_name in tasks_list:
        task_n_traj = n_traj[task_name]
        task_k = k[task_name]
        task_eps = eps[task_name]
        task_output_path = os.path.join(output_dir, f"{task_name}_{task_n_traj}x{traj_len}_k{task_k}_eps{task_eps}_vae_latent32_train.p")
        if not os.path.exists(task_output_path):
            all_files_exist = False
            break
    
    if all_files_exist:
        print("所有轨迹文件已存在，跳过轨迹构建！")
        return output_dir, None
    
    # 计算距离矩阵（如果不存在）
    distance_file = os.path.join(output_dir, "distance_vae.p")
    if os.path.exists(distance_file):
        print("加载预计算的距离矩阵...")
        distances = pkl.load(open(distance_file, "rb"))
    else:
        print("计算距离矩阵（优先 GPU 分块 torch.cdist，见 GTG_DISTANCE_ON_GPU / GTG_DEVICE）...")
        distances = pairwise_l2_distance_matrix(points)
        pkl.dump(distances, open(distance_file, "wb"))
        print(f"距离矩阵已保存至: {distance_file}")
    
    # 区分单任务和多任务处理
    if is_multitask:
        # 多任务模式：为每个任务生成轨迹并创建混合轨迹
        all_trajectories = {}
        all_our_data = {}
        all_our_data_vals = {}
        all_pr = {}
        all_cumulative_regret_to_go = {}
        all_timesteps = {}
        
        for task_name in tasks_list:
            print(f"为任务 {task_name} 生成轨迹...")
            start_idx, end_idx = task_start_indices[task_name]
            task_points = points[start_idx:end_idx]
            task_values = values[start_idx:end_idx]
            
            # 获取该任务的参数
            task_n_traj = n_traj[task_name]
            task_k = k[task_name]
            task_eps = eps[task_name]
            
            # 选择起始点（前20%的低价值点）
            start_percentile = np.percentile(task_values.numpy(), 20)
            start_candidates_idx = np.where(task_values.numpy() >= start_percentile)[0]
            # 调整索引到全局范围
            global_start_candidates = start_candidates_idx + start_idx
            
            trajectories = []
            for i in tqdm(range(task_n_traj), desc=f"生成任务 {task_name} 的轨迹"):
                idx_list = []
                trajectory = []
                
                # 随机选择起始点
                starting_idx = global_start_candidates[np.random.randint(0, len(global_start_candidates))]
                starting_point = points[starting_idx]
                starting_value = values[[starting_idx]]
                
                idx_list.append(starting_idx)
                trajectory.append(np.concatenate([starting_point, starting_value], axis=0))
                
                # 生成轨迹的其余部分
                for j in range(traj_len-1):
                    # 排除已访问的点
                    distances[starting_idx, np.array(idx_list)] = 1000.0
                    # 选择价值在起始值附近的候选点
                    candidate_idxs = np.arange(N)[values.squeeze() >= starting_value - task_eps]
                    
                    if len(candidate_idxs) <= 1:
                        # 如果候选点太少，选择最近的k个点
                        candidate_idxs = np.argsort(distances[starting_idx])[:task_k]
                        candidate_idx = candidate_idxs[np.argmax(values[candidate_idxs])].item()
                    else:
                        # 从候选点中选择最近的k个点，然后随机选择一个
                        candidate_idxs = candidate_idxs[np.argsort(distances[starting_idx, candidate_idxs])[:task_k]]
                        candidate_idx = np.random.choice(candidate_idxs)
                    
                    # 记录候选点
                    candidate_point = points[candidate_idx]
                    candidate_value = values[[candidate_idx]]
                    trajectory.append(np.concatenate([candidate_point, candidate_value], axis=0))
                    
                    # 更新起始点
                    starting_idx = candidate_idx
                    starting_point = candidate_point
                    starting_value = max(starting_value, candidate_value)
                
                # 完成一个轨迹
                trajectory = np.stack(trajectory, axis=0)
                trajectories.append(trajectory)
            
            # 转换为张量
            trajectories = torch.from_numpy(np.stack(trajectories, axis=0)).float()
            print(f"任务 {task_name} 轨迹形状: {trajectories.shape}")
            
            # 提取数据和值
            our_data = trajectories[..., :-1]
            our_data_vals = trajectories[..., -1]
            
            # 计算pr和累积遗憾
            pr = 1.0 - our_data_vals
            cumulative_regret_to_go = torch.flip(torch.cumsum(torch.flip(pr, [1]), 1), [1])
            
            # 生成时间步
            timesteps = torch.arange(traj_len).repeat(task_n_traj, 1)
            
            # 保存到字典中
            all_trajectories[task_name] = trajectories
            all_our_data[task_name] = our_data
            all_our_data_vals[task_name] = our_data_vals
            all_pr[task_name] = pr
            all_cumulative_regret_to_go[task_name] = cumulative_regret_to_go
            all_timesteps[task_name] = timesteps
            
            # 保存任务特定的轨迹数据
            task_output_path = os.path.join(output_dir, f"{task_name}_{task_n_traj}x{traj_len}_k{task_k}_eps{task_eps}_vae_latent32_train.p")
            task_obj = [our_data, our_data_vals, pr, cumulative_regret_to_go, timesteps]
            pkl.dump(task_obj, open(task_output_path, "wb"))
            print(f"任务 {task_name} 的轨迹数据已保存至: {task_output_path}")
        
        # 保存VAE信息
        vae_info = {
            'latent_dim': 32,
            'vae_path': os.path.join(model_save_dir, "vae_latent32.pt"),
            'fixed_dim': fixed_dim,
            'tasks': tasks_list,
            'task_dims_info': task_dims_info
        }
        vae_info_path = os.path.join(output_dir, "vae_info.p")
        pkl.dump(vae_info, open(vae_info_path, "wb"))
        print(f"VAE信息已保存至: {vae_info_path}")
        
        # 生成混合轨迹文件，包含所有任务的轨迹及其来源
        print("生成混合轨迹文件...")
        all_our_data_list = []
        all_our_data_vals_list = []
        all_pr_list = []
        all_cumulative_regret_to_go_list = []
        all_timesteps_list = []
        all_task_indices = []
        
        # 收集所有任务的数据
        for task_idx, task_name in enumerate(tasks_list):
            all_our_data_list.append(all_our_data[task_name])
            all_our_data_vals_list.append(all_our_data_vals[task_name])
            all_pr_list.append(all_pr[task_name])
            all_cumulative_regret_to_go_list.append(all_cumulative_regret_to_go[task_name])
            all_timesteps_list.append(all_timesteps[task_name])
            
            # 创建任务索引标记
            task_indices = torch.full_like(all_timesteps[task_name], task_idx, dtype=torch.long)
            all_task_indices.append(task_indices)
        
        # 合并所有数据
        mixed_our_data = torch.cat(all_our_data_list, dim=0)
        mixed_our_data_vals = torch.cat(all_our_data_vals_list, dim=0)
        mixed_pr = torch.cat(all_pr_list, dim=0)
        mixed_cumulative_regret_to_go = torch.cat(all_cumulative_regret_to_go_list, dim=0)
        mixed_timesteps = torch.cat(all_timesteps_list, dim=0)
        mixed_task_indices = torch.cat(all_task_indices, dim=0)
        
        # 保存混合轨迹文件
        mixed_trajectory_obj = [
            mixed_our_data,
            mixed_our_data_vals,
            mixed_pr,
            mixed_cumulative_regret_to_go,
            mixed_timesteps,
            mixed_task_indices,
            tasks_list  # 保存任务列表，用于映射任务索引
        ]
        
        mixed_output_path = os.path.join(output_dir, "mixed_trajectories_train.p")
        pkl.dump(mixed_trajectory_obj, open(mixed_output_path, "wb"))
        print(f"混合轨迹文件已保存至: {mixed_output_path}")
        print(f"混合轨迹数量: {mixed_our_data.shape[0]}")
        
        print("多任务轨迹构建完成！")
        return output_dir, all_trajectories
    else:
        # 单任务模式：参考dfgo-main的实现，直接生成轨迹
        task_name = tasks_list[0]
        print(f"为任务 {task_name} 生成轨迹...")
        
        # 获取参数
        task_n_traj = n_traj[task_name]
        task_k = k[task_name]
        task_eps = eps[task_name]
        
        # 选择起始点（前20%的低价值点）
        start_percentile = np.percentile(values.numpy(), 20)
        start_candidates_idx = np.arange(N)[values.numpy() >= start_percentile]
        
        trajectories = []
        for i in tqdm(range(task_n_traj), desc=f"生成任务 {task_name} 的轨迹"):
            idx_list = []
            trajectory = []
            
            # 随机选择起始点
            starting_idx = start_candidates_idx[np.random.randint(0, len(start_candidates_idx))]
            starting_point = points[starting_idx]
            starting_value = values[[starting_idx]]
            
            idx_list.append(starting_idx)
            trajectory.append(np.concatenate([starting_point, starting_value], axis=0))
            
            # 生成轨迹的其余部分
            for j in range(traj_len-1):
                # 排除已访问的点
                distances[starting_idx, np.array(idx_list)] = 1000.0
                # 选择价值在起始值附近的候选点
                candidate_idxs = np.arange(N)[values >= starting_value - task_eps]
                
                if len(candidate_idxs) <= 1:
                    # 如果候选点太少，选择最近的k个点
                    candidate_idxs = np.argsort(distances[starting_idx])[:task_k]
                    candidate_idx = candidate_idxs[np.argmax(values[candidate_idxs])].item()
                else:
                    # 从候选点中选择最近的k个点，然后随机选择一个
                    candidate_idxs = candidate_idxs[np.argsort(distances[starting_idx, candidate_idxs])[:task_k]]
                    candidate_idx = np.random.choice(candidate_idxs)
                
                # 记录候选点
                candidate_point = points[candidate_idx]
                candidate_value = values[[candidate_idx]]
                trajectory.append(np.concatenate([candidate_point, candidate_value], axis=0))
                
                # 更新起始点
                starting_idx = candidate_idx
                starting_point = candidate_point
                starting_value = max(starting_value, candidate_value)
                idx_list.append(starting_idx)
            
            # 完成一个轨迹
            trajectory = np.stack(trajectory, axis=0)
            trajectories.append(trajectory)
        
        # 转换为张量
        trajectories = torch.from_numpy(np.stack(trajectories, axis=0)).float()
        print(f"任务 {task_name} 轨迹形状: {trajectories.shape}")
        
        # 提取数据和值
        our_data = trajectories[..., :-1]
        our_data_vals = trajectories[..., -1]
        
        # 计算pr和累积遗憾
        pr = 1.0 - our_data_vals
        cumulative_regret_to_go = torch.flip(torch.cumsum(torch.flip(pr, [1]), 1), [1])
        
        # 生成时间步
        timesteps = torch.arange(traj_len).repeat(task_n_traj, 1)
        
        # 保存轨迹数据（参考dfgo-main的命名方式）
        task_output_path = os.path.join(output_dir, f"{task_name}_{task_n_traj}x{traj_len}_k{task_k}_eps{task_eps}_vae_latent32_train.p")
        task_obj = [our_data, our_data_vals, pr, cumulative_regret_to_go, timesteps]
        pkl.dump(task_obj, open(task_output_path, "wb"))
        print(f"任务 {task_name} 的轨迹数据已保存至: {task_output_path}")
        
        # 保存VAE信息
        vae_info = {
            'latent_dim': 32,
            'vae_path': os.path.join(model_save_dir, "vae_latent32.pt"),
            'fixed_dim': fixed_dim,
            'task': task_name,
            'original_dim': task_dims_info[task_name]['original_dim']
        }
        vae_info_path = os.path.join(output_dir, "vae_info.p")
        pkl.dump(vae_info, open(vae_info_path, "wb"))
        print(f"VAE信息已保存至: {vae_info_path}")
        
        print("单任务轨迹构建完成！")
        return output_dir, trajectories

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 添加多任务支持
    parser.add_argument('--tasks', type=str, default='dkitty', help="训练数据集列表，用逗号分隔，例如: dkitty,ant,tfbind8")
    parser.add_argument('--task', type=str, default='', help="兼容旧版API，将被废弃")
    parser.add_argument('--frac', type=float, default=1.0)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_traj', type=int, default=None, help="每任务生成的轨迹数量")
    parser.add_argument('--k', type=int, default=None, help="每步选择的候选点数量")
    parser.add_argument('--eps', type=float, default=None, help="允许的目标值下降范围")
    parser.add_argument('--fixed_dim', type=int, default=128, help="固定的输入维度，用于统一不同数据集")
    parser.add_argument('--horizon', type=int, default=64, help="合成轨迹长度，需与训练时 horizon 一致")

    args = parser.parse_args()
    
    # 兼容旧版API - 当指定task时，优先使用task参数覆盖tasks
    if args.task:
        args.tasks = args.task
    
    # 解析任务列表（去空白；多任务时字典序排序，与 train/evaluate 的 multi_* 路径一致）
    tasks_list = [t.strip() for t in args.tasks.split(',') if t.strip()]

    # 验证任务列表中的任务是否都受支持
    for task in tasks_list:
        if task not in SUPPORTED_TASKS:
            print(f"警告: 任务 '{task}' 不被支持，将被忽略")

    # 过滤出有效的任务
    tasks_list = [task for task in tasks_list if task in SUPPORTED_TASKS]
    if len(tasks_list) > 1:
        tasks_list = sorted(tasks_list)
    
    if not tasks_list:
        print("错误: 没有有效的任务列表")
        exit(1)
    
    # 调用统一的轨迹构建函数
    construct_trajectories(
        tasks_list=tasks_list,
        frac=args.frac,
        sigma=args.sigma,
        seed=args.seed,
        n_traj=args.n_traj,
        k=args.k,
        eps=args.eps,
        fixed_dim=args.fixed_dim,
        horizon=args.horizon,
    )
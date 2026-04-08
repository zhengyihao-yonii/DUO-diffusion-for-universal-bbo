# -*- coding: utf-8 -*-
import torch
import numpy as np
import pickle as pkl
import os
from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset
import design_bench
from diffuser.utils.soo_gtopx import (
    TASKNAME_TO_VAR_NUM,
    load_gtopx_offline_arrays,
    is_gtopx_task,
)
from diffuser.datasets.sequence import TASKNAME2MAX_SAMPLES, TASKNAME2TASK

# 任务映射字典
TASKNAME2FULL = {
        'dkitty': DKittyMorphologyDataset,
        'ant': AntMorphologyDataset,
        'tfbind8': TFBind8Dataset,
        'tfbind10': TFBind10Dataset,
        'superconductor': SuperconductorDataset,
}

TASKNAME2DIM = {
    'dkitty': 56,
    'ant': 60,
    # 以下为「原始」维；tfbind 在 train_vae / ZipDataset 中使用 map_to_logits 后维数会变，wrapper 会覆盖 original_dim
    'tfbind8': 8,
    'tfbind10': 10,
    'superconductor': 86,
    'gtopx2': TASKNAME_TO_VAR_NUM['gtopx2'],
    'gtopx3': TASKNAME_TO_VAR_NUM['gtopx3'],
    'gtopx4': TASKNAME_TO_VAR_NUM['gtopx4'],
    'gtopx6': TASKNAME_TO_VAR_NUM['gtopx6'],
}

class DesignBenchDatasetWrapper:
    """
    设计基准数据集包装器，支持数据填充到固定长度
    """
    def __init__(self, dataset_name, fixed_length=None, mode="train", frac=1.0, sigma=0.0):
        self.dataset_name = dataset_name
        self.fixed_length = fixed_length
        self.mode = mode
        self.frac = frac
        self.sigma = sigma
        
        if is_gtopx_task(dataset_name):
            self.dataset = None
            self.original_dim = TASKNAME2DIM[dataset_name]
            if self.fixed_length is None:
                self.fixed_length = self.original_dim
            x, y_norm, _, _, _ = load_gtopx_offline_arrays(
                dataset_name, frac=frac, sigma=sigma, seed=42
            )
            self.x = x
            self.y = y_norm.reshape(-1, 1) if y_norm.ndim == 1 else y_norm
            self.processed_x = self._preprocess_to_fixed_length(self.x)
            self.y_normalized = self.y.squeeze(-1) if self.y.ndim > 1 else self.y
            self.x_tensor = torch.tensor(self.processed_x, dtype=torch.float32)
            self.y_tensor = torch.tensor(self.y_normalized, dtype=torch.float32)
            # 与 Design-Bench 路径一致：统一为 [N]，避免多任务合并时 1D/2D 混用导致 torch.cat 失败
            self.y_tensor = self.y_tensor.reshape(-1)
            return

        # TFBind：必须与 train_vae.py / ZipDataset 一致，先 map_to_logits 再标准化，否则 VAE scaler 维数与数据不一致
        if dataset_name.startswith("tfbind"):
            task = design_bench.make(
                TASKNAME2TASK[dataset_name],
                dataset_kwargs=dict(
                    max_samples=int(TASKNAME2MAX_SAMPLES[dataset_name] * frac),
                    distribution=None,
                    min_percentile=0,
                ),
            )
            task.map_to_logits()
            self.dataset = TASKNAME2FULL[dataset_name]()
            flat = task.x.reshape(task.x.shape[0], -1).astype(np.float32)
            self.x = flat
            self.y = task.y
            self.original_dim = flat.shape[1]
        else:
            # 加载原始数据集（Design-Bench）
            self.dataset = TASKNAME2FULL[dataset_name]()
            self.original_dim = TASKNAME2DIM[dataset_name]

            # 获取完整数据
            x_full = self.dataset.x
            y_full = self.dataset.y

            # 应用采样比例
            if frac < 1.0:
                num_samples = int(len(x_full) * frac)
                indices = np.random.choice(len(x_full), num_samples, replace=False)
                self.x = x_full[indices]
                self.y = y_full[indices]
            else:
                self.x = x_full
                self.y = y_full

        # 如果没有指定固定长度，使用原始维度
        if self.fixed_length is None:
            self.fixed_length = self.original_dim

        # 预处理数据：填充到固定长度
        self.processed_x = self._preprocess_to_fixed_length(self.x)
        
        # 归一化和添加噪声到y值
        y_min, y_max = self.dataset.y.min(), self.dataset.y.max()
        self.y_normalized = (self.y - y_min) / (y_max - y_min)
        if sigma > 0.0:
            self.y_normalized = np.clip(self.y_normalized + np.random.randn(*self.y_normalized.shape) * sigma, 0.0, 1.0)
        
        # 转换为张量（Design-Bench 的 y 常为 (N,1)，归一化后仍为 2D；压成 [N] 与 gtopx / 下游 cat 一致）
        self.x_tensor = torch.tensor(self.processed_x, dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y_normalized, dtype=torch.float32).reshape(-1)
    
    def _preprocess_to_fixed_length(self, data):
        """
        将数据统一填充/截断到固定长度
        """
        processed_data = []
        for x in data:
            x_flat = x.flatten()
            if len(x_flat) < self.fixed_length:
                # 填充零
                padded = np.pad(x_flat, (0, self.fixed_length - len(x_flat)), mode='constant')
            else:
                # 截断
                padded = x_flat[:self.fixed_length]
            processed_data.append(padded)
        return np.array(processed_data)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]

class MultiDatasetLoader:
    """
    多数据集加载器，支持加载和处理多个数据集
    """
    def __init__(self, task_list, fixed_length=None):
        self.task_list = task_list
        self.fixed_length = fixed_length
        self.datasets = []
        self.dataset_info = {}
        
        # 设置随机种子以确保可重复性
        np.random.seed(42)
        
        # 加载每个数据集
        for task in task_list:
            # 暂时不应用frac和sigma，在get_combined_data中应用
            dataset = DesignBenchDatasetWrapper(task, fixed_length=fixed_length)
            self.datasets.append(dataset)
            self.dataset_info[task] = {
                'original_dim': dataset.original_dim,
                'fixed_length': fixed_length,
                'size': len(dataset)
            }
    
    def get_combined_data(self, frac=1.0, sigma=0.0):
        """
        获取所有数据集的合并数据
        
        Args:
            frac: 采样比例
            sigma: 添加到y值的噪声标准差
            
        Returns:
            tuple: (combined_x, combined_y, task_start_indices, dataset_info)
        """
        all_x = []
        all_y = []
        task_start_indices = {}
        current_start_idx = 0
        
        # 重新加载数据集以应用frac和sigma
        self.datasets = []
        for task in self.task_list:
            dataset = DesignBenchDatasetWrapper(task, fixed_length=self.fixed_length, frac=frac, sigma=sigma)
            self.datasets.append(dataset)
            
            # 更新数据集信息
            self.dataset_info[task]['size'] = len(dataset)
            
            # 记录任务起始索引
            task_start_indices[task] = (current_start_idx, current_start_idx + len(dataset))
            current_start_idx += len(dataset)
            
            all_x.append(dataset.x_tensor)
            all_y.append(dataset.y_tensor)
        
        # 合并数据
        combined_x = torch.cat(all_x, dim=0) if all_x else torch.tensor([])
        combined_y = torch.cat(all_y, dim=0) if all_y else torch.tensor([])
        
        return combined_x, combined_y, task_start_indices, self.dataset_info
    
    def get_dataset_info(self):
        """
        获取数据集信息
        """
        return self.dataset_info

class MixedDataset(torch.utils.data.Dataset):
    """
    混合数据集类，用于真正随机混合多个数据集
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.lengths)
        
        # 创建每个数据集中样本的索引映射
        self.dataset_indices = []
        self.sample_indices = []
        
        for i, length in enumerate(self.lengths):
            self.dataset_indices.extend([i] * length)
            self.sample_indices.extend(range(length))
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # 获取原始数据集索引和样本索引
        dataset_idx = self.dataset_indices[idx]
        sample_idx = self.sample_indices[idx]
        
        # 返回对应数据集的样本
        return self.datasets[dataset_idx][sample_idx]

# 保存数据集信息函数
def save_dataset_info(dataset_info, save_path):
    """
    保存数据集信息到文件
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pkl.dump(dataset_info, f)
    print(f"数据集信息已保存到: {save_path}")

# 加载数据集信息函数
def load_dataset_info(load_path):
    """
    从文件加载数据集信息
    """
    with open(load_path, 'rb') as f:
        dataset_info = pkl.load(f)
    return dataset_info
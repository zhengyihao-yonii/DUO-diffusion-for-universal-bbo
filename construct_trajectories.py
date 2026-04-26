import os
import sys

if __name__ == "__main__":
    from diffuser.cpu_threads import maybe_apply_from_argv_and_env

    maybe_apply_from_argv_and_env()

import hashlib
import random
import argparse
from typing import Optional, Tuple
from tqdm import tqdm
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pickle as pkl

import diffuser.numpy_design_bench_compat  # noqa: F401
import numpy as np

# 禁止 `from diffuser.utils import set_seed`：会执行 utils/__init__ → training（曾顶层 import wandb），
# 旧 typing_extensions 下 wandb 导入失败，construct 在 Step 1 即 exit 1，run_multitask 因 set -e 直接结束。
def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from sklearn.preprocessing import StandardScaler
from dataset_utils import TASKNAME2DIM, MultiDatasetLoader, save_dataset_info, load_dataset_info

from diffuser.utils.vae_layout import (
    distance_matrix_cache_filename,
    generated_vae_info_filename,
    multitask_generated_dim_latent_suffix,
    per_task_latent_train_filename,
    raw_train_pkl_path_from_latent_path,
    vae_state_pt_filename,
)

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

from diffuser.datasets.sequence import TASKNAME2FULL, TASKNAME2TASK, TASKNAME2MAX_SAMPLES, SUPPORTED_TASKS
from diffuser.datasets.real_world_fewshot import (
    is_real_world_fewshot_task,
    load_real_world_for_pipeline,
)
from diffuser.utils.soo_gtopx import TASKNAME_TO_VAR_NUM, load_gtopx_offline_arrays, is_gtopx_task
from diffuser.utils.construct_runtime import pairwise_l2_distance_matrix
from diffuser.utils.multitask_canon import canonical_train_tasks_csv, multitask_path_token


def _generated_task_dir(task_name: str, frac: float, sigma: float) -> str:
    """每任务独立目录：距离矩阵与单任务轨迹 pkl（与单任务命名一致）。"""
    return f"./generated_datasets/{task_name}_frac{frac}_sigma{sigma}"


def _latent_fingerprint(pts: torch.Tensor) -> str:
    """用于距离矩阵的那块 latent 的字节指纹（联合 VAE 与单任务 VAE 潜空间一般不同）。"""
    x = pts.detach().cpu().contiguous().float().numpy()
    return hashlib.sha256(x.tobytes()).hexdigest()


def _load_distance_matrix_cache(
    path: str,
    pts: torch.Tensor,
    *,
    allow_legacy_plain: bool,
) -> Tuple[Optional[torch.Tensor], bool]:
    """
    加载 ``distance_vae.p``。若缓存含 ``latent_fp`` 且与当前 ``pts`` 一致则复用。
    多任务设 ``allow_legacy_plain=False``：无指纹的旧缓存（多为单任务 VAE 所算）一律不采用，避免误复用。
    """
    if not os.path.isfile(path):
        return None, False
    with open(path, "rb") as f:
        obj = pkl.load(f)
    n = int(pts.shape[0])
    if isinstance(obj, dict) and "matrix" in obj:
        mat = obj["matrix"]
        if getattr(mat, "shape", None) is None or tuple(mat.shape) != (n, n):
            return None, False
        if obj.get("latent_fp") != _latent_fingerprint(pts):
            return None, False
        return mat, True
    if isinstance(obj, torch.Tensor):
        if tuple(obj.shape) != (n, n):
            return None, False
        if not allow_legacy_plain:
            return None, False
        return obj, True
    return None, False


def _save_distance_matrix_cache(path: str, matrix: torch.Tensor, pts: torch.Tensor) -> None:
    pkl.dump(
        {"matrix": matrix, "latent_fp": _latent_fingerprint(pts)},
        open(path, "wb"),
    )


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

def construct_trajectories(
    tasks_list,
    frac=1.0,
    sigma=0.0,
    seed=0,
    n_traj=None,
    k=None,
    eps=None,
    fixed_dim=128,
    horizon=64,
    traj_params_json=None,
    fewshot_k=None,
    fewshot_mode="all",
    pretrained_vae_info=None,
    finetune_epochs=50,
    finetune_lr=3e-5,
    latent_dim=32,
):
    """
    构建轨迹，支持单任务和多任务模式
    
    参数:
    - tasks_list: 任务列表
    - frac: 每个任务使用的数据比例
    - sigma: 噪声水平
    - seed: 随机种子
    - n_traj: 每任务生成的轨迹数量（可选 int，或 per-task dict）
    - k: 每步选择的候选点数量（可选 int 或 dict）
    - eps: 允许的目标值下降范围（可选 float 或 dict）
    - fixed_dim: 固定的输入维度，用于统一不同数据集
    - horizon: 每条轨迹长度（时间步数），需与后续扩散训练 Config.horizon 一致
    - traj_params_json: 可选 JSON 路径，在标量/默认字典上按任务覆盖 n_traj/k/eps（见 diffuser.utils.traj_params）
    - fewshot_k: real-world 任务 few-shot 点数（None=全量）；配合 fewshot_mode
    - fewshot_mode: ``all`` | ``random`` | ``worst``（y 越小越差）
    - pretrained_vae_info: **单任务 real-world 必填**，指向仅含 Design-Bench 等多任务预训练的 ``vae_info.p``，
      在 few-shot 数据上 **微调** VAE（小学习率），兼顾预训练表征与真实域分布。
    - finetune_epochs / finetune_lr: 微调轮数与学习率（仅 real-world）
    - latent_dim: VAE 隐空间维度；决定 ``_vae_latent{d}_train.p`` / ``vae_latent{d}.pt`` / mixed 文件名后缀。
    """
    set_seed(seed)
    traj_len = horizon
    _latent = int(latent_dim)

    from diffuser.utils.traj_params import (
        coerce_traj_param_dicts,
        merge_traj_params_json,
        multitask_mixed_basename,
        multitask_trajectory_signature,
    )

    n_traj, k, eps = coerce_traj_param_dicts(tasks_list, n_traj, k, eps)
    if traj_params_json:
        n_traj, k, eps = merge_traj_params_json(
            traj_params_json, tasks_list, n_traj, k, eps
        )

    # 根据任务数量选择不同的数据加载方式
    is_multitask = len(tasks_list) > 1
    rw_tasks = [t for t in tasks_list if is_real_world_fewshot_task(t)]
    if rw_tasks and len(tasks_list) != 1:
        raise ValueError(
            "真实任务 / few-shot 实验仅支持单任务；当前列表: %s" % tasks_list
        )
    if rw_tasks and not pretrained_vae_info:
        raise ValueError(
            "real-world 单任务需指定 --pretrained_vae_info（多任务预训练得到的 vae_info.p），"
            "以便在 few-shot 上微调 VAE。"
        )
    if (
        is_multitask
        and (fewshot_k is not None or (fewshot_mode not in (None, "all")))
    ):
        raise ValueError("few-shot 参数（fewshot_k / fewshot_mode）仅用于单任务 real-world。")
    mt_sig = (
        multitask_trajectory_signature(tasks_list, n_traj, k, eps, traj_len)
        if is_multitask
        else None
    )

    # 与 train/evaluate / train_vae 一致：latent≠32 时用 ``_dim{fixed}_latent{lat}`` 与 VAE 权重目录对齐
    if is_multitask:
        _multi_tok = multitask_path_token(canonical_train_tasks_csv(",".join(tasks_list)))
        _gds_suf = multitask_generated_dim_latent_suffix(fixed_dim, _latent)
        output_dir_early = f"./generated_datasets/multi_{_multi_tok}_frac{frac}_sigma{sigma}{_gds_suf}"
    else:
        output_dir_early = f"./generated_datasets/{tasks_list[0]}_frac{frac}_sigma{sigma}"

    # 产物已存在则跳过数据加载、VAE、降维（避免仅加 USE_RETURNS 仍整段重跑 Step 1）
    # 多任务必须与单任务一致：按 n_traj/k/eps/horizon 检查各任务 pkl；不能仅因 mixed 存在就跳过，
    # 否则更换 n_traj 后仍会命中旧的 mixed，且 run_multitask 后续 train 会指向不存在的 data_path。
    if is_multitask:
        mixed_short = os.path.join(
            output_dir_early, multitask_mixed_basename(mt_sig, _latent)
        )
        mixed_long = os.path.join(output_dir_early, f"mixed_{mt_sig}.p")
        mixed_legacy = os.path.join(output_dir_early, "mixed_trajectories_train.p")
        all_task_pkls_exist = True
        for task_name in tasks_list:
            task_n_traj = n_traj[task_name]
            task_k = k[task_name]
            task_eps = eps[task_name]
            task_pkl = os.path.join(
                _generated_task_dir(task_name, frac, sigma),
                per_task_latent_train_filename(
                    task_name, task_n_traj, traj_len, task_k, task_eps, _latent
                ),
            )
            if not os.path.isfile(task_pkl):
                all_task_pkls_exist = False
                break
        mixed_ok = (
            os.path.isfile(mixed_short)
            or os.path.isfile(mixed_long)
            or os.path.isfile(mixed_legacy)
        )
        if mixed_ok and all_task_pkls_exist:
            if os.path.isfile(mixed_short):
                _mp = mixed_short
            elif os.path.isfile(mixed_long):
                _mp = mixed_long
            else:
                _mp = mixed_legacy
            print(
                f"已存在多任务混合轨迹（训练实际加载此文件），跳过数据加载与 VAE：{_mp}"
            )
            return output_dir_early, None
    else:
        tn = tasks_list[0]
        task_pkl = os.path.join(
            output_dir_early,
            per_task_latent_train_filename(
                tn, n_traj[tn], traj_len, k[tn], eps[tn], _latent
            ),
        )
        if os.path.isfile(task_pkl):
            print(f"已存在轨迹文件，跳过数据加载与 VAE：{task_pkl}")
            return output_dir_early, None

    # 延后：避免「仅跳过 Step 1」时 import VAE/train_vae 拉入不必要依赖
    from train_vae import main as train_vae_main

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
                self.latent_dim = _latent  # 与轨迹 / vae_latent{d}.pt 命名一致
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
        elif is_real_world_fewshot_task(task_name):
            proc, y_norm, _odim = load_real_world_for_pipeline(
                task_name,
                fixed_length=fixed_dim,
                frac=frac,
                sigma=sigma,
                fewshot_k=fewshot_k,
                fewshot_mode=fewshot_mode,
                fewshot_seed=seed,
            )
            data_x = torch.tensor(proc, dtype=torch.float32)
            data_y = torch.tensor(y_norm, dtype=torch.float32)
            print(
                f"Real-world 任务 {task_name}: x={tuple(data_x.shape)}, few-shot k={fewshot_k} mode={fewshot_mode}, "
                f"y 归一化后 min/max",
                float(data_y.min()),
                float(data_y.max()),
            )
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
                self.latent_dim = _latent
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
                self.pretrained_vae_info = pretrained_vae_info
                self.finetune_epochs = finetune_epochs
                self.finetune_lr = finetune_lr
                self.fewshot_k = fewshot_k
                self.fewshot_mode = fewshot_mode
                self.fewshot_seed = seed

        vae_args = VAEArgs()

        # 单任务的任务维度信息（real-world 任务 data_x 已填充到 fixed_dim，不能用 shape[1]）
        if is_gtopx_task(task_name):
            odim = TASKNAME_TO_VAR_NUM[task_name]
        elif is_real_world_fewshot_task(task_name):
            odim = TASKNAME2DIM[task_name]
        else:
            odim = data_x.shape[1]
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

    # 训练或加载VAE模型（real-world：在多任务预训练权重上 few-shot 微调）
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
    points_latent = reduce_dimension(
        vae, data_x, scaler, fixed_dim, task_start_indices, task_dims_info
    )

    print(f"降维后特征维度: {points_latent.shape[1]}")
    
    # 使用降维后的数据点和对应的值进行轨迹构建
    points = points_latent
    values = data_y
    N = points.shape[0]
    
    # 多任务：multi_* 仅放 mixed_*.p 与 vae_info.p；各任务距离矩阵与轨迹在 {task}_frac_sigma/ 下。
    if is_multitask:
        _multi_tok = multitask_path_token(canonical_train_tasks_csv(",".join(tasks_list)))
        _gds_suf = multitask_generated_dim_latent_suffix(fixed_dim, _latent)
        multi_dir = f"./generated_datasets/multi_{_multi_tok}_frac{frac}_sigma{sigma}{_gds_suf}"
        output_dir = multi_dir
        os.makedirs(multi_dir, exist_ok=True)
    else:
        output_dir = f"./generated_datasets/{tasks_list[0]}_frac{frac}_sigma{sigma}"
        os.makedirs(output_dir, exist_ok=True)
    
    # 检查轨迹文件是否已存在（多任务需各任务 pkl + multi 下 mixed）
    all_files_exist = True
    if is_multitask:
        mixed_here = os.path.join(output_dir, multitask_mixed_basename(mt_sig, _latent))
        mixed_long = os.path.join(output_dir, f"mixed_{mt_sig}.p")
        mixed_legacy = os.path.join(output_dir, "mixed_trajectories_train.p")
        if not (
            os.path.isfile(mixed_here)
            or os.path.isfile(mixed_long)
            or os.path.isfile(mixed_legacy)
        ):
            all_files_exist = False
        if all_files_exist:
            for task_name in tasks_list:
                task_n_traj = n_traj[task_name]
                task_k = k[task_name]
                task_eps = eps[task_name]
                task_output_path = os.path.join(
                    _generated_task_dir(task_name, frac, sigma),
                    per_task_latent_train_filename(
                        task_name, task_n_traj, traj_len, task_k, task_eps, _latent
                    ),
                )
                if not os.path.exists(task_output_path):
                    all_files_exist = False
                    break
    else:
        for task_name in tasks_list:
            task_n_traj = n_traj[task_name]
            task_k = k[task_name]
            task_eps = eps[task_name]
            task_output_path = os.path.join(
                output_dir,
                per_task_latent_train_filename(
                    task_name, task_n_traj, traj_len, task_k, task_eps, _latent
                ),
            )
            if not os.path.exists(task_output_path):
                all_files_exist = False
                break
    
    if all_files_exist:
        print("所有轨迹文件已存在，跳过轨迹构建！")
        return output_dir, None
    
    # 与 GTG-main 一致：预计算 pairwise L2 距离矩阵，轨迹步内原地对已访问列置 1000 再 argsort 选点。
    # 多任务时每个任务只在 ``points[start:end]`` 上建 task_n×task_n 矩阵（局部下标），不跨任务选点。
    
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
            print(f"为任务 {task_name} 生成轨迹（仅在同一任务数据块 [start,end) 内选点）...")
            task_out_dir = _generated_task_dir(task_name, frac, sigma)
            os.makedirs(task_out_dir, exist_ok=True)
            start_idx, end_idx = task_start_indices[task_name]
            task_values = values[start_idx:end_idx]
            task_points = points[start_idx:end_idx]
            task_n = end_idx - start_idx
            if task_n <= 0:
                raise ValueError(f"任务 {task_name} 数据块为空: start={start_idx}, end={end_idx}")
            
            dist_task_path = os.path.join(
                task_out_dir, distance_matrix_cache_filename(_latent)
            )
            distances_t, dist_cache_ok = _load_distance_matrix_cache(
                dist_task_path, task_points, allow_legacy_plain=False
            )
            if dist_cache_ok:
                print(
                    f"复用任务 {task_name} 距离矩阵（与当前多任务 VAE latent 指纹一致）: {dist_task_path}"
                )
            else:
                if os.path.isfile(dist_task_path):
                    print(
                        f"任务 {task_name}: 已有缓存但与当前联合 VAE latent 不一致，或为无指纹旧文件；"
                        "重新计算距离矩阵（单任务缓存不可直接用于多任务潜空间）。"
                    )
                print(
                    f"计算任务 {task_name} 距离矩阵（L2，GPU 分块见 GTG_DISTANCE_ON_GPU / GTG_DEVICE）..."
                )
                distances_t = pairwise_l2_distance_matrix(task_points)
                _save_distance_matrix_cache(dist_task_path, distances_t, task_points)
                print(f"已保存（含 latent 指纹）: {dist_task_path}")
            
            # 获取该任务的参数
            task_n_traj = n_traj[task_name]
            task_k = k[task_name]
            task_eps = eps[task_name]
            
            # 选择起始点（前20%的低价值点）
            start_percentile = np.percentile(task_values.numpy(), 20)
            start_candidates_idx = np.where(task_values.numpy() >= start_percentile)[0]
            global_start_candidates = start_candidates_idx + start_idx
            
            task_vals_np = task_values.squeeze().detach().cpu().numpy()
            trajectories = []
            for i in tqdm(range(task_n_traj), desc=f"生成任务 {task_name} 的轨迹"):
                trajectory = []
                starting_idx = int(
                    global_start_candidates[np.random.randint(0, len(global_start_candidates))]
                )
                idx_list_local = [starting_idx - start_idx]
                starting_point = points[starting_idx]
                starting_value = values[[starting_idx]]
                trajectory.append(np.concatenate([starting_point, starting_value], axis=0))
                
                for j in range(traj_len-1):
                    sl = starting_idx - start_idx
                    distances_t[sl, np.array(idx_list_local, dtype=np.int64)] = 1000.0
                    thresh_np = float((starting_value - task_eps).squeeze().cpu())
                    candidate_local = np.where(task_vals_np >= thresh_np)[0]
                    row = distances_t[sl].detach().cpu().numpy()
                    
                    if len(candidate_local) <= 1:
                        cand_local = np.argsort(row)[:task_k]
                        cl = int(cand_local[np.argmax(task_vals_np[cand_local])])
                    else:
                        d_sub = row[candidate_local]
                        cand_local = candidate_local[np.argsort(d_sub)[:task_k]]
                        cl = int(np.random.choice(cand_local))
                    
                    candidate_idx = cl + start_idx
                    candidate_point = points[candidate_idx]
                    candidate_value = values[[candidate_idx]]
                    trajectory.append(np.concatenate([candidate_point, candidate_value], axis=0))
                    starting_idx = candidate_idx
                    starting_value = max(starting_value, candidate_value)
                    idx_list_local.append(cl)
                
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
            
            # 保存任务特定的轨迹数据（任务名目录下）
            task_output_path = os.path.join(
                task_out_dir,
                per_task_latent_train_filename(
                    task_name, task_n_traj, traj_len, task_k, task_eps, _latent
                ),
            )
            task_obj = [our_data, our_data_vals, pr, cumulative_regret_to_go, timesteps]
            pkl.dump(task_obj, open(task_output_path, "wb"))
            print(f"任务 {task_name} 的轨迹数据已保存至: {task_output_path}")
        
        # 保存VAE信息（仅 multi 目录，供 evaluate 等解析）
        vae_info = {
            "latent_dim": _latent,
            "vae_path": os.path.join(model_save_dir, vae_state_pt_filename(_latent)),
            "fixed_dim": fixed_dim,
            "tasks": tasks_list,
            "task_dims_info": task_dims_info,
        }
        vae_info_path = os.path.join(multi_dir, generated_vae_info_filename(_latent))
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
        
        mixed_output_path = os.path.join(
            multi_dir, multitask_mixed_basename(mt_sig, _latent)
        )
        _mw = int(mixed_our_data.shape[-1])
        if _mw != int(_latent):
            raise RuntimeError(
                f"混合轨迹 observation 宽度 {_mw} 与目标 latent_dim={_latent} 不一致；"
                f"请检查联合 VAE 的 train_vae_main / reduce_dimension。"
            )
        pkl.dump(mixed_trajectory_obj, open(mixed_output_path, "wb"))
        print(f"混合轨迹文件已保存至: {mixed_output_path}")
        print(f"混合轨迹数量: {mixed_our_data.shape[0]}")
        from diffuser.utils.multitask_slug_registry import write_multitask_manifest

        _manifest_written = write_multitask_manifest(
            multi_dir,
            traj_signature=mt_sig,
            tasks_list=tasks_list,
            n_traj=n_traj,
            k=k,
            eps=eps,
            frac=frac,
            sigma=sigma,
            horizon=traj_len,
            traj_params_json=traj_params_json,
            latent_dim=_latent,
        )
        print(f"多任务 slug 清单已写入: {_manifest_written}")
        
        print("多任务轨迹构建完成！")
        return multi_dir, all_trajectories
    else:
        # 单任务：与 GTG-main 一致，全库 N×N 距离矩阵 + 原地惩罚列
        task_name = tasks_list[0]
        print(f"为任务 {task_name} 生成轨迹...")
        
        distance_file = os.path.join(
            output_dir, distance_matrix_cache_filename(_latent)
        )
        distances, dist_cache_ok = _load_distance_matrix_cache(
            distance_file, points, allow_legacy_plain=True
        )
        if dist_cache_ok:
            print("加载预计算的距离矩阵（形状与 latent 指纹一致，或兼容无指纹旧缓存）...")
        else:
            if os.path.isfile(distance_file):
                print("已有 distance_vae.p 但与当前 latent 不一致，重新计算...")
            print("计算距离矩阵（L2 优先 GPU 分块，见 GTG_DISTANCE_ON_GPU / GTG_DEVICE）...")
            distances = pairwise_l2_distance_matrix(points)
            _save_distance_matrix_cache(distance_file, distances, points)
            print(f"距离矩阵已保存至（含 latent 指纹）: {distance_file}")
        
        task_n_traj = n_traj[task_name]
        task_k = k[task_name]
        task_eps = eps[task_name]
        
        vals_np = values.squeeze().detach().cpu().numpy()
        start_percentile = np.percentile(vals_np, 20)
        start_candidates_idx = np.arange(N)[vals_np >= start_percentile]
        
        trajectories = []
        for i in tqdm(range(task_n_traj), desc=f"生成任务 {task_name} 的轨迹"):
            trajectory = []
            starting_idx = int(start_candidates_idx[np.random.randint(0, len(start_candidates_idx))])
            starting_point = points[starting_idx]
            starting_value = values[[starting_idx]]
            idx_list = [starting_idx]
            trajectory.append(np.concatenate([starting_point, starting_value], axis=0))
            
            for j in range(traj_len-1):
                distances[starting_idx, np.array(idx_list, dtype=np.int64)] = 1000.0
                thresh_np = float((starting_value - task_eps).squeeze().cpu())
                candidate_idxs = np.arange(N)[vals_np >= thresh_np]
                row = distances[starting_idx].detach().cpu().numpy()
                
                if len(candidate_idxs) <= 1:
                    candidate_idxs = np.argsort(row)[:task_k]
                    candidate_idx = int(candidate_idxs[np.argmax(vals_np[candidate_idxs])])
                else:
                    d_sub = row[candidate_idxs]
                    candidate_idxs = candidate_idxs[np.argsort(d_sub)[:task_k]]
                    candidate_idx = int(np.random.choice(candidate_idxs))
                
                candidate_point = points[candidate_idx]
                candidate_value = values[[candidate_idx]]
                trajectory.append(np.concatenate([candidate_point, candidate_value], axis=0))
                starting_idx = candidate_idx
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
        task_output_path = os.path.join(
            output_dir,
            per_task_latent_train_filename(
                task_name, task_n_traj, traj_len, task_k, task_eps, _latent
            ),
        )
        task_obj = [our_data, our_data_vals, pr, cumulative_regret_to_go, timesteps]
        pkl.dump(task_obj, open(task_output_path, "wb"))
        print(f"任务 {task_name} 的轨迹数据已保存至: {task_output_path}")
        
        latent_d = int(getattr(vae_args, "latent_dim", 32))
        vae_info = {
            "latent_dim": latent_d,
            "vae_path": os.path.join(model_save_dir, f"vae_latent{latent_d}.pt"),
            "fixed_dim": fixed_dim,
            "task": task_name,
            "original_dim": task_dims_info[task_name]["original_dim"],
        }
        if is_real_world_fewshot_task(task_name):
            # 供 evaluate 中 D(best)=few-shot 池内 max(y) 与 construct 子集一致
            vae_info["real_world_fewshot_k"] = fewshot_k
            vae_info["real_world_fewshot_mode"] = fewshot_mode
            vae_info["real_world_fewshot_seed"] = int(seed)
        if pretrained_vae_info:
            vae_info["pretrained_vae_info_source"] = os.path.abspath(pretrained_vae_info)
        vae_info_path = os.path.join(output_dir, generated_vae_info_filename(_latent))
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
    parser.add_argument(
        "--traj_params_json",
        type=str,
        default=None,
        help="JSON：按任务覆盖 n_traj/k/eps（与 train/evaluate 同路径）",
    )
    parser.add_argument(
        '--cpu_threads',
        type=int,
        default=None,
        help="限制 CPU 线程数（OpenMP/BLAS）；等价于环境变量 CPU_THREADS",
    )
    parser.add_argument(
        "--fewshot_k",
        type=int,
        default=None,
        help="real-world：取 k 个点（与 --fewshot_mode 搭配；None=全量）",
    )
    parser.add_argument(
        "--fewshot_mode",
        type=str,
        default="all",
        choices=("all", "random", "worst"),
        help="real-world：random=随机 k 点；worst=y 最小的 k 点（越大越好）",
    )
    parser.add_argument(
        "--pretrained_vae_info",
        type=str,
        default=None,
        help="单任务 real-world 必填：多任务预训练 vae_info.p，用于在 few-shot 上微调 VAE",
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=50,
        help="real-world 微调轮数（默认 50）",
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=3e-5,
        help="real-world 微调学习率（默认 3e-5）",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=32,
        help="VAE 隐空间维度；32 保持历史文件名，其它维度使用 _vae_latent{d}_train.p / vae_latent{d}.pt / mixed_*_latent{d}.p",
    )

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
        traj_params_json=args.traj_params_json,
        fewshot_k=args.fewshot_k,
        fewshot_mode=args.fewshot_mode,
        pretrained_vae_info=args.pretrained_vae_info,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        latent_dim=int(args.latent_dim),
    )
    sys.exit(0)
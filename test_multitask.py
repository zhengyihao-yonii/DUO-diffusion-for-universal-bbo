# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_vae import main as train_vae_main
from construct_trajectories import construct_trajectories

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def test_multitask_pipeline():
    """
    测试完整的多任务训练流程
    """
    print("开始测试多任务训练流程...")
    
    # 设置随机种子
    set_seed(42)
    
    # 选择要测试的任务列表
    tasks_list = ['tfbind8', 'tfbind10']  # 选择两个小数据集进行快速测试
    fixed_dim = 128  # 统一的固定维度
    
    print(f"测试任务列表: {tasks_list}")
    print(f"固定维度: {fixed_dim}")
    
    # 1. 测试VAE多任务训练
    print("\n=== 测试VAE多任务训练 ===")
    vae_args = argparse.Namespace(
        tasks=tasks_list,
        batch_size=64,
        lr=1e-4,
        epochs=5,  # 使用少量epoch进行快速测试
        latent_dim=16,
        hidden_dim=128,
        fixed_dim=fixed_dim,
        save_dir="./models/test_multitask/vae",
        train=True,
        test=False
    )
    
    try:
        vae_model, vae_save_path, dataset_info = train_vae_main(vae_args)
        print(f"VAE训练成功，模型保存路径: {vae_save_path}")
        print(f"数据集信息: {dataset_info}")
    except Exception as e:
        print(f"VAE训练失败: {e}")
        return False
    
    # 2. 测试多任务轨迹构建
    print("\n=== 测试多任务轨迹构建 ===")
    try:
        # 使用小参数进行快速测试
        output_dir, trajectories_info = construct_trajectories(
            tasks_list=tasks_list,
            frac=0.1,  # 使用小部分数据进行测试
            sigma=0.01,
            seed=42,
            fixed_dim=fixed_dim
        )
        print("多任务轨迹构建成功")
    except Exception as e:
        print(f"多任务轨迹构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== 测试完成 ===")
    return True

if __name__ == "__main__":
    success = test_multitask_pipeline()
    if success:
        print("🎉 完整的多任务训练流程测试成功！")
        sys.exit(0)
    else:
        print("❌ 多任务训练流程测试失败，请检查代码。")
        sys.exit(1)
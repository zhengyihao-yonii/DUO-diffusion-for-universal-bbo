#!/bin/bash

# 新的bash脚本：支持对单个任务运行多次，每次使用不同的随机种子
# VAE训练和轨迹构建只执行一次，然后多次运行训练和评估
# 使用方式：bash run_multiple_times.sh <task_name> <num_runs> <n_traj> <k> <eps>

# 检查参数数量
if [ $# -ne 5 ]; then
    echo "使用方式: bash run_multiple_times.sh <task_name> <num_runs> <n_traj> <k> <eps>"
    echo "示例: bash run_multiple_times.sh dkitty 3 4000 20 0.01"
    exit 1
fi

# 解析参数
task_name=$1
num_runs=$2
n_traj=$3
k=$4
eps=$5

# 激活环境（如果需要）
# source activate gtg

# 基础命令路径
python_cmd="/home/xk/anaconda3/envs/gtg/bin/python"
project_dir="/data/xk/zyh_dfgo/GTGdfgo"

# 创建基础结果目录
base_dir="$project_dir/results/${task_name}_multiple_runs"
mkdir -p "$base_dir"

# 只执行一次的步骤
echo "=== 开始执行一次性步骤 ==="

# 1. 运行 train_vae.py（只执行一次）
echo "Step 1: 训练 VAE 模型..."
vae_log="$base_dir/vae_train.log"
$python_cmd "$project_dir/train_vae.py" --task "$task_name" > "$vae_log" 2>&1
if [ $? -ne 0 ]; then
    echo "VAE 训练失败，查看日志: $vae_log"
    exit 1
fi

# 2. 运行 construct_trajectories.py（只执行一次）
echo "Step 2: 构建轨迹数据..."
construct_log="$base_dir/construct_trajectories.log"
$python_cmd "$project_dir/construct_trajectories.py" --task "$task_name" > "$construct_log" 2>&1
if [ $? -ne 0 ]; then
    echo "轨迹构建失败，查看日志: $construct_log"
    exit 1
fi

echo "一次性步骤完成！"

# 运行多次训练和评估
echo "\n=== 开始运行 $num_runs 次训练和评估 ==="

for ((run=0; run<num_runs; run++)); do
    echo "\n--- 运行第 $((run+1))/$num_runs 次 ---"
    
    # 设置随机种子
    seed=$run
    
    # 创建本次运行结果目录
    run_dir="$base_dir/run${run+1}_seed${seed}"
    mkdir -p "$run_dir"
    
    # 3. 运行 train.py
echo "Step 3: 训练扩散模型..."
$python_cmd "$project_dir/train.py" --task "$task_name" --n_traj "$n_traj" --k "$k" --eps "$eps" --seed "$seed" > "$run_dir/train.log" 2>&1
if [ $? -ne 0 ]; then
    echo "模型训练失败，查看日志: $run_dir/train.log"
    continue
fi

# 4. 运行 evaluate.py
echo "Step 4: 评估模型..."
$python_cmd "$project_dir/evaluate.py" --task "$task_name" --n_traj "$n_traj" --k "$k" --eps "$eps" --seed "$seed" > "$run_dir/evaluate.log" 2>&1
if [ $? -ne 0 ]; then
    echo "模型评估失败，查看日志: $run_dir/evaluate.log"
    continue
fi

    echo "运行 $((run+1)) 完成，结果保存在: $run_dir"
done

echo "\n所有运行已完成！"
echo "结果目录: $base_dir/"

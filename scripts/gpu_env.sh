# GTGdfgo：供 run_multitask.sh / run_singletask.sh source，用于多卡服务器上选择物理 GPU。
#
# 用法（任选其一）:
#   1) 运行前: export CUDA_VISIBLE_DEVICES=2
#   2) 运行前: GPU_ID=2 bash run_multitask.sh ...   （仅当未设置 CUDA_VISIBLE_DEVICES 时生效）
#   3) 在 run_*.sh 顶部附近取消注释: export CUDA_VISIBLE_DEVICES=1
#
# 设置后，PyTorch 通常只看到一张「逻辑 cuda:0」，可避免多任务抢同一张物理卡导致 OOM。

if [[ -n "${GPU_ID:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[GPU] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

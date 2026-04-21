# Third-party benchmarks (DUO)

## SOO-Bench

### 与 `universal-offline-bbo-main` 文档的差异（常见报错原因）

`universal-offline-bbo-main/README.md` 里的安装命令包含：

```bash
pip install -e ./revive_hybrid/
```

**当前** [SOO-Bench 官方仓库](https://github.com/zhuyiyi-123/SOO-Bench) **根目录下往往没有 `revive_hybrid/`**，继续执行会报错，例如：

- `ERROR: ./revive_hybrid/ is not a valid editable requirement`
- 或 `No such file or directory`

官方 README（2025）推荐流程是：**只需** `pip install -e .` 再 `pip install -r requirements.txt`，见仓库内 `install.sh`。

本仓库的安装脚本会先检测：若存在 `revive_hybrid/` 再安装，否则跳过。

### 方式一：脚本安装（推荐）

在 **已激活** 的 Conda 环境（如 `gtg`）中，于 **DUO 根目录**执行：

```bash
bash scripts/setup_soo_bench.sh
```

### 方式二：手动安装

```bash
cd /path/to/DUO/thirdparty_benchmark
git clone https://github.com/zhuyiyi-123/SOO-Bench.git
cd SOO-Bench/

# 仅当仓库里确实有该目录时（旧版/ fork）：
# pip install -e ./revive_hybrid/

pip install -e .
# Python 3.8：请用 DUO 提供的兼容依赖（见下），勿直接用上游 requirements.txt
pip install -r ../soo_bench_requirements_dfgo.txt
cd ../..
```

**常见 pip 报错：**

| 现象 | 原因 | 处理 |
|------|------|------|
| `No matching distribution found for revive==0.7.3` | 上游 `setup.py` 固定旧版 revive，部分镜像只有 1.0.0 | 本仓库已把本地 `SOO-Bench/setup.py` 的 `install_requires` 清空（核心不依赖 revive）；或 `pip install -e . --no-deps` |
| `No matching distribution found for tensorflow_probability==0.24` | tfp 0.24 需 **Python ≥3.9** | 使用 `thirdparty_benchmark/soo_bench_requirements_dfgo.txt`（将 tfp 固定为 `0.21.0`），或换 **Python 3.9+** 后按上游 `requirements.txt` 安装 |

### 环境冲突说明

SOO-Bench 的 `requirements.txt` 含 `tensorflow`、`numpy<1.24` 等，与部分 **仅用于 Design-Bench / DUO** 的环境可能冲突。若安装失败，可：

1. **单独建环境** 专用于 SOO（官方也推荐 `conda create -n soo-bench python=3.8`），仅在做 SOO 数据实验时激活；或  
2. 在虚拟环境中 **手动** 按需安装子集，避免与 `torch`/`numpy` 强冲突。

### 验证

```bash
python -c "from soo_bench.Taskdata import OfflineTask, REGISTERED_TASK; print(REGISTERED_TASK.keys())"
```

### 与 DUO 主流程

当前 DUO 默认数据仍来自 **Design-Bench**（`design_bench.make` 等）。安装 SOO-Bench 仅提供 **`soo_bench` Python 包**；要把任务接进 `construct_trajectories` / `train_vae`，需要额外编写数据适配。

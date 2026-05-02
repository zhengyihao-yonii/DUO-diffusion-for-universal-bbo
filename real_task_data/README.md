# Real-world 数据（few-shot / full）

使用单文件聚合格式：`real_task_data/meta_dataset.json`。

## 目录结构（相对 DUO 根目录）

```
real_task_data/
  meta_dataset.json
```

`meta_dataset.json` 顶层按任务名组织（`LunarLander` / `RobotPush` / `Rover`），每个任务下面可以有多个来源 key；
每个来源必须包含 `X`（二维数组）和 `y`（一维数组），代码会把同一任务下所有来源拼接在一起。

DUO 任务短名与目录对应关系在 `diffuser/datasets/real_world_fewshot.py` 的 `TASK_KEY_TO_DATA_DIR`（`lunar_lander` → `LunarLander` 等）。

## 环境变量

- **`GTG_REAL_WORLD_FEWSHOT_DIR`**：若数据不在 `<DUO>/real_task_data`，设为**包含** `meta_dataset.json` 的父目录绝对路径。

## 分析实验

`all_improved` 等多任务目录说明见 `scripts/analyze_eval_results.py` 与 `run_multitask.sh` 注释；`EVAL_ALL_TASK_FRAC_SIG` 可按实验调整。

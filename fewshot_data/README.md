# Few-shot real-world 数据

每个任务一个子目录，内含 **`similar/`**、**`unsimilar/`**，其中为含 **`X`**、**`y`** 的 JSON 文件。

## 目录结构（相对 DUO 根目录）

```
fewshot_data/
  LunarLander/
    similar/*.json
    unsimilar/*.json
  RobotPush/
    similar/*.json
    unsimilar/*.json
  Rover/
    similar/*.json
    unsimilar/*.json
```

DUO 任务短名与目录对应关系在 `diffuser/datasets/real_world_fewshot.py` 的 `TASK_KEY_TO_DATA_DIR`（`lunar_lander` → `LunarLander` 等）。

## 环境变量

- **`GTG_REAL_WORLD_FEWSHOT_DIR`**：若数据不在 `<DUO>/fewshot_data`，设为**包含**上述 `LunarLander` / `RobotPush` / `Rover` 父目录的绝对路径。

## 分析实验

`all_improved` 等多任务目录说明见 `scripts/analyze_eval_results.py` 与 `run_multitask.sh` 注释；`EVAL_ALL_TASK_FRAC_SIG` 可按实验调整。

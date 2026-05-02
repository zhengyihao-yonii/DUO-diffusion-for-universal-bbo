# QualityExperiment

English doc: Synthetic **latent objective landscapes** (RGDiff-style **filled contours**: color = objective level / basins) plus **diffusion denoise trajectories** for the four DUO settings aligned with `visualize.sh`:

| Tag | Meaning |
|-----|---------|
| `st_duo` | Single-task, no text |
| `st_text` | Single-task + text embed |
| `mt_label` | Multitask + task index (`mt_task` in scripts) |
| `mt_text` | Multitask + text (`mt_text` / `mttextonly` family) |

## Metadata 从哪里来？（与 comparison1 **完全一致**）

- **唯一作者**：`comparisonExperiment/experiment1/run_exp1.py` 写入  
  `UniSO/data/exp1_<task>.meta.json`（字段 **`metadata_text`**）与同目录 **`exp1_<task>.metadata`**（纯文本）。
- **QualityExperiment 不再写 metadata**：只 **读取** 上述文件；Shift 阶段只是把 **`metadata_text` 编成向量**（与 step 3 里训练域矩阵 **同一套 encoder**）。
- 若你看到旧目录里没有 `metadata_text`，那是旧版脚本生成的数据——**重新跑一次 `run_exp1.py`** 即可。

## Naming：`gap0p500` 与目录对应关系

`run_exp1.py` 使用：

- **DUO PKL 根目录**：`generated_datasets/exp1_gap<GAP>/`，其中 `<GAP>` 为 `--gaps` 里每个值的三位小数把 `.` 换成 `p`（例如 `0.5` → **`exp1_gap0p500`**）。
- **`--out_root`**：在其下再建一层 **`gap0p500/`** 写 `family_meta.json`（每个 gap 一轮）。

示例：`--out_root results/comparison1/exp1 --gaps 0.5` →  
`results/comparison1/exp1/gap0p500/family_meta.json`。

Quality / README 里凡涉及 gap，一律写 **`gap0p500`**，不要混用 `gap05`。

## Data (`comparisonExperiment/experiment1`)

```bash
cd DUO
python comparisonExperiment/experiment1/run_exp1.py \
  --out_root results/comparison1/exp1 \
  --gaps 0.5 \
  --uniso_data_dir ../UniSO/data
```

**Writes (paths relative to DUO)：**

| Artifact | Location |
|----------|----------|
| UniSO points + metadata text | `../UniSO/data/exp1_<task_id>.json`, `.metadata` |
| Structured meta（含 **`metadata_text`**） | `../UniSO/data/exp1_<task_id>.meta.json` |
| Few-shot meta（测试任务，`metadata_text` 含 few-shot 说明） | `../UniSO/data/exp1_<task_id>_fewshot.meta.json` |
| DUO PKLs | `generated_datasets/exp1_gap0p500/<task_id>_h*.pkl` |
| Few-shot DUO PKLs | `generated_datasets/exp1_gap0p500/<task_id>_fewshot_h*.pkl` |
| Family summary | `results/comparison1/exp1/gap0p500/family_meta.json` |

## Checkpoints

Train with `comparisonExperiment/experiment1/duo_train_and_sample.py` or main DUO；保存 **full EMA**（`GaussianDiffusion`），与 `QualityExperiment/trace_sampling.py` 兼容。

```
results/quality_exp/
  ckpt_st_duo.pt
  ckpt_st_text.pt
  ckpt_mt_label.pt
  ckpt_mt_text.pt
```

Few-shot 微调（可选）：suite 里 `--ckpt_*_fs`。

## [T,E] 矩阵：从 comparison1 的 meta **直接编码**（非新 metadata）

```bash
cd DUO
python -m QualityExperiment.build_task_text_embeds \
  --uniso_data_dir ../UniSO/data \
  --n_train_tasks 5 \
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --out_npy results/quality_exp/task_embeds_gap0p500.npy
```

输出 `task_embeds_gap0p500.npy`，行 `i` = **`exp1_D_train_{i+1}.meta.json` 的 `metadata_text`** 的嵌入。

## Wandb keys + 本地文件

- **`quality_figure/latent_landscape`**
- **`quality_table/trajectories_<tag>`**
- **`quality_table/trajectories_all_methods`**

NPZ/PNG：`quality_trace_*.npz`；若设 `--local_out_dir`，另存 `*_latent_landscape.png`。

## 完整流程（命名已对齐 `gap0p500`）

```bash
cd /data/xk/zyh_dfgo/DUO
pip install wandb matplotlib sentence-transformers

# 1) 数据（comparison1 唯一写 metadata 的一步）
python comparisonExperiment/experiment1/run_exp1.py \
  --out_root results/comparison1/exp1 \
  --gaps 0.5 \
  --uniso_data_dir ../UniSO/data

# 2) 训练四个 ckpt（示例）
python comparisonExperiment/experiment1/duo_train_and_sample.py \
  --train_pkl generated_datasets/exp1_gap0p500/D_train_1_h32_n100_dx16.pkl \
  --horizon 32 --train_steps 2000 --sample_traj 32 \
  --mode train_and_sample \
  --save_ckpt results/quality_exp/ckpt_st_duo.pt --no_wandb \
  --out_jsonl results/quality_exp/st_duo_candidates.jsonl

# 3) [T,E]：从 **已有** meta.json 编码（不写新文案）
python -m QualityExperiment.build_task_text_embeds \
  --uniso_data_dir ../UniSO/data \
  --n_train_tasks 5 \
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --out_npy results/quality_exp/task_embeds_gap0p500.npy

# 4) Quality suite（train / shift ZS / shift FS）
python -m QualityExperiment.run_quality_suite \
  --uniso_data_dir ../UniSO/data \
  --pkl_dir generated_datasets/exp1_gap0p500 \
  --phases train_domain,shift_zero_shot,shift_few_shot \
  --task_text_embeds_npy results/quality_exp/task_embeds_gap0p500.npy \
  --mt_num_tasks 5 \
  --text_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --local_out_dir results/quality_exp/artifacts_gap0p500 \
  --ckpt_st_duo results/quality_exp/ckpt_st_duo.pt \
  --ckpt_st_text results/quality_exp/ckpt_st_text.pt \
  --ckpt_mt_label results/quality_exp/ckpt_mt_label.pt \
  --ckpt_mt_text results/quality_exp/ckpt_mt_text.pt \
  --wandb_project duo-quality-suite \
  --wandb_group exp1_gap0p500_full
```

Shift 阶段 **`st_text`/`mt_text`**：默认从当前 run 的 **`meta_json`** 读 **`metadata_text`** 再编码（与 comparison1 同源）；也可用 **`--held_out_text_embed_npy`** 强行指定向量。

## Single-task CLI

```bash
python -m QualityExperiment.run_landscape_experiment \
  --meta_json ../UniSO/data/exp1_D_train_1.meta.json \
  --train_pkl generated_datasets/exp1_gap0p500/D_train_1_h32_n100_dx16.pkl \
  --phase_label train_domain \
  --task_text_embeds_npy results/quality_exp/task_embeds_gap0p500.npy \
  --ckpt_st_duo results/quality_exp/ckpt_st_duo.pt \
  --wandb_project duo-quality --wandb_group demo
```

## Notes

- Landscapes：latent \(z\)（`d_z=2`），\(x\) 由 `meta.json` 的 `A,b` 最小二乘回投。
- 主 DUO 真实任务 metadata 与 exp1 **无关**；本目录仅服务 comparison Experiment 1 人造族。

## Experiment 1: Geometry-distance generalization gap (DUO vs UniSO-T)

### Goal
Study how the **generalization gap** of DUO / UniSO-T changes as the **geometric distance**
between **train-task instances** and a **test-task instance** increases.

We construct a *family* of synthetic BoTorch-style black-box tasks that:
- share the same **low-dimensional latent objective** \(z \mapsto f(z)\)
- differ only by an **instance transform** mapping \(z \to x\) (affine + optional noise)
- allow a single scalar **gap** parameter that monotonically increases the geometric distance
  between train instances (A/B/C) and test instance (D).

### Outputs
All outputs are written under:
- `DUO/results/comparison1/exp1/...`

This experiment generates **two dataset formats**:
- **DUO**: trajectory PKLs in `generated_datasets/exp1_*` (single-task pkl format compatible
  with `PointRegretDataset`)
- **UniSO**: `UniSO/data/*.json` + `UniSO/data/*.metadata` (OmniPred text-value dataset)

### Entry points
- `run_exp1.py`: generate tasks + datasets + a small evaluation table scaffold
- `run_comparison_v4_duo_uniso.sh`: **DUO v4 `mt_text` vs UniSO-T** on scene-aware `data_exp1_sim_*_3`
  (export trace candidates → UniSO `run_exp1_uniso_scene_aware.sh` → oracle eval → LaTeX table)
- `export_quality_trace_jsonl.py` / `export_duo_uniso_comparison_table.py`: helpers for the v4 comparison

### Notes
- Keep this experiment small by default (few tasks, short horizon) to compare architectures.
- Avoid changing DUO/UniSO internal logic; this folder should only generate data + run commands.


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

### Notes
- Keep this experiment small by default (few tasks, short horizon) to compare architectures.
- Avoid changing DUO/UniSO internal logic; this folder should only generate data + run commands.


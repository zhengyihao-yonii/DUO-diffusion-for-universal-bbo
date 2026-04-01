# Evaluate 结果汇总（GTGdfgo vs GTG）

- GTGdfgo results: `/data/xk/zyh_dfgo/GTGdfgo/results`
- GTG results: `/data/xk/zyh_dfgo/GTG/results`

各实验在多次 run（如 `run*_seed*`）上聚合：**max** / **nmax** 为 `mean ± std`（std 为样本标准差，单次 run 时 std 记为 0）。

| experiment | task | GTGdfgo n | max (mean±std) | nmax (mean±std) | GTG n | max (mean±std) | nmax (mean±std) |
|------------|------|-----------|----------------|-----------------|-------|----------------|-----------------|
| ant_multiple_runs | ant | 5 | 524.331 ± 61.415 | 0.933 ± 0.063 | 5 | 452.330 ± 61.502 | 0.859 ± 0.063 |
| dkitty_multiple_runs | dkitty | 5 | 248.027 ± 13.147 | 0.924 ± 0.011 | 5 | 264.857 ± 17.571 | 0.938 ± 0.014 |
| gtopx2_multiple_runs | gtopx2 | 2 | -90.412 ± 0.840 | 1.000 ± 0.000 | — | — | — |
| multitask_dkitty_ant | dkitty | 1 | 289.591 ± 0.000 | 0.958 ± 0.000 | — | — | — |
| tfbind10_multiple_runs | tfbind10 | 5 | 0.626 ± 0.148 | 0.623 ± 0.037 | 5 | 0.632 ± 0.106 | 0.625 ± 0.027 |
| tfbind8_multiple_runs | tfbind8 | 5 | 0.950 ± 0.025 | 0.950 ± 0.025 | 5 | 0.934 ± 0.030 | 0.934 ± 0.030 |

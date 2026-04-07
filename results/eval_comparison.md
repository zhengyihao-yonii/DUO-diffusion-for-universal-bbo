# Evaluate 结果汇总（GTGdfgo vs GTG）

- GTGdfgo results: `/data/xk/zyh_dfgo/GTGdfgo/results`
- GTG results: `/data/xk/zyh_dfgo/GTG/results`

各实验在多次 run（如 `run*_seed*`）上聚合：**max** / **nmax** 为 `mean ± std`（std 为样本标准差，单次 run 时 std 记为 0）。

| experiment | task | GTGdfgo n | max (mean±std) | nmax (mean±std) | GTG n | max (mean±std) | nmax (mean±std) |
|------------|------|-----------|----------------|-----------------|-------|----------------|-----------------|
| ant_multiple_runs | ant | 5 | 523.935 ± 55.443 | 0.932 ± 0.057 | 5 | 452.330 ± 61.502 | 0.859 ± 0.063 |
| dkitty_multiple_runs | dkitty | 5 | 250.007 ± 14.505 | 0.926 ± 0.012 | 5 | 264.857 ± 17.571 | 0.938 ± 0.014 |
| gtopx2_multiple_runs | gtopx2 | 10 | -102.223 ± 12.230 | 1.000 ± 0.000 | 5 | -72.163 ± 8.495 | 1.000 ± 0.000 |
| gtopx3_multiple_runs | gtopx3 | 5 | -51.762 ± 3.893 | 1.000 ± 0.000 | 5 | -53.866 ± 9.877 | 1.000 ± 0.000 |
| gtopx4_multiple_runs | gtopx4 | 5 | -68.361 ± 13.480 | 1.000 ± 0.000 | 5 | -72.749 ± 6.632 | 1.000 ± 0.000 |
| gtopx6_multiple_runs | gtopx6 | 5 | -49.386 ± 8.267 | 1.000 ± 0.000 | 5 | -55.834 ± 9.655 | 1.000 ± 0.000 |
| multitask_ant_dkitty | ant | 1 | 586.691 ± 0.000 | 0.996 ± 0.000 | — | — | — |
| multitask_ant_dkitty | dkitty | 1 | 289.591 ± 0.000 | 0.958 ± 0.000 | — | — | — |
| multitask_gtopx2_gtopx3_gtopx4_gtopx6 | gtopx2 | 5 | -99.293 ± 22.175 | 1.000 ± 0.000 | — | — | — |
| multitask_gtopx2_gtopx3_gtopx4_gtopx6 | gtopx3 | 5 | -58.310 ± 14.914 | 1.000 ± 0.000 | — | — | — |
| multitask_gtopx2_gtopx3_gtopx4_gtopx6 | gtopx4 | 5 | -69.035 ± 7.255 | 1.000 ± 0.000 | — | — | — |
| multitask_gtopx2_gtopx3_gtopx4_gtopx6 | gtopx6 | 5 | -55.267 ± 7.939 | 1.000 ± 0.000 | — | — | — |
| multitask_tfbind10_tfbind8 | tfbind10 | 5 | 0.626 ± 0.131 | 0.623 ± 0.033 | — | — | — |
| multitask_tfbind10_tfbind8 | tfbind8 | 5 | 0.973 ± 0.009 | 0.973 ± 0.009 | — | — | — |
| tfbind10_multiple_runs | tfbind10 | 5 | 0.626 ± 0.148 | 0.623 ± 0.037 | 5 | 0.632 ± 0.106 | 0.625 ± 0.027 |
| tfbind8_multiple_runs | tfbind8 | 5 | 0.954 ± 0.019 | 0.954 ± 0.019 | 5 | 0.934 ± 0.030 | 0.934 ± 0.030 |

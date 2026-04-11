# Evaluate 结果汇总（GTGdfgo vs GTG）

- GTGdfgo results: `/data/xk/zyh_dfgo/GTGdfgo/results`
- GTG results: `/data/xk/zyh_dfgo/GTG/results`

各实验在多次 run（如 `run*_seed*`）上聚合：**max** / **nmax** 为 `mean ± std`（std 为样本标准差，单次 run 时 std 记为 0）。
仅统计日志中能解析出 `[task] max_ep_reward` / `nmax_ep_reward` 的 run；评估中断或报错（无上述行）的目录不计入 `n`。
表格行顺序：单任务按 ant → dkitty → tfbind8 → tfbind10 → gtopx；多任务按联立规模从小到大。
列 **GTGdfgo 小组 multi** / **全任务 multi**：按当前行的 `task`，分别从对应小组 multitask 实验与 `multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8` 中读取该任务的聚合指标（与左侧 `experiment` 列无关）。

| experiment | task | GTGdfgo n | max | nmax | GTG n | max | nmax | 小组 multi n | 小组 max | 小组 nmax | 全任务 multi n | 全 max | 全 nmax |
|------------|------|-----------|------|------|-------|------|------|--------------|----------|-----------|----------------|--------|---------|
| ant_multiple_runs | ant | 5 | 523.935 ± 55.443 | 0.932 ± 0.057 | 5 | 452.330 ± 61.502 | 0.859 ± 0.063 | 5 | 566.325 ± 13.636 | 0.976 ± 0.014 | 5 | 570.945 ± 15.468 | 0.980 ± 0.016 |
| dkitty_multiple_runs | dkitty | 5 | 250.007 ± 14.505 | 0.926 ± 0.012 | 5 | 264.857 ± 17.571 | 0.938 ± 0.014 | 5 | 270.165 ± 38.116 | 0.942 ± 0.031 | 5 | 296.782 ± 11.719 | 0.964 ± 0.010 |
| tfbind8_multiple_runs | tfbind8 | 5 | 0.954 ± 0.019 | 0.954 ± 0.019 | 5 | 0.934 ± 0.030 | 0.934 ± 0.030 | 5 | 0.973 ± 0.009 | 0.973 ± 0.009 | 5 | 0.698 ± 0.101 | 0.698 ± 0.101 |
| tfbind10_multiple_runs | tfbind10 | 5 | 0.626 ± 0.148 | 0.623 ± 0.037 | 5 | 0.632 ± 0.106 | 0.625 ± 0.027 | 5 | 0.626 ± 0.131 | 0.623 ± 0.033 | 5 | 0.215 ± 0.219 | 0.520 ± 0.055 |
| gtopx2_multiple_runs | gtopx2 | 5 | -104.344 ± 11.523 | 1.000 ± 0.000 | 5 | -72.163 ± 8.495 | 1.000 ± 0.000 | 5 | -99.293 ± 22.175 | 1.000 ± 0.000 | 5 | -99.013 ± 42.811 | 1.000 ± 0.000 |
| gtopx3_multiple_runs | gtopx3 | 5 | -51.762 ± 3.893 | 1.000 ± 0.000 | 5 | -53.866 ± 9.877 | 1.000 ± 0.000 | 5 | -58.310 ± 14.914 | 1.000 ± 0.000 | 5 | -86.325 ± 61.474 | 1.000 ± 0.000 |
| gtopx4_multiple_runs | gtopx4 | 5 | -68.361 ± 13.480 | 1.000 ± 0.000 | 5 | -72.749 ± 6.632 | 1.000 ± 0.000 | 5 | -69.035 ± 7.255 | 1.000 ± 0.000 | 5 | -115.660 ± 60.530 | 1.000 ± 0.000 |
| gtopx6_multiple_runs | gtopx6 | 10 | -54.001 ± 9.467 | 1.000 ± 0.000 | 5 | -55.834 ± 9.655 | 1.000 ± 0.000 | 5 | -55.267 ± 7.939 | 1.000 ± 0.000 | 5 | -82.709 ± 45.592 | 1.000 ± 0.000 |
| multitask_ant_dkitty | ant | 5 | 566.325 ± 13.636 | 0.976 ± 0.014 | — | — | — | 5 | 566.325 ± 13.636 | 0.976 ± 0.014 | 5 | 570.945 ± 15.468 | 0.980 ± 0.016 |
| multitask_ant_dkitty | dkitty | 5 | 270.165 ± 38.116 | 0.942 ± 0.031 | — | — | — | 5 | 270.165 ± 38.116 | 0.942 ± 0.031 | 5 | 296.782 ± 11.719 | 0.964 ± 0.010 |
| multitask_tfbind10_tfbind8 | tfbind8 | 5 | 0.973 ± 0.009 | 0.973 ± 0.009 | — | — | — | 5 | 0.973 ± 0.009 | 0.973 ± 0.009 | 5 | 0.698 ± 0.101 | 0.698 ± 0.101 |
| multitask_tfbind10_tfbind8 | tfbind10 | 5 | 0.626 ± 0.131 | 0.623 ± 0.033 | — | — | — | 5 | 0.626 ± 0.131 | 0.623 ± 0.033 | 5 | 0.215 ± 0.219 | 0.520 ± 0.055 |
| multitask_gtopx2_gtopx3_gtopx4_gtopx6 | gtopx2 | 5 | -99.293 ± 22.175 | 1.000 ± 0.000 | — | — | — | 5 | -99.293 ± 22.175 | 1.000 ± 0.000 | 5 | -99.013 ± 42.811 | 1.000 ± 0.000 |
| multitask_gtopx2_gtopx3_gtopx4_gtopx6 | gtopx3 | 5 | -58.310 ± 14.914 | 1.000 ± 0.000 | — | — | — | 5 | -58.310 ± 14.914 | 1.000 ± 0.000 | 5 | -86.325 ± 61.474 | 1.000 ± 0.000 |
| multitask_gtopx2_gtopx3_gtopx4_gtopx6 | gtopx4 | 5 | -69.035 ± 7.255 | 1.000 ± 0.000 | — | — | — | 5 | -69.035 ± 7.255 | 1.000 ± 0.000 | 5 | -115.660 ± 60.530 | 1.000 ± 0.000 |
| multitask_gtopx2_gtopx3_gtopx4_gtopx6 | gtopx6 | 5 | -55.267 ± 7.939 | 1.000 ± 0.000 | — | — | — | 5 | -55.267 ± 7.939 | 1.000 ± 0.000 | 5 | -82.709 ± 45.592 | 1.000 ± 0.000 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8 | ant | 5 | 570.945 ± 15.468 | 0.980 ± 0.016 | — | — | — | 5 | 566.325 ± 13.636 | 0.976 ± 0.014 | 5 | 570.945 ± 15.468 | 0.980 ± 0.016 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8 | dkitty | 5 | 296.782 ± 11.719 | 0.964 ± 0.010 | — | — | — | 5 | 270.165 ± 38.116 | 0.942 ± 0.031 | 5 | 296.782 ± 11.719 | 0.964 ± 0.010 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8 | tfbind8 | 5 | 0.698 ± 0.101 | 0.698 ± 0.101 | — | — | — | 5 | 0.973 ± 0.009 | 0.973 ± 0.009 | 5 | 0.698 ± 0.101 | 0.698 ± 0.101 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8 | tfbind10 | 5 | 0.215 ± 0.219 | 0.520 ± 0.055 | — | — | — | 5 | 0.626 ± 0.131 | 0.623 ± 0.033 | 5 | 0.215 ± 0.219 | 0.520 ± 0.055 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8 | gtopx2 | 5 | -99.013 ± 42.811 | 1.000 ± 0.000 | — | — | — | 5 | -99.293 ± 22.175 | 1.000 ± 0.000 | 5 | -99.013 ± 42.811 | 1.000 ± 0.000 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8 | gtopx3 | 5 | -86.325 ± 61.474 | 1.000 ± 0.000 | — | — | — | 5 | -58.310 ± 14.914 | 1.000 ± 0.000 | 5 | -86.325 ± 61.474 | 1.000 ± 0.000 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8 | gtopx4 | 5 | -115.660 ± 60.530 | 1.000 ± 0.000 | — | — | — | 5 | -69.035 ± 7.255 | 1.000 ± 0.000 | 5 | -115.660 ± 60.530 | 1.000 ± 0.000 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8 | gtopx6 | 5 | -82.709 ± 45.592 | 1.000 ± 0.000 | — | — | — | 5 | -55.267 ± 7.939 | 1.000 ± 0.000 | 5 | -82.709 ± 45.592 | 1.000 ± 0.000 |
| ant_multiple_runs_retcond | ant | 5 | 523.345 ± 54.535 | 0.932 ± 0.056 | 5 | 454.594 ± 29.590 | 0.861 ± 0.030 | 5 | 566.325 ± 13.636 | 0.976 ± 0.014 | 5 | 570.945 ± 15.468 | 0.980 ± 0.016 |
| dkitty_multiple_runs_retcond | dkitty | 5 | 271.266 ± 10.596 | 0.943 ± 0.009 | 5 | 167.231 ± 79.140 | 0.858 ± 0.065 | 5 | 270.165 ± 38.116 | 0.942 ± 0.031 | 5 | 296.782 ± 11.719 | 0.964 ± 0.010 |
| gtopx2_multiple_runs_retcond | gtopx2 | 5 | -88.551 ± 10.445 | 1.000 ± 0.000 | 5 | -84.446 ± 26.171 | 1.000 ± 0.000 | 5 | -99.293 ± 22.175 | 1.000 ± 0.000 | 5 | -99.013 ± 42.811 | 1.000 ± 0.000 |
| gtopx3_multiple_runs_retcond | gtopx3 | 5 | -58.489 ± 9.208 | 1.000 ± 0.000 | 5 | -54.650 ± 15.913 | 1.000 ± 0.000 | 5 | -58.310 ± 14.914 | 1.000 ± 0.000 | 5 | -86.325 ± 61.474 | 1.000 ± 0.000 |
| gtopx4_multiple_runs_retcond | gtopx4 | 4 | -71.906 ± 18.009 | 1.000 ± 0.000 | 5 | -73.852 ± 13.080 | 1.000 ± 0.000 | 5 | -69.035 ± 7.255 | 1.000 ± 0.000 | 5 | -115.660 ± 60.530 | 1.000 ± 0.000 |
| gtopx6_multiple_runs_retcond | gtopx6 | 5 | -54.051 ± 5.265 | 1.000 ± 0.000 | 5 | -58.879 ± 7.985 | 1.000 ± 0.000 | 5 | -55.267 ± 7.939 | 1.000 ± 0.000 | 5 | -82.709 ± 45.592 | 1.000 ± 0.000 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8_retcond_textcond_mttextonly | ant | 5 | 564.013 ± 8.828 | 0.973 ± 0.009 | — | — | — | 5 | 566.325 ± 13.636 | 0.976 ± 0.014 | 5 | 570.945 ± 15.468 | 0.980 ± 0.016 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8_retcond_textcond_mttextonly | dkitty | 5 | 267.760 ± 41.787 | 0.940 ± 0.034 | — | — | — | 5 | 270.165 ± 38.116 | 0.942 ± 0.031 | 5 | 296.782 ± 11.719 | 0.964 ± 0.010 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8_retcond_textcond_mttextonly | tfbind8 | 5 | 0.501 ± 0.050 | 0.501 ± 0.050 | — | — | — | 5 | 0.973 ± 0.009 | 0.973 ± 0.009 | 5 | 0.698 ± 0.101 | 0.698 ± 0.101 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8_retcond_textcond_mttextonly | tfbind10 | 5 | 0.228 ± 0.270 | 0.523 ± 0.068 | — | — | — | 5 | 0.626 ± 0.131 | 0.623 ± 0.033 | 5 | 0.215 ± 0.219 | 0.520 ± 0.055 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8_retcond_textcond_mttextonly | gtopx2 | 5 | -105.492 ± 9.077 | 1.000 ± 0.000 | — | — | — | 5 | -99.293 ± 22.175 | 1.000 ± 0.000 | 5 | -99.013 ± 42.811 | 1.000 ± 0.000 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8_retcond_textcond_mttextonly | gtopx3 | 5 | -59.799 ± 31.230 | 1.000 ± 0.000 | — | — | — | 5 | -58.310 ± 14.914 | 1.000 ± 0.000 | 5 | -86.325 ± 61.474 | 1.000 ± 0.000 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8_retcond_textcond_mttextonly | gtopx4 | 5 | -114.646 ± 74.268 | 1.000 ± 0.000 | — | — | — | 5 | -69.035 ± 7.255 | 1.000 ± 0.000 | 5 | -115.660 ± 60.530 | 1.000 ± 0.000 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8_retcond_textcond_mttextonly | gtopx6 | 5 | -95.020 ± 21.885 | 0.999 ± 0.000 | — | — | — | 5 | -55.267 ± 7.939 | 1.000 ± 0.000 | 5 | -82.709 ± 45.592 | 1.000 ± 0.000 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8_textcond_mttextonly | ant | 5 | 568.050 ± 7.329 | 0.977 ± 0.008 | — | — | — | 5 | 566.325 ± 13.636 | 0.976 ± 0.014 | 5 | 570.945 ± 15.468 | 0.980 ± 0.016 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8_textcond_mttextonly | dkitty | 5 | 297.174 ± 23.040 | 0.964 ± 0.019 | — | — | — | 5 | 270.165 ± 38.116 | 0.942 ± 0.031 | 5 | 296.782 ± 11.719 | 0.964 ± 0.010 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8_textcond_mttextonly | tfbind8 | 5 | 0.689 ± 0.257 | 0.689 ± 0.257 | — | — | — | 5 | 0.973 ± 0.009 | 0.973 ± 0.009 | 5 | 0.698 ± 0.101 | 0.698 ± 0.101 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8_textcond_mttextonly | tfbind10 | 5 | 0.356 ± 0.128 | 0.555 ± 0.032 | — | — | — | 5 | 0.626 ± 0.131 | 0.623 ± 0.033 | 5 | 0.215 ± 0.219 | 0.520 ± 0.055 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8_textcond_mttextonly | gtopx2 | 5 | -131.395 ± 25.559 | 1.000 ± 0.000 | — | — | — | 5 | -99.293 ± 22.175 | 1.000 ± 0.000 | 5 | -99.013 ± 42.811 | 1.000 ± 0.000 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8_textcond_mttextonly | gtopx3 | 5 | -83.301 ± 51.265 | 1.000 ± 0.000 | — | — | — | 5 | -58.310 ± 14.914 | 1.000 ± 0.000 | 5 | -86.325 ± 61.474 | 1.000 ± 0.000 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8_textcond_mttextonly | gtopx4 | 5 | -134.278 ± 84.991 | 1.000 ± 0.000 | — | — | — | 5 | -69.035 ± 7.255 | 1.000 ± 0.000 | 5 | -115.660 ± 60.530 | 1.000 ± 0.000 |
| multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8_textcond_mttextonly | gtopx6 | 5 | -53.056 ± 22.553 | 1.000 ± 0.000 | — | — | — | 5 | -55.267 ± 7.939 | 1.000 ± 0.000 | 5 | -82.709 ± 45.592 | 1.000 ± 0.000 |
| multitask_ant_dkitty_retcond_textcond_mttextonly | ant | 5 | 554.884 ± 23.317 | 0.964 ± 0.024 | — | — | — | 5 | 566.325 ± 13.636 | 0.976 ± 0.014 | 5 | 570.945 ± 15.468 | 0.980 ± 0.016 |
| multitask_ant_dkitty_retcond_textcond_mttextonly | dkitty | 5 | 286.541 ± 18.190 | 0.955 ± 0.015 | — | — | — | 5 | 270.165 ± 38.116 | 0.942 ± 0.031 | 5 | 296.782 ± 11.719 | 0.964 ± 0.010 |
| multitask_ant_dkitty_textcond_mttextonly | ant | 6 | 574.791 ± 12.325 | 0.984 ± 0.013 | — | — | — | 5 | 566.325 ± 13.636 | 0.976 ± 0.014 | 5 | 570.945 ± 15.468 | 0.980 ± 0.016 |
| multitask_ant_dkitty_textcond_mttextonly | dkitty | 6 | 288.355 ± 15.088 | 0.957 ± 0.012 | — | — | — | 5 | 270.165 ± 38.116 | 0.942 ± 0.031 | 5 | 296.782 ± 11.719 | 0.964 ± 0.010 |
| multitask_gtopx2_gtopx3_gtopx4_gtopx6_retcond_textcond_mttextonly | gtopx2 | 5 | -119.328 ± 14.301 | 1.000 ± 0.000 | — | — | — | 5 | -99.293 ± 22.175 | 1.000 ± 0.000 | 5 | -99.013 ± 42.811 | 1.000 ± 0.000 |
| multitask_gtopx2_gtopx3_gtopx4_gtopx6_retcond_textcond_mttextonly | gtopx3 | 5 | -56.036 ± 12.318 | 1.000 ± 0.000 | — | — | — | 5 | -58.310 ± 14.914 | 1.000 ± 0.000 | 5 | -86.325 ± 61.474 | 1.000 ± 0.000 |
| multitask_gtopx2_gtopx3_gtopx4_gtopx6_retcond_textcond_mttextonly | gtopx4 | 5 | -67.232 ± 16.355 | 1.000 ± 0.000 | — | — | — | 5 | -69.035 ± 7.255 | 1.000 ± 0.000 | 5 | -115.660 ± 60.530 | 1.000 ± 0.000 |
| multitask_gtopx2_gtopx3_gtopx4_gtopx6_retcond_textcond_mttextonly | gtopx6 | 5 | -52.263 ± 5.259 | 1.000 ± 0.000 | — | — | — | 5 | -55.267 ± 7.939 | 1.000 ± 0.000 | 5 | -82.709 ± 45.592 | 1.000 ± 0.000 |
| multitask_tfbind10_tfbind8_retcond_textcond_mttextonly | tfbind8 | 2 | 0.976 ± 0.000 | 0.976 ± 0.000 | — | — | — | 5 | 0.973 ± 0.009 | 0.973 ± 0.009 | 5 | 0.698 ± 0.101 | 0.698 ± 0.101 |
| multitask_tfbind10_tfbind8_retcond_textcond_mttextonly | tfbind10 | 2 | 0.702 ± 0.090 | 0.642 ± 0.023 | — | — | — | 5 | 0.626 ± 0.131 | 0.623 ± 0.033 | 5 | 0.215 ± 0.219 | 0.520 ± 0.055 |
| multitask_tfbind10_tfbind8_textcond_mttextonly | tfbind8 | 5 | 0.986 ± 0.008 | 0.986 ± 0.008 | — | — | — | 5 | 0.973 ± 0.009 | 0.973 ± 0.009 | 5 | 0.698 ± 0.101 | 0.698 ± 0.101 |
| multitask_tfbind10_tfbind8_textcond_mttextonly | tfbind10 | 5 | 0.722 ± 0.096 | 0.647 ± 0.024 | — | — | — | 5 | 0.626 ± 0.131 | 0.623 ± 0.033 | 5 | 0.215 ± 0.219 | 0.520 ± 0.055 |
| tfbind10_multiple_runs_retcond | tfbind10 | 5 | 0.580 ± 0.079 | 0.612 ± 0.020 | 5 | 0.577 ± 0.049 | 0.611 ± 0.012 | 5 | 0.626 ± 0.131 | 0.623 ± 0.033 | 5 | 0.215 ± 0.219 | 0.520 ± 0.055 |
| tfbind8_multiple_runs_retcond | tfbind8 | 4 | 0.973 ± 0.010 | 0.973 ± 0.010 | 5 | 0.905 ± 0.065 | 0.905 ± 0.065 | 5 | 0.973 ± 0.009 | 0.973 ± 0.009 | 5 | 0.698 ± 0.101 | 0.698 ± 0.101 |

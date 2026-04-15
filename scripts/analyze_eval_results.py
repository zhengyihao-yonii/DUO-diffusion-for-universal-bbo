#!/usr/bin/env python3
"""
Aggregate evaluate.log metrics across runs (run*_seed* / run*) per experiment,
then compare GTGdfgo vs GTG results in one report (CSV + LaTeX table).

Metrics: max_ep_reward -> max, nmax_ep_reward -> nmax (mean ± std over runs).

DFGO 列在每种模式（单任务 / 小组 multi / 全任务 multi / text）下，对
``single_task|multi_task|text_conditioned_only/<…>/<hyper>/run*`` 中**不同超参目录**
（如 ``NxH_k*_eps*``、``w*_…``）分别聚合后，**按该任务取 max 均值最高**的一组作为该列结果。

Per-task columns ``gtgdfgo_msub_*`` / ``gtgdfgo_mfull_*`` look up multitask runs by task:
subgroup (ant+dkitty, tfbind8+10, four gtopx) vs full 8-task multitask — see
``TASK_TO_SUBGROUP_MULTITASK_EXP`` and ``FULL_MULTITASK_EXP``.

所有汇总表与 UniSO 输入均位于 ``results/analysis_table/``：宽表 ``max_short.*``、矩阵 ``max_extended.*``、``text_conditioned_result_analysis.*``、``nmax.tex``，以及 ``uniso_result.tex``、``uniso_nresult.tex``、``d_best.json``（可选）。
``text_conditioned_result_analysis``（``--mode final``）：``text_conditioned_only/all_frac1.0_sigma0.0/``（可用 ``EVAL_ALL_TASK_FRAC_SIG`` 覆盖）下**每个超参子目录一列** DFGO。默认一次生成全部；也可用 ``--mode short|full|final``（见 ``run_analyze_eval.sh``）。
矩阵列为 **UniSO-T**（``uniso_result.tex`` 中 **UniSO-T Improved** 列）+ 12 列方法；文本多任务目录后缀见 ``MATRIX12_TEXT_SUFFIXES``。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import rankdata

# CSI: color (…m) and cursor / erase (…F, …J, …), etc.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Task names are alphanumeric/underscore (dkitty, ant, gtopx2); avoid matching
# progress bars like "[########" or broken escapes after partial strip.
BRACKET_MAX = re.compile(
    r"\[([a-zA-Z0-9_]+)\]\s+max_ep_reward:\s*([-\d.eE+]+(?:e[-+]?\d+)?|nan|inf)"
)
BRACKET_NMAX = re.compile(
    r"\[([a-zA-Z0-9_]+)\]\s+nmax_ep_reward:\s*([-\d.eE+]+(?:e[-+]?\d+)?|nan|inf)"
)
# (?<!n) avoids matching the suffix "max_ep_reward" inside "nmax_ep_reward"
PLAIN_MAX = re.compile(
    r"(?<!n)max_ep_reward:\s*([-\d.eE+]+(?:e[-+]?\d+)?|nan|inf)"
)
PLAIN_NMAX = re.compile(
    r"nmax_ep_reward:\s*([-\d.eE+]+(?:e[-+]?\d+)?|nan|inf)"
)

# One line in evaluate.log: Task name: gtopx2 optima: 1 Dataset min/max: a/b
TASK_DATASET_LINE = re.compile(
    r"Task name:\s*(\S+)[^\n]*Dataset min/max:\s*([-\d.eE+]+)/([-\d.eE+]+)"
)
# evaluate.py 打印：与训练子集一致的离线最优 y（优先用于 D(best) 列）
OFFLINE_TRAIN_BEST_LINE = re.compile(
    r"\[([a-zA-Z0-9_]+)\]\s+offline_train_best_y:\s*([-\d.eE+]+(?:e[-+]?\d+)?|nan|inf)"
)


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _safe_float(s: str) -> float:
    x = float(s)
    if np.isnan(x):
        return float("nan")
    return x


def parse_multitask_table(text: str) -> dict[str, tuple[float, float]]:
    """Parse lines under '多任务评估汇总' with format: task max median mean | nmax ..."""
    if "多任务评估汇总" not in text:
        return {}
    chunk = text[text.rfind("多任务评估汇总") :]
    out: dict[str, tuple[float, float]] = {}
    for line in chunk.split("\n"):
        line = line.strip()
        if "|" not in line or line.startswith("-"):
            continue
        parts = line.split("|")
        if len(parts) != 2:
            continue
        left = parts[0].split()
        right = parts[1].split()
        if len(left) < 4 or len(right) < 3:
            continue
        task = left[0]
        if task.lower() in ("task",):
            continue
        try:
            max_v = _safe_float(left[1])
            nmax_v = _safe_float(right[0])
        except ValueError:
            continue
        out[task] = (max_v, nmax_v)
    return out


def infer_task_from_experiment_name(exp_name: str) -> str:
    """
    从 results 子目录名还原单任务 task key（用于无 ``[task]`` 前缀的 plain max/nmax 行）。

    GTG/GTGdfgo 的 ``{task}_multiple_runs_retcond`` 若以 ``_multiple_runs$`` 正则匹配会失败，
    原先会把整段目录名当成 task，导致矩阵里 GTG ST+r 等列永远对不上 ``ant`` 等键。
    """
    if exp_name.endswith("_multiple_runs_retcond"):
        return exp_name[: -len("_multiple_runs_retcond")]
    m = re.match(r"^(.+)_multiple_runs$", exp_name)
    if m:
        return m.group(1)
    return exp_name


def parse_evaluate_log(path: Path) -> dict[str, tuple[float, float]]:
    """Return mapping task_name -> (max, nmax). Multiple tasks for multitask eval."""
    text = strip_ansi(path.read_text(encoding="utf-8", errors="replace"))

    max_by_task: dict[str, float] = {}
    nmax_by_task: dict[str, float] = {}
    for m in BRACKET_MAX.finditer(text):
        max_by_task[m.group(1)] = _safe_float(m.group(2))
    for m in BRACKET_NMAX.finditer(text):
        nmax_by_task[m.group(1)] = _safe_float(m.group(2))

    tasks = set(max_by_task) | set(nmax_by_task)
    out: dict[str, tuple[float, float]] = {}
    for t in tasks:
        if t in max_by_task and t in nmax_by_task:
            out[t] = (max_by_task[t], nmax_by_task[t])

    if out:
        return out

    tab = parse_multitask_table(text)
    if tab:
        return tab

    pm = list(PLAIN_MAX.finditer(text))
    pn = list(PLAIN_NMAX.finditer(text))
    if pm and pn:
        exp_name = path.parent.parent.name
        task_guess = infer_task_from_experiment_name(exp_name)
        return {
            task_guess: (_safe_float(pm[-1].group(1)), _safe_float(pn[-1].group(1)))
        }

    return {}


def mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    m = float(np.mean(a))
    if a.size == 1:
        return m, 0.0
    s = float(np.std(a, ddof=1))
    return m, s


# 汇总表行顺序：单任务按 ant → dkitty → superconductor → tfbind8 → tfbind10 → gtopx2…6；
# 多任务按「联立任务数」从少到多（2 任务：ant+dkitty 先于 tfbind8+10；4：四 gtopx；8：全任务）。
TASK_ORDER: list[str] = [
    "ant",
    "dkitty",
    "superconductor",
    "tfbind8",
    "tfbind10",
    "gtopx2",
    "gtopx3",
    "gtopx4",
    "gtopx6",
]

# Design-Bench 五任务（与 ``nmax.tex`` / 归一化表一致，不含 GTOP-X）
DESIGN_BENCH_TASK_ORDER: list[str] = [
    "ant",
    "dkitty",
    "superconductor",
    "tfbind8",
    "tfbind10",
]

EXPERIMENT_ORDER: list[str] = [
    "ant_multiple_runs",
    "dkitty_multiple_runs",
    "superconductor_multiple_runs",
    "tfbind8_multiple_runs",
    "tfbind10_multiple_runs",
    "gtopx2_multiple_runs",
    "gtopx3_multiple_runs",
    "gtopx4_multiple_runs",
    "gtopx6_multiple_runs",
    "multitask_ant_dkitty",
    "multitask_tfbind10_tfbind8",
    "multitask_gtopx2_gtopx3_gtopx4_gtopx6",
    "multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_tfbind10_tfbind8",
    "multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_superconductor_tfbind10_tfbind8",
]

# 全任务一起训练的 multitask 实验名（9 任务，含 superconductor；与 run_multitask.sh _FULL_MT_TASKS_CSV 一致）
FULL_MULTITASK_EXP: str = (
    "multitask_ant_dkitty_gtopx2_gtopx3_gtopx4_gtopx6_superconductor_tfbind10_tfbind8"
)

# 各 task 对应的「小组」multitask（与 FULL_MULTITASK_EXP 区分）
TASK_TO_SUBGROUP_MULTITASK_EXP: dict[str, str] = {
    "ant": "multitask_ant_dkitty",
    "dkitty": "multitask_ant_dkitty",
    "tfbind8": "multitask_tfbind10_tfbind8",
    "tfbind10": "multitask_tfbind10_tfbind8",
    "gtopx2": "multitask_gtopx2_gtopx3_gtopx4_gtopx6",
    "gtopx3": "multitask_gtopx2_gtopx3_gtopx4_gtopx6",
    "gtopx4": "multitask_gtopx2_gtopx3_gtopx4_gtopx6",
    "gtopx6": "multitask_gtopx2_gtopx3_gtopx4_gtopx6",
}


def _experiment_rank(name: str) -> tuple[int, str]:
    try:
        return (EXPERIMENT_ORDER.index(name), name)
    except ValueError:
        return (10_000, name)


def _task_rank(name: str) -> tuple[int, str]:
    try:
        return (TASK_ORDER.index(name), name)
    except ValueError:
        return (10_000, name)


def _ordered_experiment_names(names: set[str]) -> list[str]:
    head = [e for e in EXPERIMENT_ORDER if e in names]
    tail = sorted(names - set(head))
    return head + tail


def _ordered_task_names(names: set[str]) -> list[str]:
    head = [t for t in TASK_ORDER if t in names]
    tail = sorted(names - set(head))
    return head + tail


def sort_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (_experiment_rank(r["experiment"]), _task_rank(r["task"])),
    )


_ALL_RUN_DIR_RE = re.compile(r"^run[0-9]+_seed[0-9]+$")

TEXT_CONDITIONED_ROOT = "text_conditioned_only"
MULTI_TASK_ROOT = "multi_task"
SINGLE_TASK_ROOT = "single_task"
# 三者并列于 results/；同列聚合键为 ``<root>/<tasks>_frac_sigma``（不含超参子目录名）
AGGREGATE_EXPERIMENT_ROOTS: tuple[str, ...] = (
    TEXT_CONDITIONED_ROOT,
    MULTI_TASK_ROOT,
    SINGLE_TASK_ROOT,
)
# text_conditioned_result_analysis：与全 9 任务（含 superconductor）textcond 目录 ``text_conditioned_only/all_frac*_sigma*/`` 或
# ``all_improved_frac*_sigma*/``（USE_TRAJ_PARAMS_JSON=1 每任务最优轨迹）对齐。
EVAL_ALL_TASK_FRAC_SIG: str = os.environ.get(
    "EVAL_ALL_TASK_FRAC_SIG", "all_frac1.0_sigma0.0"
)
# text_conditioned_result_analysis：每列对应 ``text_conditioned_only/<EVAL_ALL_TASK_FRAC_SIG>/<hyper>/``（hyper 为关键参数目录名）
EVAL_ALL_EXPERIMENT_PREFIX: str = (
    f"{TEXT_CONDITIONED_ROOT}/{EVAL_ALL_TASK_FRAC_SIG}"
)

# GTGdfgo results 布局：``single_task/<task>_frac*_sigma*``、``multi_task/<token>_frac*_sigma*`` 等
GTGDFGO_TASK_FRAC_SIG: str = os.environ.get("GTGDFGO_TASK_FRAC_SIG", "frac1.0_sigma0.0")

# 旧 ``multitask_*`` 名 -> 新 ``multi_task`` / ``text_conditioned_only`` 下目录 token（不含 frac_sigma）
MULTITASK_NAME_TO_DIR: dict[str, str] = {
    "multitask_ant_dkitty": "ant_dkitty",
    "multitask_tfbind10_tfbind8": "tfbind10_tfbind8",
    "multitask_gtopx2_gtopx3_gtopx4_gtopx6": "gtopx2_gtopx3_gtopx4_gtopx6",
}


def gtgdfgo_single_task_prefix(task: str) -> str:
    return f"{SINGLE_TASK_ROOT}/{task}_{GTGDFGO_TASK_FRAC_SIG}"


def gtgdfgo_subgroup_multitask_prefix(task: str) -> str | None:
    old = TASK_TO_SUBGROUP_MULTITASK_EXP.get(task)
    if not old:
        return None
    token = MULTITASK_NAME_TO_DIR.get(old)
    if not token:
        return None
    return f"{MULTI_TASK_ROOT}/{token}_{GTGDFGO_TASK_FRAC_SIG}"


def gtgdfgo_full_multitask_prefix() -> str:
    return f"{MULTI_TASK_ROOT}/all_{GTGDFGO_TASK_FRAC_SIG}"


def gtgdfgo_subgroup_text_prefix(task: str) -> str | None:
    """``text_conditioned_only/<token>_<frac_sigma>/`` 下各 ``<hyper>/``（与 ``_ret`` 后缀区分 returns 列）。"""
    old = TASK_TO_SUBGROUP_MULTITASK_EXP.get(task)
    if not old:
        return None
    token = MULTITASK_NAME_TO_DIR.get(old)
    if not token:
        return None
    return f"{TEXT_CONDITIONED_ROOT}/{token}_{GTGDFGO_TASK_FRAC_SIG}"


def gtgdfgo_all_text_prefix() -> str:
    return f"{TEXT_CONDITIONED_ROOT}/all_{GTGDFGO_TASK_FRAC_SIG}"


def _hyper_matches_returns(hyper: str, use_returns: bool) -> bool:
    return hyper.endswith("_ret") == use_returns


def best_gtgdfgo_exp_for_task(
    bucket: dict[str, dict[str, dict[str, Any]]],
    prefix: str,
    task: str,
    use_returns: bool,
) -> str | None:
    """
    在 ``prefix`` 或 ``prefix/<hyper>/…`` 下，按该任务 ``max_mean``（max_ep_reward）最大选取实验目录。
    ``hyper`` 名以 ``_ret`` 结尾表示开启 returns，与无 ``_ret`` 的列分开比较。
    """
    best_k: str | None = None
    best_m: float | None = None
    if prefix in bucket:
        st = bucket[prefix].get(task)
        if st:
            m = st.get("max_mean")
            if m is not None and np.isfinite(float(m)):
                best_m = float(m)
                best_k = prefix
    for k in bucket:
        if not k.startswith(prefix + "/"):
            continue
        rest = k[len(prefix) + 1 :]
        hyper = rest.split("/")[0]
        if not _hyper_matches_returns(hyper, use_returns):
            continue
        st = bucket[k].get(task)
        if not st:
            continue
        m = st.get("max_mean")
        if m is None or not np.isfinite(float(m)):
            continue
        mf = float(m)
        if best_m is None or mf > best_m:
            best_m = mf
            best_k = k
    return best_k


def _experiment_name_from_eval_log(
    results_root: Path, evaluate_log: Path
) -> str | None:
    """
    - ``<exp>/<run>/evaluate.log`` -> ``<exp>``（普通实验）
    - ``text_conditioned_only/<EVAL_ALL_TASK_FRAC_SIG>/<hyper>/run*/evaluate.log``
      -> ``text_conditioned_only/<EVAL_ALL_TASK_FRAC_SIG>/<hyper>``（text_conditioned_result_analysis 每列一 hyper）
    - 其它 ``<agg_root>/<tasks_frac_sigma>/<hyper>/run*/evaluate.log``
      -> ``<agg_root>/<tasks_frac_sigma>/<hyper>``（每超参子目录单独键）
    - 旧布局 ``text_conditioned_only/multi_task/<tasks_frac_sigma>/<hyper>/run*``（仍兼容至迁移完成）
      -> ``text_conditioned_only/<tasks_frac_sigma>``
    """
    try:
        rel = evaluate_log.relative_to(results_root.resolve())
    except ValueError:
        return evaluate_log.parent.parent.name
    parts = rel.parts
    if parts[-1] != "evaluate.log":
        return evaluate_log.parent.parent.name
    if not _ALL_RUN_DIR_RE.match(parts[-2]):
        return evaluate_log.parent.parent.name
    # 旧：text_conditioned_only/multi_task/<tasks_frac>/<hyper>/run/...
    if (
        len(parts) >= 6
        and parts[0] == TEXT_CONDITIONED_ROOT
        and parts[1] == MULTI_TASK_ROOT
    ):
        return f"{parts[0]}/{parts[2]}"
    # all_frac*：每个超参子目录单独一列（与 text_conditioned_result_analysis 一致）
    if (
        len(parts) >= 5
        and parts[0] == TEXT_CONDITIONED_ROOT
        and parts[1] == EVAL_ALL_TASK_FRAC_SIG
    ):
        return f"{parts[0]}/{parts[1]}/{parts[2]}"
    # 其它：text_conditioned_only|multi_task|single_task/<tasks_frac>/<hyper>/run/...
    # 含 ``<hyper>``（如 ``4000x64_k20_eps0.05``、``w1.2_1000x64_k20_eps0.05``）时键为 ``base/hyper``，
    # 以便按任务在多种 k/eps/权重 下取最优；无 hyper 层时为 ``base``。
    if len(parts) >= 5 and parts[0] in AGGREGATE_EXPERIMENT_ROOTS:
        base = f"{parts[0]}/{parts[1]}"
        hyper = parts[-3]
        return f"{base}/{hyper}"
    return evaluate_log.parent.parent.name


def scan_results_root(results_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Returns:
      experiment_name -> task -> {
        'n_runs': int,
        'max_mean', 'max_std', 'nmax_mean', 'nmax_std',
        'runs': [ { 'run', 'max', 'nmax' }, ... ]
    """
    agg: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    if not results_root.is_dir():
        return {}

    logs: list[Path] = []
    logs.extend(sorted(results_root.glob("*/*/evaluate.log")))
    logs.extend(sorted(results_root.glob("*/*/*/evaluate.log")))
    logs.extend(sorted(results_root.glob("*/*/*/*/evaluate.log")))
    logs.extend(sorted(results_root.glob("*/*/*/*/*/evaluate.log")))
    seen: set[str] = set()
    for p in logs:
        s = str(p.resolve())
        if s in seen:
            continue
        seen.add(s)
        evaluate_log = p
        exp = _experiment_name_from_eval_log(results_root, evaluate_log)
        if exp is None:
            continue
        run_name = evaluate_log.parent.name
        metrics = parse_evaluate_log(evaluate_log)
        if not metrics:
            continue
        for task, (mx, nm) in metrics.items():
            agg[exp][task].append(
                {"run": run_name, "max": mx, "nmax": nm, "log": str(evaluate_log)}
            )

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for exp, tasks in agg.items():
        out[exp] = {}
        for task, runs in tasks.items():
            maxs = [r["max"] for r in runs]
            nmaxs = [r["nmax"] for r in runs]
            mm, ms = mean_std(maxs)
            nm, ns = mean_std(nmaxs)
            out[exp][task] = {
                "n_runs": len(runs),
                "max_mean": mm,
                "max_std": ms,
                "nmax_mean": nm,
                "nmax_std": ns,
                "runs": runs,
            }
    return out


def build_comparison_rows(
    gtgdfgo: dict[str, dict[str, dict[str, Any]]],
    gtg: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    exp_names = _ordered_experiment_names(set(gtgdfgo.keys()) | set(gtg.keys()))
    rows: list[dict[str, Any]] = []
    for exp in exp_names:
        tasks_g = gtgdfgo.get(exp, {})
        tasks_c = gtg.get(exp, {})
        task_names = _ordered_task_names(set(tasks_g.keys()) | set(tasks_c.keys()))
        for task in task_names:
            row: dict[str, Any] = {
                "experiment": exp,
                "task": task,
            }
            for prefix, src in (("gtgdfgo", tasks_g), ("gtg", tasks_c)):
                d = src.get(task)
                if d is None:
                    row[f"{prefix}_n_runs"] = ""
                    row[f"{prefix}_max_mean"] = ""
                    row[f"{prefix}_max_std"] = ""
                    row[f"{prefix}_nmax_mean"] = ""
                    row[f"{prefix}_nmax_std"] = ""
                else:
                    row[f"{prefix}_n_runs"] = d["n_runs"]
                    row[f"{prefix}_max_mean"] = d["max_mean"]
                    row[f"{prefix}_max_std"] = d["max_std"]
                    row[f"{prefix}_nmax_mean"] = d["nmax_mean"]
                    row[f"{prefix}_nmax_std"] = d["nmax_std"]
            rows.append(row)
    return rows


def _lookup_task_stats(
    bucket: dict[str, dict[str, dict[str, Any]]],
    exp_name: str | None,
    task: str,
) -> dict[str, Any] | None:
    if not exp_name or exp_name not in bucket:
        return None
    return bucket[exp_name].get(task)


def _prefix_stats_flat(prefix: str, st: dict[str, Any] | None) -> dict[str, Any]:
    keys = (
        f"{prefix}_n_runs",
        f"{prefix}_max_mean",
        f"{prefix}_max_std",
        f"{prefix}_nmax_mean",
        f"{prefix}_nmax_std",
    )
    if st is None:
        return {k: "" for k in keys}
    return {
        f"{prefix}_n_runs": st["n_runs"],
        f"{prefix}_max_mean": st["max_mean"],
        f"{prefix}_max_std": st["max_std"],
        f"{prefix}_nmax_mean": st["nmax_mean"],
        f"{prefix}_nmax_std": st["nmax_std"],
    }


def enrich_multitask_columns(
    rows: list[dict[str, Any]],
    gtgdfgo: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    按当前行的 task，从「小组 multitask」与「全任务 multitask」实验中查找同 task 的聚合指标，
    写入 gtgdfgo_msub_* / gtgdfgo_mfull_*（与该行 experiment 无关，便于横向对比）。
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        task = row["task"]
        sp = gtgdfgo_subgroup_multitask_prefix(task)
        sub_k = (
            best_gtgdfgo_exp_for_task(gtgdfgo, sp, task, False) if sp else None
        )
        sub_st = _lookup_task_stats(gtgdfgo, sub_k, task)
        fp = gtgdfgo_full_multitask_prefix()
        full_k = best_gtgdfgo_exp_for_task(gtgdfgo, fp, task, False)
        full_st = _lookup_task_stats(gtgdfgo, full_k, task)
        new_row = dict(row)
        new_row.update(_prefix_stats_flat("gtgdfgo_msub", sub_st))
        new_row.update(_prefix_stats_flat("gtgdfgo_mfull", full_st))
        out.append(new_row)
    return out


DECIMALS = 3

# (task_key, LaTeX row label, single-task experiment directory under results/)
LATEX_TASK_ROWS: list[tuple[str, str, str]] = [
    ("ant", "Ant", "ant_multiple_runs"),
    ("dkitty", "D'Kitty", "dkitty_multiple_runs"),
    ("superconductor", "Superconductor", "superconductor_multiple_runs"),
    ("tfbind8", "TF Bind 8", "tfbind8_multiple_runs"),
    ("tfbind10", "TF Bind 10", "tfbind10_multiple_runs"),
    ("gtopx2", "GTOPX 2", "gtopx2_multiple_runs"),
    ("gtopx3", "GTOPX 3", "gtopx3_multiple_runs"),
    ("gtopx4", "GTOPX 4", "gtopx4_multiple_runs"),
    ("gtopx6", "GTOPX 6", "gtopx6_multiple_runs"),
]

_LATEX_NAME_BY_TASK: dict[str, str] = {t: name for t, name, _ in LATEX_TASK_ROWS}


def latex_task_display_name(task_key: str) -> str:
    """与主 LaTeX 表一致的任务列显示名。"""
    return _LATEX_NAME_BY_TASK.get(task_key, task_key)

def parse_dataset_best_raw(text: str, task_key: str) -> float | None:
    """Fallback: DesignBenchFunctionWrapper 打印的全库 y 范围；max(min,max) 通常等于全库最优 y。"""
    for m in TASK_DATASET_LINE.finditer(text):
        if m.group(1) != task_key:
            continue
        lo, hi = _safe_float(m.group(2)), _safe_float(m.group(3))
        return max(lo, hi)
    return None


def parse_offline_train_best_y(text: str, task_key: str) -> float | None:
    """与训练数据子集一致的最优 y（新 evaluate 会打印）。"""
    last: float | None = None
    for m in OFFLINE_TRAIN_BEST_LINE.finditer(text):
        if m.group(1) != task_key:
            continue
        last = _safe_float(m.group(2))
    return last


def _read_dataset_best_under_dir(exp_dir: Path, task_key: str) -> float | None:
    """在单个实验根目录下（可含多层 hyper/run）从 evaluate.log 解析 D(best)。"""
    if not exp_dir.is_dir():
        return None
    for evaluate_log in sorted(exp_dir.rglob("evaluate.log")):
        if not _ALL_RUN_DIR_RE.match(evaluate_log.parent.name):
            continue
        text = strip_ansi(
            evaluate_log.read_text(encoding="utf-8", errors="replace")
        )
        v = parse_offline_train_best_y(text, task_key)
        if v is not None:
            return v
    for evaluate_log in sorted(exp_dir.rglob("evaluate.log")):
        if not _ALL_RUN_DIR_RE.match(evaluate_log.parent.name):
            continue
        text = strip_ansi(
            evaluate_log.read_text(encoding="utf-8", errors="replace")
        )
        v = parse_dataset_best_raw(text, task_key)
        if v is not None:
            return v
    return None


def read_dataset_best_from_experiment(
    results_root: Path, task_key: str
) -> float | None:
    """从 ``results/single_task/<task>_frac*_sigma*/`` 下任意 ``run*_seed*/evaluate.log`` 解析 D(best)。"""
    st_root = results_root / SINGLE_TASK_ROOT
    if not st_root.is_dir():
        return None
    for base in sorted(p for p in st_root.glob(f"{task_key}_frac*_sigma*") if p.is_dir()):
        v = _read_dataset_best_under_dir(base, task_key)
        if v is not None:
            return v
    return None


def fmt_pm_latex(m: Any, s: Any) -> str:
    """Un-normalized max: mean $\\pm$ std for LaTeX math mode."""
    if m == "" or m is None:
        return "--"
    try:
        mf, sf = float(m), float(s)
    except (TypeError, ValueError):
        return "--"
    if np.isnan(mf):
        return "nan"
    if sf and not np.isnan(sf):
        return rf"{mf:.{DECIMALS}f} $\pm$ {sf:.{DECIMALS}f}"
    return rf"{mf:.{DECIMALS}f} $\pm$ {0:.{DECIMALS}f}"


def _parse_mean_from_latex_cell(cell: str) -> float | None:
    """从 ``452.330 $\\pm$ 61.502`` 或 ``165.326`` 中取用于比较大小的 mean（第一个数）。"""
    s = cell.strip()
    if not s or s in ("--", "—", "nan"):
        return None
    m = re.match(
        r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
        s,
    )
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    if np.isnan(v):
        return None
    return v


def _strip_uniso_best_second(cell: str) -> str:
    c = cell.strip()
    m = re.match(r"^\\(?:best|second)\{(.+)\}\s*$", c)
    if m:
        return m.group(1).strip()
    return c


def parse_mean_from_text_cell(cell: str) -> float | None:
    """解析 UniSO / 表格单元中的 mean（支持 ``±`` / ``$\\pm$``、``\\best{}``）。"""
    s = _strip_uniso_best_second(cell)
    s = s.replace(r"$\pm$", " ").replace("±", " ").strip()
    return _parse_mean_from_latex_cell(s)


# uniso_result.tex 首列任务名 -> 本脚本 task_key
UNISO_ROW_DISPLAY_TO_KEY: dict[str, str] = {
    "Ant": "ant",
    "D'Kitty": "dkitty",
    "Superconductor": "superconductor",
    "TF Bind 8": "tfbind8",
    "TF Bind 10": "tfbind10",
    "GTOPX 2": "gtopx2",
    "GTOPX 3": "gtopx3",
    "GTOPX 4": "gtopx4",
    "GTOPX 6": "gtopx6",
}


def parse_uniso_best_per_task(tex_path: Path) -> dict[str, str]:
    """
    从 ``uniso_result.tex`` 中取每个任务在 **UniSO-T → Improved** 列（表头「Vanilla | Improved」下第二列），
    即 **UniSO-T Improved** 的分数；返回 task_key -> LaTeX 单元格正文（已去掉 \\best/\\second 外壳，± 写成 `` $\\pm$ ``）。
    """
    if not tex_path.is_file():
        return {}
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if "&" not in line or "\\\\" not in line:
            continue
        if "Avg. Rank" in line:
            continue
        parts = [p.strip() for p in line.split("&")]
        if len(parts) < 6:
            continue
        task_display = parts[0].strip()
        if task_display not in UNISO_ROW_DISPLAY_TO_KEY:
            continue
        tk = UNISO_ROW_DISPLAY_TO_KEY[task_display]
        # Task | D(best) | BN+BO | BN+Grad | UniSO-T Vanilla | UniSO-T Improved | ...
        cell = parts[5]
        v = parse_mean_from_text_cell(cell)
        if v is None:
            continue
        body = _strip_uniso_best_second(cell)
        body = body.replace("±", r" $\pm$ ").strip()
        body = re.sub(r"\s+", " ", body)
        out[tk] = body
    return out


def parse_uniso_nresult_rows(
    tex_path: Path,
) -> list[tuple[str, str, list[str], str]]:
    """
    解析 ``uniso_nresult.tex``（Design-Bench **归一化**分数表）：每行
    ``(method, venue, [Ant, D'Kitty, Superconductor, TF-Bind-8, TF-Bind-10], Avg.\\ Rank)``。
    """
    if not tex_path.is_file():
        return []
    out: list[tuple[str, str, list[str], str]] = []
    for raw in tex_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if "%" in line:
            line = line.split("%")[0].strip()
        if not line or "&" not in line:
            continue
        if "\\toprule" in line or "\\bottomrule" in line:
            continue
        if "\\midrule" in line and line.count("&") < 3:
            continue
        if "\\\\" in line:
            line = line.split("\\\\")[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("&")]
        if len(parts) < 8:
            continue
        method = parts[0]
        if method.strip() == "Method":
            continue
        venue = parts[1]
        tasks = [parts[i] for i in range(2, 7)]
        avg_rank = parts[7]
        out.append((method, venue, tasks, avg_rank))
    return out


def _nresult_plain_method_name(method: str) -> str:
    return re.sub(r"\\textbf\{([^}]*)\}", r"\1", method.strip()).strip()


def _nmax_baseline_method_cell(method: str) -> str:
    """``nmax`` 表中 UniSO-T 与其它基线一致，方法名不加 ``\\textbf``。"""
    if _nresult_plain_method_name(method).startswith("UniSO-T"):
        return "UniSO-T"
    return method


def _nresult_midrule_before_dfgo(last_baseline_method: str) -> bool:
    """最后一行基线为 UniSO-T 时，其与 DFGO 之间需 ``\\midrule``（RaM 行后已有 ``\\midrule``，不再重复）。"""
    return _nresult_plain_method_name(last_baseline_method).startswith("UniSO-T")


def _nresult_midrule_after_row(method: str) -> bool:
    """与 ``uniso_nresult.tex`` 中 ``\\midrule`` 分段一致（在对应方法行之后插入）。"""
    s = method.strip()
    plain = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s).strip()
    if r"$\mathcal{D}$(best)" in s:
        return True
    if plain.startswith("Grad. Ascent Min"):
        return True
    if plain == "GTG":
        return True
    # MATCH-OPT 与 RaM 同属一段，仅 RaM 行末有 ``\\midrule``（见源表）
    if plain.startswith("RaM-ListNet"):
        return True
    return False


def column_mean_rank_stats(means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    means[i,j] = 任务 i、方法 j 的 mean；nan 表示该格无数据。
    每行内**仅对非 nan 的方法**按 **越大越好** 赋秩（并列取平均秩）；全空列在该行不参与比较。
    若某列在所有任务上均未获得秩，则 Mean rank 对应格为 nan（由 ``fmt_mean_pm_rank`` 显示为 ``--``）。
    """
    n_tasks, n_cols = means.shape
    ranks = np.full((n_tasks, n_cols), np.nan, dtype=np.float64)
    for i in range(n_tasks):
        row = means[i]
        mask = ~np.isnan(row)
        if int(mask.sum()) < 2:
            continue
        sub = row[mask]
        idx = np.where(mask)[0]
        r = rankdata(-sub, method="average")
        ranks[i, idx] = r
    mu = np.empty(n_cols, dtype=np.float64)
    sd = np.empty(n_cols, dtype=np.float64)
    for j in range(n_cols):
        col = ranks[:, j]
        fin = col[np.isfinite(col)]
        if fin.size == 0:
            mu[j] = np.nan
            sd[j] = np.nan
        else:
            mu[j] = float(np.mean(fin))
            sd[j] = float(np.std(fin, ddof=1)) if fin.size > 1 else 0.0
    return mu, sd


def fmt_mean_pm_rank(mu: float, sd: float) -> str:
    if mu is None or not np.isfinite(mu):
        return "--"
    if sd is None or (not np.isfinite(sd)):
        sd = 0.0
    return rf"{mu:.3f} $\pm$ {sd:.3f}"


def _mean_from_task_stats(st: dict[str, Any] | None) -> float | None:
    if st is None:
        return None
    m = st.get("max_mean")
    if m == "" or m is None:
        return None
    try:
        v = float(m)
    except (TypeError, ValueError):
        return None
    if np.isnan(v):
        return None
    return v


def _brace_content_after(s: str, open_i: int) -> str | None:
    """``s[open_i]`` 为 ``{``，返回与之匹配的 ``}`` 内层文本（不含最外层花括号）。"""
    if open_i >= len(s) or s[open_i] != "{":
        return None
    depth = 0
    start = open_i
    for j in range(open_i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1 : j]
    return None


def latex_cell_strip_wrappers(cell: str) -> str:
    """
    去掉单元格外层 ``\\textbf{}`` / ``\\textcolor{…}{}``，得到用于展示与比较大小的正文
    （与 ``uniso_nresult`` 源表着色无关；供 ``nmax`` 按列重算蓝/紫）。
    """
    s = cell.strip()
    for _ in range(40):
        t = _unwrap_one_texbf_or_textcolor(s)
        if t is None:
            return s
        s = t.strip()
    return s


def _unwrap_one_texbf_or_textcolor(s: str) -> str | None:
    if s.startswith(r"\textbf{"):
        inner = _brace_content_after(s, len(r"\textbf"))
        return inner
    m = re.match(r"^\\textcolor\{[^}]+\}", s)
    if m:
        pos = m.end()
        if pos < len(s) and s[pos] == "{":
            return _brace_content_after(s, pos)
    return None


def _nmax_parse_task_value(plain: str) -> float | None:
    """任务列：越大越好；``--`` / 无法解析则不参与该列着色。"""
    s = plain.strip()
    if not s or s in ("--", "—", "/", "nan"):
        return None
    return parse_mean_from_text_cell(s)


def nmax_avg_rank_denominator_total(n_base: int, parsed: list[tuple[str, str, list[str], str]]) -> int:
    """
    源表 ``/22`` 表示参与排名的方法数（不含 ``$\\mathcal{D}$(best)`` 行）。
    ``nmax`` 在相同方法集合上增加 DFGO single、multi 两行，故分母 = ``(n_base - 1) + 2``
    （首行为 D(best) 时）；否则 ``n_base + 2``。
    """
    if parsed and r"$\mathcal{D}$(best)" in parsed[0][0]:
        return (n_base - 1) + 2
    return n_base + 2


def nmax_fmt_dfgo_avg_rank(mu: float, n_total: int) -> str:
    """DFGO 两行 Avg.\\ Rank：``mean / N``（与基线 ``a / N`` 同一分母 ``N``）。"""
    if mu is None or not np.isfinite(float(mu)):
        return "--"
    return f"{float(mu):.3f} / {n_total}"


def nmax_fix_avg_rank_denominator(cell: str, n_total: int) -> str:
    """
    源表 ``uniso_nresult`` 的 Avg.\\ Rank 形如 ``a / 22``；本表分母改为 ``n_total``（见
    ``nmax_avg_rank_denominator_total``）。``/`` 等不修改。
    """
    s = cell.strip()
    if not s or s in ("/", "--", "—"):
        return cell
    m = re.match(r"^(.+?)\s*/\s*(\d+)\s*$", s)
    if m:
        return f"{m.group(1).strip()} / {n_total}"
    return cell


def _nmax_parse_avg_rank_value(plain: str) -> float | None:
    """Avg.\\ Rank：越小越好；``/``、``--`` 跳过；``a / b`` 取第一个数（含 DFGO ``m / N``）。"""
    s = plain.strip()
    if not s or s in ("--", "—", "nan"):
        return None
    if s == "/":
        return None
    return parse_mean_from_text_cell(s)


def _nmax_unique_sorted_values(
    values: list[float], *, reverse: bool
) -> list[float]:
    """去重（容差），排序后用于取第一、二档。"""
    if not values:
        return []
    ordered = sorted(values, reverse=reverse)
    out: list[float] = []
    for v in ordered:
        if not out or not np.isclose(v, out[-1], rtol=1e-5, atol=1e-8):
            out.append(v)
    return out


def nmax_colorize_column_by_value(
    bodies_plain: list[str], *, higher_is_better: bool
) -> list[str]:
    """
    单列内：可解析数值的单元格中，最优 \\textbf{\\textcolor{blue}{...}}，
    次优 \\textbf{\\textcolor{violet}{...}}；并列同档同色；不可解析的保持原正文（已去源表着色）。
    """
    if not bodies_plain:
        return bodies_plain
    parse_fn = _nmax_parse_task_value if higher_is_better else _nmax_parse_avg_rank_value
    indexed: list[tuple[int, float, str]] = []
    for i, body in enumerate(bodies_plain):
        v = parse_fn(body)
        if v is not None:
            indexed.append((i, v, body.strip()))
    if not indexed:
        return list(bodies_plain)
    values = [t[1] for t in indexed]
    uniq = _nmax_unique_sorted_values(values, reverse=higher_is_better)
    v_best = uniq[0]
    v_second = uniq[1] if len(uniq) > 1 else None
    out = list(bodies_plain)
    for i, v, body in indexed:
        if np.isclose(v, v_best, rtol=1e-5, atol=1e-8):
            out[i] = r"\textbf{\textcolor{blue}{" + body + "}}"
        elif v_second is not None and np.isclose(
            v, v_second, rtol=1e-5, atol=1e-8
        ):
            out[i] = r"\textbf{\textcolor{violet}{" + body + "}}"
        else:
            out[i] = body
    return out


def rank_colorize_latex_cells(cells: list[str]) -> list[str]:
    """
    每行内：在可解析数值的单元格中，mean 最高为 \\textbf{\\textcolor{blue}{...}}，
    次高为 \\textbf{\\textcolor{violet}{...}}（并列则同档同色）。
    """
    if not cells:
        return cells
    indexed_vals: list[tuple[int, float]] = []
    for i, c in enumerate(cells):
        v = _parse_mean_from_latex_cell(c)
        if v is not None:
            indexed_vals.append((i, v))
    if len(indexed_vals) < 1:
        return list(cells)
    values = [iv[1] for iv in indexed_vals]
    v_best = max(values)
    uniq_desc: list[float] = []
    for v in sorted(values, reverse=True):
        if not uniq_desc or not np.isclose(v, uniq_desc[-1], rtol=1e-5, atol=1e-8):
            uniq_desc.append(v)
    v_second = uniq_desc[1] if len(uniq_desc) > 1 else None

    out = list(cells)
    for i, v in indexed_vals:
        body = cells[i]
        if np.isclose(v, v_best, rtol=1e-5, atol=1e-8):
            out[i] = r"\textbf{\textcolor{blue}{" + body + "}}"
        elif v_second is not None and np.isclose(
            v, v_second, rtol=1e-5, atol=1e-8
        ):
            out[i] = r"\textbf{\textcolor{violet}{" + body + "}}"
    return out


def rank_colorize_latex_mean_rank_row(cells: list[str]) -> list[str]:
    """
    Mean rank 行：**数值越小越好**（秩 1 最优）；对 ``m $\\pm$ s`` 取第一个数比较。
    跳过 ``--``、``/``、空。
    """
    if not cells:
        return cells
    indexed_vals: list[tuple[int, float]] = []
    for i, c in enumerate(cells):
        s = c.strip()
        if not s or s in ("--", "—", "/", "nan"):
            continue
        v = _parse_mean_from_latex_cell(c)
        if v is not None:
            indexed_vals.append((i, v))
    if len(indexed_vals) < 1:
        return list(cells)
    values = [iv[1] for iv in indexed_vals]
    v_best = min(values)
    uniq_asc: list[float] = []
    for v in sorted(values):
        if not uniq_asc or not np.isclose(v, uniq_asc[-1], rtol=1e-5, atol=1e-8):
            uniq_asc.append(v)
    v_second = uniq_asc[1] if len(uniq_asc) > 1 else None
    out = list(cells)
    for i, v in indexed_vals:
        body = cells[i]
        if np.isclose(v, v_best, rtol=1e-5, atol=1e-8):
            out[i] = r"\textbf{\textcolor{blue}{" + body + "}}"
        elif v_second is not None and np.isclose(
            v, v_second, rtol=1e-5, atol=1e-8
        ):
            out[i] = r"\textbf{\textcolor{violet}{" + body + "}}"
    return out


def stats_cell_latex(
    bucket: dict[str, dict[str, Any]], exp: str, task_key: str
) -> str:
    if not exp:
        return "--"
    exp_tasks = bucket.get(exp)
    if not exp_tasks:
        return "--"
    st = exp_tasks.get(task_key)
    if not st:
        return "--"
    return fmt_pm_latex(st.get("max_mean"), st.get("max_std"))


def stats_cell_latex_nmax(
    bucket: dict[str, dict[str, dict[str, Any]]], exp: str, task_key: str
) -> str:
    if not exp:
        return "--"
    exp_tasks = bucket.get(exp)
    if not exp_tasks:
        return "--"
    st = exp_tasks.get(task_key)
    if not st:
        return "--"
    return fmt_pm_latex(st.get("nmax_mean"), st.get("nmax_std"))


def _mean_nmax_from_task_stats(st: dict[str, Any] | None) -> float | None:
    if st is None:
        return None
    m = st.get("nmax_mean")
    if m == "" or m is None:
        return None
    try:
        v = float(m)
    except (TypeError, ValueError):
        return None
    if np.isnan(v):
        return None
    return v


def column_mean_rank_stats_higher_nan_worst(
    means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    与 ``column_mean_rank_stats`` 相同，但每行中 **nan 视为该任务上最差**（用于 nmax 表缺测格）。
    """
    n_tasks, n_cols = means.shape
    ranks = np.full((n_tasks, n_cols), np.nan, dtype=np.float64)
    for i in range(n_tasks):
        row = means[i].astype(np.float64)
        fin = row[np.isfinite(row)]
        if fin.size == 0:
            continue
        lo = float(np.min(fin))
        row_f = row.copy()
        row_f[~np.isfinite(row_f)] = lo - 1.0 - max(abs(lo), 1.0) * 1e-6
        r = rankdata(-row_f, method="average")
        ranks[i, :] = r
    mu = np.empty(n_cols, dtype=np.float64)
    sd = np.empty(n_cols, dtype=np.float64)
    for j in range(n_cols):
        col = ranks[:, j]
        fin = col[np.isfinite(col)]
        if fin.size == 0:
            mu[j] = np.nan
            sd[j] = np.nan
        else:
            mu[j] = float(np.mean(fin))
            sd[j] = float(np.std(fin, ddof=1)) if fin.size > 1 else 0.0
    return mu, sd


# --- 12 列矩阵：单任务 / 局部 multi(label) / 全部 multi(label) / 局部 multi(text) / 全部 multi(text) × returns ---

RETCOND_INFIX = "_retcond"


def _single_exp_name(task: str, use_returns: bool) -> str:
    return f"{task}_multiple_runs{RETCOND_INFIX if use_returns else ''}"


def _msub_label_exp_name(task: str, use_returns: bool) -> str:
    base = TASK_TO_SUBGROUP_MULTITASK_EXP.get(task)
    if not base:
        return ""
    return f"{base}{RETCOND_INFIX if use_returns else ''}"


def _mfull_label_exp_name(use_returns: bool) -> str:
    return f"{FULL_MULTITASK_EXP}{RETCOND_INFIX if use_returns else ''}"


def _msub_text_exp_name(task: str, use_returns: bool, text_suffix: str) -> str:
    """text_suffix 如 ``_textcond`` 或 ``_textcond_mttextonly``（与 run_multitask 中 _rs+_tc+_mto 顺序一致）。"""
    base = TASK_TO_SUBGROUP_MULTITASK_EXP.get(task)
    if not base:
        return ""
    r = RETCOND_INFIX if use_returns else ""
    return f"{base}{r}{text_suffix}"


def _mfull_text_exp_name(use_returns: bool, text_suffix: str) -> str:
    r = RETCOND_INFIX if use_returns else ""
    return f"{FULL_MULTITASK_EXP}{r}{text_suffix}"


def _stats_cell_pm(
    bucket: dict[str, dict[str, dict[str, Any]]], exp: str, task_key: str
) -> str:
    if not exp or exp not in bucket:
        return "—"
    st = bucket[exp].get(task_key)
    if not st:
        return "—"
    return fmt_pm(st.get("max_mean"), st.get("max_std"))


def _stats_cell_pm_latex(
    bucket: dict[str, dict[str, dict[str, Any]]], exp: str, task_key: str
) -> str:
    if not exp or exp not in bucket:
        return "--"
    st = bucket[exp].get(task_key)
    if not st:
        return "--"
    return fmt_pm_latex(st.get("max_mean"), st.get("max_std"))


def _matrix12_text_cell(
    bucket: dict[str, dict[str, dict[str, Any]]],
    task_key: str,
    use_returns: bool,
    text_suffixes: Sequence[str],
    *,
    full_multitask: bool,
    latex: bool,
) -> str:
    """text：先尝试旧 ``multitask_*_textcond`` 键；再在 ``text_conditioned_only/…/<hyper>/`` 下按任务取最优 hyper。"""
    for suf in text_suffixes:
        exp = (
            _mfull_text_exp_name(use_returns, suf)
            if full_multitask
            else _msub_text_exp_name(task_key, use_returns, suf)
        )
        if exp and exp in bucket:
            st = bucket[exp].get(task_key)
            if st is not None:
                if latex:
                    return fmt_pm_latex(st.get("max_mean"), st.get("max_std"))
                return fmt_pm(st.get("max_mean"), st.get("max_std"))
    if full_multitask:
        ap = gtgdfgo_all_text_prefix()
        k = best_gtgdfgo_exp_for_task(bucket, ap, task_key, use_returns)
        if k and k in bucket:
            st = bucket[k].get(task_key)
            if st is not None:
                if latex:
                    return fmt_pm_latex(st.get("max_mean"), st.get("max_std"))
                return fmt_pm(st.get("max_mean"), st.get("max_std"))
    else:
        sp = gtgdfgo_subgroup_text_prefix(task_key)
        if sp:
            k = best_gtgdfgo_exp_for_task(bucket, sp, task_key, use_returns)
            if k and k in bucket:
                st = bucket[k].get(task_key)
                if st is not None:
                    if latex:
                        return fmt_pm_latex(st.get("max_mean"), st.get("max_std"))
                    return fmt_pm(st.get("max_mean"), st.get("max_std"))
    if latex:
        return "--"
    return "—"


def matrix12_row(
    task_key: str,
    gtg: dict[str, dict[str, dict[str, Any]]],
    gtgdfgo: dict[str, dict[str, dict[str, Any]]],
    text_suffixes: Sequence[str],
    *,
    uniso_cell: str = "—",
    latex: bool = False,
) -> dict[str, Any]:
    """单行：task + UniSO-T（Improved）+ 12 列 max 的 mean±std（Markdown/CSV 用 —；LaTeX 用 latex=True）。"""

    def F(bucket: dict[str, dict[str, dict[str, Any]]], exp: str) -> str:
        if not latex:
            return _stats_cell_pm(bucket, exp, task_key)
        return _stats_cell_pm_latex(bucket, exp, task_key)

    def G(
        bucket: dict[str, dict[str, dict[str, Any]]],
        prefix: str | None,
        use_ret: bool,
    ) -> str:
        if not prefix:
            return "--" if latex else "—"
        bk = best_gtgdfgo_exp_for_task(bucket, prefix, task_key, use_ret)
        return F(bucket, bk or "")

    out: dict[str, Any] = {"task": task_key}
    if latex:
        out["c00_uniso"] = (
            uniso_cell
            if (uniso_cell and uniso_cell not in ("", "—"))
            else "--"
        )
    else:
        out["c00_uniso"] = (
            uniso_cell.replace(r" $\pm$ ", " ± ")
            if uniso_cell and uniso_cell not in ("", "—")
            else "—"
        )
    out["c01_gtg_st"] = F(gtg, _single_exp_name(task_key, False))
    out["c02_gtg_st_ret"] = F(gtg, _single_exp_name(task_key, True))
    out["c03_gdf_st"] = G(gtgdfgo, gtgdfgo_single_task_prefix(task_key), False)
    out["c04_gdf_st_ret"] = G(gtgdfgo, gtgdfgo_single_task_prefix(task_key), True)
    out["c05_gdf_msub_l"] = G(
        gtgdfgo, gtgdfgo_subgroup_multitask_prefix(task_key), False
    )
    out["c06_gdf_msub_l_ret"] = G(
        gtgdfgo, gtgdfgo_subgroup_multitask_prefix(task_key), True
    )
    out["c07_gdf_mfull_l"] = G(gtgdfgo, gtgdfgo_full_multitask_prefix(), False)
    out["c08_gdf_mfull_l_ret"] = G(gtgdfgo, gtgdfgo_full_multitask_prefix(), True)
    out["c09_gdf_msub_t"] = _matrix12_text_cell(
        gtgdfgo, task_key, False, text_suffixes, full_multitask=False, latex=latex
    )
    out["c10_gdf_msub_t_ret"] = _matrix12_text_cell(
        gtgdfgo, task_key, True, text_suffixes, full_multitask=False, latex=latex
    )
    out["c11_gdf_mfull_t"] = _matrix12_text_cell(
        gtgdfgo, task_key, False, text_suffixes, full_multitask=True, latex=latex
    )
    out["c12_gdf_mfull_t_ret"] = _matrix12_text_cell(
        gtgdfgo, task_key, True, text_suffixes, full_multitask=True, latex=latex
    )
    return out


MATRIX12_COLUMN_KEYS: list[tuple[str, str]] = [
    ("c00_uniso", "UniSO-T"),
    ("c01_gtg_st", "GTG 单任务"),
    ("c02_gtg_st_ret", "GTG 单任务+ret"),
    ("c03_gdf_st", "DFGO 单任务"),
    ("c04_gdf_st_ret", "DFGO 单任务+ret"),
    ("c05_gdf_msub_l", "DFGO 局部multi(label)"),
    ("c06_gdf_msub_l_ret", "DFGO 局部multi(label)+ret"),
    ("c07_gdf_mfull_l", "DFGO 全部multi(label)"),
    ("c08_gdf_mfull_l_ret", "DFGO 全部multi(label)+ret"),
    ("c09_gdf_msub_t", "DFGO 局部multi(text)"),
    ("c10_gdf_msub_t_ret", "DFGO 局部multi(text)+ret"),
    ("c11_gdf_mfull_t", "DFGO 全部multi(text)"),
    ("c12_gdf_mfull_t_ret", "DFGO 全部multi(text)+ret"),
]


def build_matrix12_rows(
    gtg: dict[str, dict[str, dict[str, Any]]],
    gtgdfgo: dict[str, dict[str, dict[str, Any]]],
    text_suffixes: Sequence[str],
    uniso_by_task: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t in TASK_ORDER:
        if t not in TASK_TO_SUBGROUP_MULTITASK_EXP:
            continue
        u = uniso_by_task.get(t, "")
        ucell = u if u else "—"
        rows.append(matrix12_row(t, gtg, gtgdfgo, text_suffixes, uniso_cell=ucell))
    return rows


def write_matrix12_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = ["task"] + [k for k, _ in MATRIX12_COLUMN_KEYS]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({fn: row.get(fn, "") for fn in fieldnames})


def write_matrix12_latex(
    path: Path,
    caption: str,
    label: str,
    gtg: dict[str, dict[str, dict[str, Any]]],
    gtgdfgo: dict[str, dict[str, dict[str, Any]]],
    text_suffixes: Sequence[str],
    uniso_by_task: dict[str, str],
) -> None:
    """与 ``write_latex`` 相同版式：含 UniSO-T 列与 Mean rank 行。"""
    lines = [
        r"% Requires: \usepackage{booktabs}, \usepackage{graphicx}, \usepackage{xcolor}.",
        r"% Per row: best / runner-up across UniSO-T + 12 method columns.",
        r"\begin{table*}[t!]",
        rf"\caption{{{caption}}}",
        r"\vspace{0.3em}",
        r"\centering",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{l|c|c|c|*{12}{c}}",
        r"\toprule",
        r"Task & \shortstack{UniSO-T} & \shortstack{GTG\\ST} & \shortstack{GTG\\ST+r} & \shortstack{DFGO\\ST} & \shortstack{DFGO\\ST+r} & "
        r"\shortstack{DFGO\\subL} & \shortstack{DFGO\\subL+r} & \shortstack{DFGO\\fullL} & \shortstack{DFGO\\fullL+r} & "
        r"\shortstack{DFGO\\subT} & \shortstack{DFGO\\subT+r} & \shortstack{DFGO\\fullT} & \shortstack{DFGO\\fullT+r} \\",
        r"\midrule",
    ]
    task_keys_list = [t for t in TASK_ORDER if t in TASK_TO_SUBGROUP_MULTITASK_EXP]
    n_tasks = len(task_keys_list)
    means_m = np.full((n_tasks, len(MATRIX12_COLUMN_KEYS)), np.nan, dtype=np.float64)
    for ti, task_key in enumerate(task_keys_list):
        u = uniso_by_task.get(task_key, "")
        mr = matrix12_row(
            task_key,
            gtg,
            gtgdfgo,
            text_suffixes,
            uniso_cell=u if u else "—",
            latex=True,
        )
        keys = [k for k, _ in MATRIX12_COLUMN_KEYS]
        cells = [mr[k] for k in keys]
        cells = rank_colorize_latex_cells(cells)
        row_name = latex_task_display_name(task_key)
        lines.append(" & ".join([row_name] + cells) + r" \\")
        for j, k in enumerate(keys):
            v = parse_mean_from_text_cell(mr[k])
            if v is None:
                v = _parse_mean_from_latex_cell(mr[k])
            if v is not None:
                means_m[ti, j] = v
    mu_r, sd_r = column_mean_rank_stats(means_m)
    rank_cells = [fmt_mean_pm_rank(mu_r[j], sd_r[j]) for j in range(len(MATRIX12_COLUMN_KEYS))]
    rank_cells = rank_colorize_latex_mean_rank_row(rank_cells)
    lines.extend(
        [
            r"\midrule",
            "Mean rank & " + " & ".join(rank_cells) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            rf"\label{{{label}}}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def d_best_cell(
    gtgdfgo_root: Path,
    task_key: str,
    overrides: dict[str, float],
) -> str:
    if task_key in overrides:
        return f"{overrides[task_key]:.{DECIMALS}f}"
    v = read_dataset_best_from_experiment(gtgdfgo_root, task_key)
    if v is None:
        return "--"
    return f"{v:.{DECIMALS}f}"


def collect_all_experiment_keys(
    gtgdfgo: dict[str, dict[str, dict[str, Any]]],
    results_root: Path,
) -> list[str]:
    """
    ``text_conditioned_result_analysis``：``text_conditioned_only/<EVAL_ALL_TASK_FRAC_SIG>/`` 下
    每个**直接子目录**（超参文件夹名 = 关键参数）一列 DFGO；列顺序为目录名字典序。
    """
    base = results_root / TEXT_CONDITIONED_ROOT / EVAL_ALL_TASK_FRAC_SIG
    if not base.is_dir():
        return []
    keys: list[str] = []
    for hyper in sorted(base.iterdir()):
        if not hyper.is_dir() or hyper.name.startswith("."):
            continue
        if not any(hyper.rglob("evaluate.log")):
            continue
        k = f"{EVAL_ALL_EXPERIMENT_PREFIX}/{hyper.name}"
        keys.append(k)
    return keys


def _tex_col_short_name(exp_key: str) -> str:
    """LaTeX 列标题：去掉一级根目录前缀，便于排版。"""
    if exp_key.startswith(f"{TEXT_CONDITIONED_ROOT}/"):
        return exp_key[len(TEXT_CONDITIONED_ROOT) + 1 :]
    for p in (f"{r}/" for r in (MULTI_TASK_ROOT, SINGLE_TASK_ROOT)):
        if exp_key.startswith(p):
            return exp_key[len(p) :]
    return exp_key


def _tex_col_all_hyper_param_name(exp_key: str) -> str:
    """text_conditioned_result_analysis 列名：仅超参目录名（w1.2_*、NxH_k_eps、_ret 等）。"""
    pfx = f"{EVAL_ALL_EXPERIMENT_PREFIX}/"
    if exp_key.startswith(pfx):
        return exp_key[len(pfx) :]
    return _tex_col_short_name(exp_key)


def _latex_sig_header(sig: str) -> str:
    """列标题：下划线转义，过长时用 shortstack。"""
    esc = sig.replace("_", r"\_")
    if len(esc) > 42:
        esc = esc[:40] + r"\ldots"
    return rf"\shortstack{{\texttt{{{esc}}}}}"


def write_latex_all_fulltext(
    path: Path,
    gtgdfgo_root: Path,
    gtg: dict[str, dict[str, dict[str, Any]]],
    gtgdfgo: dict[str, dict[str, dict[str, Any]]],
    d_best_overrides: dict[str, float],
    uniso_by_task: dict[str, str],
    caption: str,
    label: str,
) -> None:
    """多列 DFGO：``all_frac*_sigma*/`` 下每个超参子目录一列，列名为子目录名（关键参数）。"""
    exp_keys = collect_all_experiment_keys(gtgdfgo, gtgdfgo_root)
    if not exp_keys:
        path.write_text(
            rf"% No data for text_conditioned_result_analysis (expected subdirs under {EVAL_ALL_EXPERIMENT_PREFIX}/ with evaluate.log).\n",
            encoding="utf-8",
        )
        return
    n_sub = len(exp_keys)
    tab_spec = "l|c|c|cc|" + ("c" * n_sub)
    hdr_dfgo = " & ".join(
        _latex_sig_header(_tex_col_all_hyper_param_name(k)) for k in exp_keys
    )
    hdr_row = (
        r"Task & $\mathcal{D}$(best) & \shortstack{UniSO-T} & \shortstack{GTG\\ST} & "
        r"\shortstack{GTG\\ST+r} & " + hdr_dfgo + r" \\"
    )
    lines = [
        r"% Requires: \usepackage{booktabs}, \usepackage{graphicx}, \usepackage{xcolor}.",
        r"% Blocks: Task+D(best) | UniSO-T | GTG ST/ST+r | DFGO (text_conditioned_only/all_frac…).",
        r"\begin{table*}[t!]",
        rf"\caption{{{caption}}}",
        r"\vspace{0.3em}",
        r"\centering",
        r"\resizebox{\linewidth}{!}{",
        rf"\begin{{tabular}}{{{tab_spec}}}",
        r"\toprule",
        hdr_row,
        r"\midrule",
    ]
    n_rows = len(LATEX_TASK_ROWS)
    # UniSO + GTG ST + GTG ST+r + 每列 DFGO
    n_cols_data = 3 + n_sub
    means_a = np.full((n_rows, n_cols_data), np.nan, dtype=np.float64)
    for ri, (task_key, latex_name, exp_name) in enumerate(LATEX_TASK_ROWS):
        db = d_best_cell(gtgdfgo_root, task_key, d_best_overrides)
        u_raw = uniso_by_task.get(task_key, "")
        u_cell = u_raw if u_raw else "--"
        gtg_st = stats_cell_latex(gtg, _single_exp_name(task_key, False), task_key)
        gtg_sr = stats_cell_latex(gtg, _single_exp_name(task_key, True), task_key)
        dfgo_cells: list[str] = []
        col_vals: list[float | None] = [
            parse_mean_from_text_cell(u_cell),
            _mean_from_task_stats(gtg.get(_single_exp_name(task_key, False), {}).get(task_key)),
            _mean_from_task_stats(gtg.get(_single_exp_name(task_key, True), {}).get(task_key)),
        ]
        for exp_key in exp_keys:
            c = stats_cell_latex(gtgdfgo, exp_key, task_key)
            dfgo_cells.append(c)
            col_vals.append(_mean_from_task_stats(gtgdfgo.get(exp_key, {}).get(task_key)))
        means_a[ri, :] = [np.nan if v is None else v for v in col_vals]
        row_cells = [u_cell, gtg_st, gtg_sr] + dfgo_cells
        row_cells = rank_colorize_latex_cells(row_cells)
        lines.append(
            f"{latex_name} & {db} & " + " & ".join(row_cells) + r" \\"
        )
    mu_a, sd_a = column_mean_rank_stats(means_a)
    rank_parts = [fmt_mean_pm_rank(mu_a[j], sd_a[j]) for j in range(n_cols_data)]
    rank_parts = rank_colorize_latex_mean_rank_row(rank_parts)
    lines.extend(
        [
            r"\midrule",
            "Mean rank & / & " + " & ".join(rank_parts) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            rf"\label{{{label}}}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(
    path: Path,
    gtgdfgo_root: Path,
    gtg: dict[str, dict[str, dict[str, Any]]],
    gtgdfgo: dict[str, dict[str, dict[str, Any]]],
    d_best_overrides: dict[str, float],
    caption: str,
    label: str,
    uniso_by_task: dict[str, str],
) -> None:
    lines = [
        r"% Requires: \usepackage{booktabs}, \usepackage{graphicx}, \usepackage{xcolor}.",
        r"% Per row: best / runner-up among UniSO-T + GTG + three GTGdfgo columns.",
        r"\begin{table*}[t!]",
        rf"\caption{{{caption}}}",
        r"\vspace{0.3em}",
        r"\centering",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{l|c|c|c|ccc}",
        r"\toprule",
        r"Task & $\mathcal{D}$(best) & \shortstack{UniSO-T} & GTG & GTGdfgo (single) & GTGdfgo (multi, subgroup) & GTGdfgo (multi, all tasks) \\",
        r"\midrule",
    ]
    n_rows = len(LATEX_TASK_ROWS)
    means_w = np.full((n_rows, 5), np.nan, dtype=np.float64)
    for ri, (task_key, latex_name, exp_name) in enumerate(LATEX_TASK_ROWS):
        db = d_best_cell(gtgdfgo_root, task_key, d_best_overrides)
        u_raw = uniso_by_task.get(task_key, "")
        u_cell = u_raw if u_raw else "--"
        gtg_c = stats_cell_latex(gtg, exp_name, task_key)
        k1 = best_gtgdfgo_exp_for_task(
            gtgdfgo, gtgdfgo_single_task_prefix(task_key), task_key, False
        )
        g1 = stats_cell_latex(gtgdfgo, k1 or "", task_key)
        sp = gtgdfgo_subgroup_multitask_prefix(task_key)
        ksub = (
            best_gtgdfgo_exp_for_task(gtgdfgo, sp, task_key, False) if sp else None
        )
        g_sub = stats_cell_latex(gtgdfgo, ksub or "", task_key)
        kfull = best_gtgdfgo_exp_for_task(
            gtgdfgo, gtgdfgo_full_multitask_prefix(), task_key, False
        )
        g_full = stats_cell_latex(gtgdfgo, kfull or "", task_key)
        u_m = parse_mean_from_text_cell(u_cell)
        gtg_m = _mean_from_task_stats(gtg.get(exp_name, {}).get(task_key))
        g1_m = _mean_from_task_stats(
            gtgdfgo.get(k1, {}).get(task_key) if k1 else None
        )
        gsub_m = _mean_from_task_stats(
            gtgdfgo.get(ksub, {}).get(task_key) if ksub else None
        )
        gfull_m = _mean_from_task_stats(
            gtgdfgo.get(kfull, {}).get(task_key) if kfull else None
        )
        means_w[ri, 0] = np.nan if u_m is None else u_m
        means_w[ri, 1] = np.nan if gtg_m is None else gtg_m
        means_w[ri, 2] = np.nan if g1_m is None else g1_m
        means_w[ri, 3] = np.nan if gsub_m is None else gsub_m
        means_w[ri, 4] = np.nan if gfull_m is None else gfull_m
        u_show, gtg_c, g1, g_sub, g_full = rank_colorize_latex_cells(
            [u_cell, gtg_c, g1, g_sub, g_full]
        )
        lines.append(
            f"{latex_name} & {db} & {u_show} & {gtg_c} & {g1} & {g_sub} & {g_full} \\\\"
        )
    mu_w, sd_w = column_mean_rank_stats(means_w)
    rank_parts = [fmt_mean_pm_rank(mu_w[j], sd_w[j]) for j in range(5)]
    rank_parts = rank_colorize_latex_mean_rank_row(rank_parts)
    lines.extend(
        [
            r"\midrule",
            "Mean rank & / & " + " & ".join(rank_parts) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            rf"\label{{{label}}}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_nmax_design_bench(
    path: Path,
    _gtgdfgo_root: Path,
    _gtg: dict[str, dict[str, dict[str, Any]]],
    gtgdfgo: dict[str, dict[str, dict[str, Any]]],
    _d_best_overrides: dict[str, float],
    uniso_nresult_path: Path,
    caption: str,
    label: str,
) -> None:
    """
    归一化分数：基线行为 ``uniso_nresult.tex`` 全文；DFGO 为日志中
    ``nmax_ep_reward``（与 evaluate 中 ``(y-f.min)/(f.max-f.min)`` 一致，与表内基线可比）。
    """
    parsed = parse_uniso_nresult_rows(uniso_nresult_path)
    if not parsed:
        path.write_text(
            rf"% Missing or empty {uniso_nresult_path.name} (normalized Design-Bench table).\n",
            encoding="utf-8",
        )
        return
    n_t = len(DESIGN_BENCH_TASK_ORDER)
    n_base = len(parsed)
    means_all = np.full((n_t, n_base + 2), np.nan, dtype=np.float64)
    for ti in range(n_t):
        for j in range(n_base):
            v = parse_mean_from_text_cell(parsed[j][2][ti])
            means_all[ti, j] = np.nan if v is None else v
        tk = DESIGN_BENCH_TASK_ORDER[ti]
        k1 = best_gtgdfgo_exp_for_task(
            gtgdfgo, gtgdfgo_single_task_prefix(tk), tk, False
        )
        g1m = _mean_nmax_from_task_stats(
            gtgdfgo.get(k1, {}).get(tk) if k1 else None
        )
        means_all[ti, n_base] = np.nan if g1m is None else g1m
        kfull = best_gtgdfgo_exp_for_task(
            gtgdfgo, gtgdfgo_full_multitask_prefix(), tk, False
        )
        if tk == "superconductor":
            means_all[ti, n_base + 1] = np.nan
        else:
            gfm = _mean_nmax_from_task_stats(
                gtgdfgo.get(kfull, {}).get(tk) if kfull else None
            )
            means_all[ti, n_base + 1] = np.nan if gfm is None else gfm
    mu_r, _sd_r = column_mean_rank_stats_higher_nan_worst(means_all)
    n_rank_total = nmax_avg_rank_denominator_total(n_base, parsed)
    rank_dfgo_s_plain = nmax_fmt_dfgo_avg_rank(mu_r[n_base], n_rank_total)
    rank_dfgo_m_plain = nmax_fmt_dfgo_avg_rank(mu_r[n_base + 1], n_rank_total)
    g1_cells: list[str] = []
    gm_cells: list[str] = []
    for ti, task_key in enumerate(DESIGN_BENCH_TASK_ORDER):
        k1 = best_gtgdfgo_exp_for_task(
            gtgdfgo, gtgdfgo_single_task_prefix(task_key), task_key, False
        )
        g1_cells.append(stats_cell_latex_nmax(gtgdfgo, k1 or "", task_key))
        kfull = best_gtgdfgo_exp_for_task(
            gtgdfgo, gtgdfgo_full_multitask_prefix(), task_key, False
        )
        if task_key == "superconductor":
            gm_cells.append("--")
        else:
            gm_cells.append(stats_cell_latex_nmax(gtgdfgo, kfull or "", task_key))
    n_row = n_base + 2
    # 列方向：先剥掉源表行内着色，再与 DFGO 一起按数值重算蓝/紫（任务列越大越好，Avg.\ Rank 越小越好）
    task_cols_plain: list[list[str]] = [[] for _ in range(5)]
    for ti in range(5):
        col: list[str] = []
        for j in range(n_base):
            col.append(latex_cell_strip_wrappers(parsed[j][2][ti]))
        col.append(g1_cells[ti])
        col.append(gm_cells[ti])
        task_cols_plain[ti] = col
    for ti in range(5):
        task_cols_plain[ti] = nmax_colorize_column_by_value(
            task_cols_plain[ti], higher_is_better=True
        )
    avg_plain: list[str] = [
        nmax_fix_avg_rank_denominator(
            latex_cell_strip_wrappers(parsed[j][3]), n_rank_total
        )
        for j in range(n_base)
    ]
    avg_plain.append(rank_dfgo_s_plain)
    avg_plain.append(rank_dfgo_m_plain)
    avg_plain = nmax_colorize_column_by_value(avg_plain, higher_is_better=False)
    hdr = (
        r"Method & Venue & Ant & D'Kitty & Superconductor & TF-Bind-8 & TF-Bind-10 & Avg.\ Rank \\"
    )
    lines = [
        r"% Requires: \usepackage{booktabs}, \usepackage{graphicx}, \usepackage{xcolor}.",
        r"% Baselines: uniso_nresult.tex; per-column blue/violet = best/2nd in this table (tasks: higher better; Avg.\ Rank: lower better).",
        r"\begin{table*}[t!]",
        rf"\caption{{{caption}}}",
        r"\vspace{0.3em}",
        r"\centering",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{c|c|ccccc|c}",
        r"\toprule",
        hdr,
        r"\midrule",
    ]
    for j, (method, venue, _tasks, _avg) in enumerate(parsed):
        mcell = _nmax_baseline_method_cell(method)
        row_tasks = [task_cols_plain[ti][j] for ti in range(5)]
        lines.append(
            " & ".join([mcell, venue] + row_tasks + [avg_plain[j]]) + r" \\"
        )
        if _nresult_midrule_after_row(method):
            lines.append(r"\midrule")
    if parsed and _nresult_midrule_before_dfgo(parsed[-1][0]):
        lines.append(r"\midrule")
    lines.append(
        r"\textbf{DFGO} (single) & / & "
        + " & ".join(task_cols_plain[ti][n_base] for ti in range(5))
        + r" & "
        + avg_plain[n_base]
        + r" \\"
    )
    lines.append(
        r"\textbf{DFGO} (multi, all tasks) & / & "
        + " & ".join(task_cols_plain[ti][n_base + 1] for ti in range(5))
        + r" & "
        + avg_plain[n_base + 1]
        + r" \\"
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            rf"\label{{{label}}}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _round_csv_value(key: str, v: Any) -> Any:
    if key.endswith(("_mean", "_std")) and isinstance(v, (int, float)) and not isinstance(v, bool):
        if np.isnan(float(v)):
            return v
        return round(float(v), DECIMALS)
    return v


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: _round_csv_value(k, v) for k, v in row.items()})


def fmt_pm(m: Any, s: Any) -> str:
    if m == "" or m is None:
        return "—"
    try:
        mf, sf = float(m), float(s)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(mf):
        return "nan"
    if sf and not np.isnan(sf):
        return f"{mf:.{DECIMALS}f} ± {sf:.{DECIMALS}f}"
    return f"{mf:.{DECIMALS}f} ± {0:.{DECIMALS}f}"


# 相对本脚本：GTGdfgo 仓库根；实验日志在 results/；汇总表与 UniSO 输入在 results/analysis_table/。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
GTGDFGO_RESULTS = _PROJECT_ROOT / "results"
GTG_RESULTS = _PROJECT_ROOT.parent / "GTG" / "results"
ANALYSIS_TABLE_DIR = _PROJECT_ROOT / "results" / "analysis_table"
OUTPUT_BASE = ANALYSIS_TABLE_DIR / "max_short"
MATRIX12_BASE = ANALYSIS_TABLE_DIR / "max_extended"
NMAX_TEX = ANALYSIS_TABLE_DIR / "nmax.tex"
UNISO_RESULT_TEX = ANALYSIS_TABLE_DIR / "uniso_result.tex"
UNISO_NRESULT_TEX = ANALYSIS_TABLE_DIR / "uniso_nresult.tex"
# run_multitask：仅 --use_text_condition → …_textcond；再加 --multitask_text_only → …_textcond_mttextonly。优先匹配后者。
MATRIX12_TEXT_SUFFIXES: tuple[str, ...] = ("_textcond_mttextonly", "_textcond")
D_BEST_JSON = ANALYSIS_TABLE_DIR / "d_best.json"

LATEX_CAPTION = (
    "Un-normalized \\texttt{max\\_ep\\_reward} (mean $\\pm$ std over runs). "
    "\\textbf{UniSO-T}: the \\textbf{Improved} column under UniSO-T in \\texttt{uniso\\_result.tex} (not the max over UniSO-T/N). "
    "Mean rank: per task, rank methods by mean (higher better); report mean $\\pm$ std across tasks. "
    "$\\mathcal{D}$(best): offline training subset (\\texttt{offline\\_train\\_best\\_y}); "
    "optional \\texttt{results/analysis\\_table/d\\_best.json} overrides. "
    "\\textbf{Hyperparameters (DFGO):} each result subfolder name encodes trajectory settings "
    "(e.g.\\ \\texttt{4000x64\\_k20\\_eps0.05}), optional \\texttt{\\_ret} for return-conditioning, "
    "and in text-conditioned multitask a \\texttt{w*} prefix for the text loss weight. "
    "Per task and training mode, we report the subfolder with the largest mean \\texttt{max\\_ep\\_reward}."
)
LATEX_LABEL = "tab:gtg-gtgdfgo-eval"
LATEX_CAPTION_MATRIX12 = (
    "Un-normalized \\texttt{max\\_ep\\_reward} (mean $\\pm$ std): UniSO-T (Improved) + GTG / DFGO single-task, "
    "subgroup vs full multitask (label / text $\\times$ returns). "
    "Hyperparameter selection matches the wide max-score table. "
    "Mean rank: per task among UniSO-T + 12 columns; mean $\\pm$ std across tasks."
)
LATEX_LABEL_MATRIX12 = "tab:gtg-gtgdfgo-eval-m12"
LATEX_CAPTION_NMAX = (
    "Normalized scores on the five Design-Bench tasks (same setting as \\texttt{uniso\\_nresult.tex}). "
    "Baseline rows are copied from that table; \\textbf{DFGO} rows report \\texttt{nmax\\_ep\\_reward} "
    "(mean $\\pm$ std over runs), i.e.\\ $(y-y_{\\min})/(y_{\\max}-y_{\\min})$ in \\texttt{evaluate.py}, "
    "comparable to the normalized baselines. "
    "Hyper selection for DFGO matches the wide max table (best subfolder per task by mean \\texttt{max\\_ep\\_reward}). "
    "\\textbf{DFGO (multi, all tasks):} Superconductor is blank (not in the original full multitask run); "
    "missing cells are worst when ranking. "
    "Avg.\\ Rank for DFGO: mean per-task rank / $N$ ($N$ = pool size, same as baselines)."
)
LATEX_LABEL_NMAX = "tab:dfgo-nmax-designbench"
TEXT_CONDITIONED_RESULT_ANALYSIS_BASE = (
    ANALYSIS_TABLE_DIR / "text_conditioned_result_analysis"
)
_LATEX_ESC_ALL_SIG = EVAL_ALL_TASK_FRAC_SIG.replace("_", r"\_")
LATEX_CAPTION_ALL = (
    "DFGO (full multitask, text-conditioned): one column per immediate subfolder of "
    f"\\texttt{{results/text\\_conditioned\\_only/{_LATEX_ESC_ALL_SIG}/}} "
    "(folder names encode key hyperparameters, e.g.\\ \\texttt{w1.2\\_}$\\cdots$); "
    "each column pools runs inside that folder only. "
    "UniSO-T (Improved from \\texttt{uniso\\_result.tex}), GTG ST / ST+r, and DFGO. Mean rank: higher mean reward is better; "
    "report mean $\\pm$ std of per-task ranks."
)
LATEX_LABEL_ALL = "tab:gtg-gtgdfgo-eval-all-text"


def main(mode: str = "all") -> None:
    d_best_overrides: dict[str, float] = {}
    if D_BEST_JSON.is_file():
        raw = json.loads(D_BEST_JSON.read_text(encoding="utf-8"))
        d_best_overrides = {str(k): float(v) for k, v in raw.items()}

    uniso_by_task = parse_uniso_best_per_task(UNISO_RESULT_TEX)

    gtgdfgo = scan_results_root(GTGDFGO_RESULTS)
    gtg = scan_results_root(GTG_RESULTS)

    do_short = mode in ("all", "short")
    do_full = mode in ("all", "full")
    do_final = mode in ("all", "final")

    if do_short:
        out_base = OUTPUT_BASE
        rows = enrich_multitask_columns(
            sort_comparison_rows(build_comparison_rows(gtgdfgo, gtg)), gtgdfgo
        )
        tex_path = out_base.with_suffix(".tex")
        out_base.parent.mkdir(parents=True, exist_ok=True)

        write_csv(out_base.with_suffix(".csv"), rows)
        write_latex(
            tex_path,
            GTGDFGO_RESULTS,
            gtg,
            gtgdfgo,
            d_best_overrides,
            caption=LATEX_CAPTION,
            label=LATEX_LABEL,
            uniso_by_task=uniso_by_task,
        )
        write_latex_nmax_design_bench(
            NMAX_TEX,
            GTGDFGO_RESULTS,
            gtg,
            gtgdfgo,
            d_best_overrides,
            UNISO_NRESULT_TEX,
            caption=LATEX_CAPTION_NMAX,
            label=LATEX_LABEL_NMAX,
        )
        print(f"Wrote {out_base.with_suffix('.csv')}")
        print(f"Wrote {tex_path}")
        print(f"Wrote {NMAX_TEX}")
        print(
            f"GTGdfgo experiments: {len(gtgdfgo)}, GTG experiments: {len(gtg)}, comparison rows: {len(rows)}"
        )

    if do_final:
        all_base = TEXT_CONDITIONED_RESULT_ANALYSIS_BASE
        all_base.parent.mkdir(parents=True, exist_ok=True)
        write_latex_all_fulltext(
            all_base.with_suffix(".tex"),
            GTGDFGO_RESULTS,
            gtg,
            gtgdfgo,
            d_best_overrides,
            uniso_by_task,
            caption=LATEX_CAPTION_ALL,
            label=LATEX_LABEL_ALL,
        )
        print(f"Wrote {all_base.with_suffix('.tex')}")

    if do_full:
        mbase = MATRIX12_BASE
        mrows = build_matrix12_rows(
            gtg, gtgdfgo, MATRIX12_TEXT_SUFFIXES, uniso_by_task
        )
        mbase.parent.mkdir(parents=True, exist_ok=True)
        write_matrix12_csv(mbase.with_suffix(".csv"), mrows)
        write_matrix12_latex(
            mbase.with_suffix(".tex"),
            LATEX_CAPTION_MATRIX12,
            LATEX_LABEL_MATRIX12,
            gtg,
            gtgdfgo,
            MATRIX12_TEXT_SUFFIXES,
            uniso_by_task,
        )
        print(f"Wrote {mbase.with_suffix('.csv')}")
        print(f"Wrote {mbase.with_suffix('.tex')}")
        print(
            f"Matrix12 rows: {len(mrows)} (tasks with subgroup multitask mapping)"
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate evaluate.log and write tables under results/analysis_table/ (see README)."
    )
    p.add_argument(
        "--mode",
        choices=("all", "short", "full", "final"),
        default="all",
        help=(
            "Which outputs to generate: "
            "short = analysis_table/max_short (wide CSV+TeX) and nmax.tex; "
            "full = analysis_table/max_extended (13-column matrix CSV+TeX); "
            "final = analysis_table/text_conditioned_result_analysis (DFGO: one column per hyper subdir under all_frac…); "
            "all = three tables (default)."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(mode=args.mode)

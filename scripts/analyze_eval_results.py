#!/usr/bin/env python3
"""
Aggregate evaluate.log metrics across runs (run*_seed* / run*) per experiment,
then compare DUO vs GTG results in one report (CSV + LaTeX table).

Metrics: max_ep_reward -> max, nmax_ep_reward -> nmax (mean ± std over runs).

DFGO 列在每种模式（单任务 / 小组 multi / 全任务 multi / text）下，对
``single_task|multi_task|text_conditioned_only/<…>/<hyper>/run*`` 中**不同超参目录**
（如 ``NxH_k*_eps*``、``w*_…``）分别聚合后，**按该任务取 max 均值最高**的一组作为该列结果。

``max_short`` 表中四列 DUO（single / single+text / multi+label / multi+text）限制为
``examples/traj_params_per_task_example2.json`` 对应的轨迹超参（见 ``max_short_traj_context``）。

``duo_mfull_*`` (``multi_task/…`` labels, **unified** hyperfolder across tasks) and
``duo_mfull_text_*`` (full multitask text) — see
``best_duo_exp_full_multitask_unified``, ``duo_all_text_prefix``; ``duo_st_text_*`` 为单任务 + textcond。

所有汇总表与 UniSO 输入均位于 ``results/analysis_table/``：宽表 ``max_short.*``、矩阵 ``max_extended.*``、``text_conditioned_result_analysis.*``、``nmax.tex``，以及 ``max_ablation.*``（text CFG 消融，由 ``--mode sweep_w`` 生成）、``uniso_result.tex``、``uniso_nresult.tex``、``d_best.json``（可选）。
``text_conditioned_result_analysis``（``--mode final``）：``text_conditioned_only/all_frac1.0_sigma0.0/``（可用 ``EVAL_ALL_TASK_FRAC_SIG`` 覆盖）下**每个超参子目录一列** DFGO。默认一次生成全部；也可用 ``--mode short|full|final``（见 ``run_analyze_eval.sh``）。``--mode sweep_w`` 或 ``bash run_analyze_eval.sh -sweep-w`` 生成 ``max_ablation.{tex,csv}``（``eval_sweep_w_text`` 下 text CFG 消融表）。
矩阵 ``max_extended``、``nmax`` 中 DFGO「全任务 multitask text」列：若存在 ``results/eval_sweep_w_text/<mt_*>/`` 且 ``DUO_SWEEP_W_DISABLE`` 未置 1，则按 ``max_ablation`` **Mean rank 行各 w 列**（UniSO、GTG ST 与各 w 联合排名后的平均秩）选出最优 ``condition_guidance_w_text``，该列数据来自对应 ``eval_w*.log``（caption 中备注 ``mt_<16hex>`` 与统一 ``w``）。
矩阵列为 **UniSO-T**（``uniso_result.tex`` 中 **UniSO-T Improved** 列）+ 14 列方法（含全任务 text 的 ``all`` 与 ``all_improved`` 两列）；文本多任务目录后缀见 ``MATRIX12_TEXT_SUFFIXES``。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import rankdata


def _env_compat(duo_key: str, legacy_key: str, default: str = "") -> str:
    """读 ``DUO_*`` 环境变量，未设时回退到旧名 ``GTGDFGO_*``（重命名前脚本兼容）。"""
    for k in (duo_key, legacy_key):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    return default

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

    GTG/DUO 的 ``{task}_multiple_runs_retcond`` 若以 ``_multiple_runs$`` 正则匹配会失败，
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

# DUO results 布局：``single_task/<task>_frac*_sigma*``、``multi_task/<token>_frac*_sigma*`` 等
DUO_TASK_FRAC_SIG: str = _env_compat(
    "DUO_TASK_FRAC_SIG", "GTGDFGO_TASK_FRAC_SIG", "frac1.0_sigma0.0"
)

# 旧 ``multitask_*`` 名 -> 新 ``multi_task`` / ``text_conditioned_only`` 下目录 token（不含 frac_sigma）
MULTITASK_NAME_TO_DIR: dict[str, str] = {
    "multitask_ant_dkitty": "ant_dkitty",
    "multitask_tfbind10_tfbind8": "tfbind10_tfbind8",
    "multitask_gtopx2_gtopx3_gtopx4_gtopx6": "gtopx2_gtopx3_gtopx4_gtopx6",
}


def duo_single_task_prefix(task: str) -> str:
    return f"{SINGLE_TASK_ROOT}/{task}_{DUO_TASK_FRAC_SIG}"


def duo_subgroup_multitask_prefix(task: str) -> str | None:
    old = TASK_TO_SUBGROUP_MULTITASK_EXP.get(task)
    if not old:
        return None
    token = MULTITASK_NAME_TO_DIR.get(old)
    if not token:
        return None
    return f"{MULTI_TASK_ROOT}/{token}_{DUO_TASK_FRAC_SIG}"


def duo_full_multitask_prefix() -> str:
    """
    全任务 multitask（**分类 label**）：``multi_task/all_<DUO_TASK_FRAC_SIG>/``。
    用于 max_short / max_extended 的 DFGO fullL、``enrich_multitask_columns`` 的 mfull。
    汇总表在 **所有任务上共用同一子目录**：见 :func:`best_duo_exp_full_multitask_unified`；
    可用 ``DUO_FULL_MULTITASK_HYPER`` 强制子目录名。
    覆盖：环境变量 ``DUO_FULL_MULTITASK_PREFIX``（含 hyper 的完整相对路径亦可）。
    """
    v = _env_compat(
        "DUO_FULL_MULTITASK_PREFIX",
        "GTGDFGO_FULL_MULTITASK_PREFIX",
        "",
    )
    if v:
        return v
    return f"{MULTI_TASK_ROOT}/all_{DUO_TASK_FRAC_SIG}"


def duo_all_improved_text_prefix() -> str:
    """全任务 text（``all_improved_*`` 轨迹目录），与 ``run_multitask.sh`` 中 USE_TRAJ_PARAMS_JSON 布局一致。"""
    sp = _env_compat("DUO_SWEEP_W_PREFIX", "GTGDFGO_SWEEP_W_PREFIX", "")
    if sp:
        return sp
    subdir = os.environ.get(
        "EVAL_ALL_IMPROVED_TASK_FRAC_SIG",
        f"all_improved_{DUO_TASK_FRAC_SIG}",
    )
    return f"{TEXT_CONDITIONED_ROOT}/{subdir}"


def duo_full_multitask_nmax_prefix() -> str:
    """
    nmax 表中 DFGO「multi, all tasks, text_conditioned」行：默认 ``text_conditioned_only/all_improved_*``。
    覆盖：``DUO_NMAX_MULTITASK_PREFIX``。
    若已注入 ``eval_sweep_w_text/...``（``DUO_SWEEP_W_PREFIX``），则与全任务 text 列一致，用 sweep 上选出的统一 ``w``。
    """
    sp = _env_compat("DUO_SWEEP_W_PREFIX", "GTGDFGO_SWEEP_W_PREFIX", "")
    if sp:
        return sp
    return _env_compat(
        "DUO_NMAX_MULTITASK_PREFIX",
        "GTGDFGO_NMAX_MULTITASK_PREFIX",
        f"{TEXT_CONDITIONED_ROOT}/all_improved_{DUO_TASK_FRAC_SIG}",
    )


def duo_subgroup_text_prefix(task: str) -> str | None:
    """``text_conditioned_only/<token>_<frac_sigma>/`` 下各 ``<hyper>/``（与 ``_ret`` 后缀区分 returns 列）。"""
    old = TASK_TO_SUBGROUP_MULTITASK_EXP.get(task)
    if not old:
        return None
    token = MULTITASK_NAME_TO_DIR.get(old)
    if not token:
        return None
    return f"{TEXT_CONDITIONED_ROOT}/{token}_{DUO_TASK_FRAC_SIG}"


def duo_all_text_prefix() -> str:
    """全任务 multitask text（``text_conditioned_only/all_<frac_sigma>/``）；若存在 sweep 注入则与 ``all_improved`` 列共用 ``eval_sweep_w_text/...``。"""
    sp = _env_compat("DUO_SWEEP_W_PREFIX", "GTGDFGO_SWEEP_W_PREFIX", "")
    if sp:
        return sp
    return f"{TEXT_CONDITIONED_ROOT}/all_{DUO_TASK_FRAC_SIG}"


def _hyper_matches_returns(hyper: str, use_returns: bool) -> bool:
    return hyper.endswith("_ret") == use_returns


def _candidate_exp_keys_for_prefix(
    bucket: dict[str, dict[str, dict[str, Any]]],
    prefix: str,
    use_returns: bool,
) -> list[str]:
    """``prefix`` 自身及 ``prefix/<hyper>/…`` 下第一层 ``hyper`` 与 ``use_returns`` 匹配的实验键。"""
    cand: list[str] = []
    if prefix in bucket:
        cand.append(prefix)
    pslash = prefix + "/"
    for k in bucket:
        if not k.startswith(pslash):
            continue
        rest = k[len(pslash) :]
        hyper = rest.split("/")[0]
        if not _hyper_matches_returns(hyper, use_returns):
            continue
        cand.append(k)
    return sorted(set(cand))


def best_duo_exp_full_multitask_unified(
    bucket: dict[str, dict[str, dict[str, Any]]],
    prefix: str,
    task_keys: Sequence[str],
    use_returns: bool,
    *,
    allowed_first_hyper: str | None = None,
) -> str | None:
    """
    全任务 multitask（label）**单列共用同一超参子目录**：在 ``prefix`` 下选一个实验键，使
    ``task_keys`` 上有数据的 ``max_mean`` **跨任务算术平均**最大（与逐任务取最优不同）。

    可用环境变量 ``DUO_FULL_MULTITASK_HYPER`` 强制子目录名（如 ``mt_f6fca707c7e948df``），
    则使用 ``{prefix}/{DUO_FULL_MULTITASK_HYPER}``（若存在于 ``bucket``）。

    ``allowed_first_hyper``：若给定，仅考虑 ``prefix/<该名>/…`` 形式的候选（用于 max_short 与
    ``traj_params_per_task_example2.json`` 一致的 ``mt_<hex>``）。
    """
    fixed = _env_compat(
        "DUO_FULL_MULTITASK_HYPER",
        "GTGDFGO_FULL_MULTITASK_HYPER",
        "",
    ).strip()
    if fixed:
        candidate = f"{prefix}/{fixed}"
        if candidate in bucket:
            return candidate
    candidates = _candidate_exp_keys_for_prefix(bucket, prefix, use_returns)
    if allowed_first_hyper and not fixed:
        filt: list[str] = []
        for k in candidates:
            if k == prefix:
                continue
            rest = k[len(prefix) + 1 :]
            hyper = rest.split("/")[0]
            if hyper == allowed_first_hyper:
                filt.append(k)
        candidates = filt
    if not candidates:
        return None
    best_k: str | None = None
    best_score = float("-inf")
    for k in sorted(candidates):
        vals: list[float] = []
        for task in task_keys:
            st = bucket.get(k, {}).get(task)
            if not st:
                continue
            m = st.get("max_mean")
            if m is not None and np.isfinite(float(m)):
                vals.append(float(m))
        score = float(np.mean(vals)) if vals else float("-inf")
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def best_duo_exp_for_task(
    bucket: dict[str, dict[str, dict[str, Any]]],
    prefix: str,
    task: str,
    use_returns: bool,
    *,
    hyper_eq: str | None = None,
) -> str | None:
    """
    在 ``prefix`` 或 ``prefix/<hyper>/…`` 下，按该任务 ``max_mean``（max_ep_reward）最大选取实验目录。
    ``hyper`` 名以 ``_ret`` 结尾表示开启 returns，与无 ``_ret`` 的列分开比较。

    ``hyper_eq``：若给定，仅比较首段超参目录名等于该串的候选（如 ``1000x64_k20_eps0.05``、
    ``1000x64_k20_eps0.05_textcond``，与 ``traj_params_per_task_example2.json`` 一致）。
    """
    best_k: str | None = None
    best_m: float | None = None
    if prefix in bucket and hyper_eq is None:
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
        if hyper_eq is not None and hyper != hyper_eq:
            continue
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


_MAX_SHORT_TRAJ_CTX: dict[str, Any] | None = None


def max_short_traj_json_path() -> Path:
    """max_short 表 DUO 四列：与 ``examples/traj_params_per_task_example2.json`` 对齐（可用 ``DUO_MAX_SHORT_TRAJ_JSON`` 覆盖）。"""
    o = os.environ.get("DUO_MAX_SHORT_TRAJ_JSON", "").strip()
    if o:
        return Path(o).resolve()
    return Path(__file__).resolve().parent.parent / "examples" / "traj_params_per_task_example2.json"


def max_short_traj_context() -> dict[str, Any]:
    """单任务 slug、全任务 ``mt_<hex>``（label / text 目录名）等，与 train/evaluate 中 prepare_multitask_traj 一致。"""
    global _MAX_SHORT_TRAJ_CTX
    if _MAX_SHORT_TRAJ_CTX is not None:
        return _MAX_SHORT_TRAJ_CTX
    from diffuser.utils.multitask_canon import (
        canonical_train_tasks_csv,
        multitask_text_only_path_infix,
        returns_cond_path_infix,
        text_cond_path_infix,
    )
    from diffuser.utils.traj_params import (
        multitask_checkpoint_hyper_dir,
        multitask_slug_id,
        prepare_multitask_traj,
    )

    jpath = max_short_traj_json_path()
    traj_arg = str(jpath) if jpath.is_file() else None
    horizon = int(os.environ.get("DUO_MAX_SHORT_HORIZON", "64"))
    csv = os.environ.get(
        "DUO_MAX_SHORT_FULL_MT_TASKS",
        "ant,dkitty,gtopx2,gtopx3,gtopx4,gtopx6,superconductor,tfbind10,tfbind8",
    )
    tasks = [t.strip() for t in canonical_train_tasks_csv(csv).split(",") if t.strip()]
    _, _, _, sig_full = prepare_multitask_traj(
        tasks, 1000, 50, 0.05, horizon, traj_arg
    )
    mt_core = multitask_slug_id(sig_full)

    def _mt_dir(text: bool, mto: bool) -> str:
        ns = SimpleNamespace(
            returns_condition=False,
            include_returns=False,
            use_text_condition=text,
            multitask_text_only=mto if text else False,
        )
        _ret = returns_cond_path_infix(ns)
        _txt = text_cond_path_infix(ns)
        _mto = multitask_text_only_path_infix(ns)
        _, _, _, sig = prepare_multitask_traj(
            tasks, 1000, 50, 0.05, horizon, traj_arg
        )
        return multitask_checkpoint_hyper_dir(sig, _ret, _txt, _mto)

    st_slug: dict[str, str] = {}
    st_text_slug: dict[str, str] = {}
    for t in tasks:
        _, _, _, sig_t = prepare_multitask_traj(
            [t], 1000, 50, 0.05, horizon, traj_arg
        )
        st_slug[t] = sig_t
        st_text_slug[t] = sig_t + "_textcond"

    _MAX_SHORT_TRAJ_CTX = {
        "json": jpath,
        "horizon": horizon,
        "mt_label_hyper": _mt_dir(False, False),
        "mt_text_dir_hyper": _mt_dir(True, True),
        "mt_core": mt_core,
        "st_slug": st_slug,
        "st_text_slug": st_text_slug,
    }
    return _MAX_SHORT_TRAJ_CTX


def best_duo_exp_for_multitask_text_maxshort(
    bucket: dict[str, dict[str, dict[str, Any]]],
    task: str,
    use_returns: bool,
    mt_core_substr: str,
) -> str | None:
    """
    全任务 multitask + text：在 ``duo_all_text_prefix()`` 下选该任务 ``max_mean`` 最大者，
    且实验键须包含 ``mt_<hex>``（与 example2 一致），以排除标量 n_traj/k/eps 的其它目录。
    """
    tp = duo_all_text_prefix()
    best_k: str | None = None
    best_m: float | None = None
    for k in bucket:
        if mt_core_substr not in k:
            continue
        if not (k == tp or k.startswith(tp + "/")):
            continue
        if k.startswith(tp + "/"):
            rest = k[len(tp) + 1 :]
            hyper = rest.split("/")[0]
            if not _hyper_matches_returns(hyper, use_returns):
                continue
        st = bucket.get(k, {}).get(task)
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


def _hyper_dir_name_from_exp(exp_key: str | None, base_prefix: str) -> str:
    """``base_prefix/<hyper>/…`` 中的 ``hyper`` 目录名；无子目录时为 ``(base)``。"""
    if not exp_key:
        return "—"
    if exp_key == base_prefix:
        return "(base)"
    sep = base_prefix + "/"
    if not exp_key.startswith(sep):
        return "—"
    rest = exp_key[len(sep) :]
    return rest.split("/")[0] if rest else "—"


def _latex_tt(s: str) -> str:
    """``\\texttt{...}`` 内转义。"""
    return s.replace("\\", r"\textbackslash{}").replace("_", r"\_")


_MT_SLUG_HEX_RE = re.compile(r"^mt_([0-9a-f]{16})")


def merge_eval_sweep_w_text_into_duo(
    duo: dict[str, dict[str, dict[str, Any]]],
    results_root: Path,
) -> str:
    """
    从 ``results/eval_sweep_w_text/<mt_*>/`` 的 ``eval_w*.log`` 按与 ``max_ablation`` **Mean rank 行中各 w 列**
    相同的准则（UniSO、GTG ST 与各 w 联合排名后，取平均秩最小的 ``w``）选出最优 ``w``，
    注入虚拟实验键 ``eval_sweep_w_text/<mt>/w_text<w>``（及 ``…_ret``，数值相同供 +returns 列使用），
    并设置 ``DUO_SWEEP_W_PREFIX``，使 ``duo_all_text_prefix`` / ``all_improved`` / nmax multi text 均指向该前缀。

    可用 ``DUO_SWEEP_W_DISABLE=1`` 关闭；``SWEEP_W_MODEL_DIR`` 指定 ``mt_*`` 目录（否则选最新 ``mt_*``）。
    返回供 LaTeX caption 追加的英文说明片段（无注入时为空串）。
    """
    _swd = _env_compat("DUO_SWEEP_W_DISABLE", "GTGDFGO_SWEEP_W_DISABLE", "")
    if _swd.strip().lower() in ("1", "true", "yes"):
        return ""
    import importlib.util

    mod_path = Path(__file__).resolve().parent / "make_sweep_w_ablation_table.py"
    spec = importlib.util.spec_from_file_location("_sw_mrg", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    sweep_root = results_root / "eval_sweep_w_text"
    override = os.environ.get("SWEEP_W_MODEL_DIR", "").strip()
    try:
        if override:
            model_dir = Path(override).resolve()
        else:
            model_dir = mod.discover_default_model_dir(sweep_root)
    except (FileNotFoundError, OSError):
        return ""

    max_v, nmax_v, w_list = mod.collect_sweep_max_nmax_lists(model_dir)
    if not w_list:
        return ""

    task_keys = [t for t, _, _ in MAX_TEX_TASK_ROWS]
    best_w = mod.sweep_best_w_match_max_ablation(model_dir, GTG_RESULTS)
    if best_w is None:
        return ""

    stats = mod.build_sweep_injected_task_stats(max_v, nmax_v, best_w, task_keys)
    if not stats:
        return ""

    mt = model_dir.name
    base = f"eval_sweep_w_text/{mt}"
    os.environ["DUO_SWEEP_W_PREFIX"] = base
    os.environ["GTGDFGO_SWEEP_W_PREFIX"] = base
    os.environ["DUO_SWEEP_W_VALUE"] = str(best_w)
    os.environ["GTGDFGO_SWEEP_W_VALUE"] = str(best_w)

    w_tag = f"w_text{best_w:g}"
    duo[f"{base}/{w_tag}"] = stats
    duo[f"{base}/{w_tag}_ret"] = stats

    hex_m = _MT_SLUG_HEX_RE.match(mt)
    hex_note = hex_m.group(1) if hex_m else re.sub(r"[^0-9a-f]", "", mt)[:16]
    mt_esc = _latex_tt(mt)
    slug_tt = _latex_tt(f"mt_{hex_note}")
    return (
        " \\textbf{DFGO full multitask text} (matrix cols.\\ fullT / nmax multi text): "
        f"\\texttt{{eval\\_sweep\\_w\\_text/{mt_esc}}} "
        f"with unified $\\texttt{{condition\\_guidance\\_w\\_text}}={best_w:g}$ "
        r"(best $w$: lowest mean rank among $w$ columns in \texttt{max\_ablation}, joint per-task ranking with UniSO and GTG ST). "
        rf"Trajectory slug \texttt{{{slug_tt}}} is the 16-hex-digit digest prefix of the multitask trajectory signature (SHA-256). "
        r"+returns text columns use the same sweep logits (no separate ret eval). "
    )


def latex_hyper_selection_minipage_lines(
    bucket: dict[str, dict[str, dict[str, Any]]],
    sections: list[
        tuple[str, Callable[[str], str | None]]
        | tuple[str, Callable[[str], str | None], Callable[..., str | None]]
    ],
    task_keys: Sequence[str],
    use_returns: bool,
    *,
    unified_full_multitask_labels: tuple[str, str, str | None] | None = None,
) -> list[str]:
    """
    在表末附注：各 DFGO 列在对应基路径下按任务选取的最优超参子目录名（task→hyper）。
    ``sections``：每项为 (标题, task→base_prefix)；可选第三项
    ``(bucket, prefix, task) -> exp_key | None`` 覆盖默认的 ``best_duo_exp_for_task``。

    ``unified_full_multitask_labels``：若给定 ``(title, base_prefix, exp_key)``，在附注最前插入一行
    **全任务 multitask（label）单列共用** 的超参说明（``all tasks → hyper``），不再按任务拆分。
    """
    block_parts: list[str] = []
    if unified_full_multitask_labels is not None:
        utitle, upfx, uexp = unified_full_multitask_labels
        if uexp and upfx:
            h = _hyper_dir_name_from_exp(uexp, upfx)
            block_parts.append(
                f"\\textbf{{{_latex_tt(utitle)}}}: "
                f"all tasks$\\rightarrow$\\texttt{{{_latex_tt(h)}}}"
            )
        elif upfx:
            try:
                slug = str(max_short_traj_context()["mt_label_hyper"])
            except Exception:
                slug = "mt_<hex>"
            block_parts.append(
                f"\\textbf{{{_latex_tt(utitle)}}}: "
                f"no aggregated runs under \\texttt{{{_latex_tt(upfx + '/' + slug)}}} "
                r"(\texttt{traj\_params\_per\_task\_example2.json}); table shows ``--''."
            )
    for section in sections:
        if len(section) == 3:
            title, resolv, resolve_key = section
        else:
            title, resolv = section
            resolve_key = None
        row_parts: list[str] = []
        for tk in task_keys:
            pfx = resolv(tk)
            if not pfx:
                continue
            if resolve_key is not None:
                k = resolve_key(bucket, pfx, tk)
            else:
                k = best_duo_exp_for_task(bucket, pfx, tk, use_returns)
            h = _hyper_dir_name_from_exp(k, pfx)
            row_parts.append(f"{tk}$\\rightarrow$\\texttt{{{_latex_tt(h)}}}")
        if row_parts:
            block_parts.append(
                f"\\textbf{{{_latex_tt(title)}}}: " + "; ".join(row_parts)
            )
    if not block_parts:
        return []
    inner = r" \par ".join(block_parts)
    return [
        r"\vspace{0.4em}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        r"\textit{Hyper selection (best subfolder per task):} " + inner,
        r"\end{minipage}",
    ]


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
    duo: dict[str, dict[str, dict[str, Any]]],
    gtg: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    exp_names = _ordered_experiment_names(set(duo.keys()) | set(gtg.keys()))
    rows: list[dict[str, Any]] = []
    for exp in exp_names:
        tasks_g = duo.get(exp, {})
        tasks_c = gtg.get(exp, {})
        task_names = _ordered_task_names(set(tasks_g.keys()) | set(tasks_c.keys()))
        for task in task_names:
            row: dict[str, Any] = {
                "experiment": exp,
                "task": task,
            }
            for prefix, src in (("duo", tasks_g), ("gtg", tasks_c)):
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
    duo: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    全任务 label 列（``duo_mfull_*``）使用 **同一** 超参目录（跨任务 ``max_mean`` 均值最大）；
    text 列（``duo_mfull_text_*``）按任务在 ``duo_all_text_prefix()`` 下择优；
    二者均限制为 ``traj_params_per_task_example2.json`` 对应的 ``mt_<hex>``（见 ``max_short_traj_context``）。
    ``duo_st_text_*``：单任务 + ``_textcond``、与 example2 轨迹一致的超参目录。
    """
    ctx = max_short_traj_context()
    fp = duo_full_multitask_prefix()
    full_unified = best_duo_exp_full_multitask_unified(
        duo,
        fp,
        [t for t, _, _ in MAX_TEX_TASK_ROWS],
        False,
        allowed_first_hyper=str(ctx["mt_label_hyper"]),
    )
    out: list[dict[str, Any]] = []
    for row in rows:
        task = row["task"]
        full_st = _lookup_task_stats(duo, full_unified, task)
        text_k = best_duo_exp_for_multitask_text_maxshort(
            duo, task, False, str(ctx["mt_core"])
        )
        text_st = _lookup_task_stats(duo, text_k, task)
        st_text_k = best_duo_exp_for_task(
            duo,
            duo_single_task_prefix(task),
            task,
            False,
            hyper_eq=ctx["st_text_slug"].get(task),
        )
        st_text_st = _lookup_task_stats(duo, st_text_k, task)
        new_row = dict(row)
        new_row.update(_prefix_stats_flat("duo_mfull", full_st))
        new_row.update(_prefix_stats_flat("duo_mfull_text", text_st))
        new_row.update(_prefix_stats_flat("duo_st_text", st_text_st))
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

# 与宽表一致的任务行（``max_ablation`` 等同）；**必须包含 Superconductor**（在 D'Kitty 与 TF Bind 8 之间）。
MAX_TEX_TASK_ROWS: list[tuple[str, str, str]] = list(LATEX_TASK_ROWS)

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
    full_multitask_text_prefix: str | None = None,
    latex: bool,
) -> str:
    """text：先尝试旧 ``multitask_*_textcond`` 键；再在 ``text_conditioned_only/…/<hyper>/`` 下按任务取最优 hyper。
    全任务 multitask 时 ``full_multitask_text_prefix`` 为 ``None`` 则用 ``all_<frac_sigma>``；否则用给定基路径（如 ``all_improved_*``）。
    若设置了 ``DUO_SWEEP_W_PREFIX``（全任务 text 来自 ``eval_sweep_w_text`` 上选出的统一 ``w``），优先使用该前缀。
    """
    sweep_p = _env_compat("DUO_SWEEP_W_PREFIX", "GTGDFGO_SWEEP_W_PREFIX", "")
    if sweep_p and full_multitask:
        bk = best_duo_exp_for_task(bucket, sweep_p, task_key, use_returns)
        if bk and bk in bucket:
            st = bucket[bk].get(task_key)
            if st is not None:
                if latex:
                    return fmt_pm_latex(st.get("max_mean"), st.get("max_std"))
                return fmt_pm(st.get("max_mean"), st.get("max_std"))
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
        ap = (
            full_multitask_text_prefix
            if full_multitask_text_prefix is not None
            else duo_all_text_prefix()
        )
        k = best_duo_exp_for_task(bucket, ap, task_key, use_returns)
        if k and k in bucket:
            st = bucket[k].get(task_key)
            if st is not None:
                if latex:
                    return fmt_pm_latex(st.get("max_mean"), st.get("max_std"))
                return fmt_pm(st.get("max_mean"), st.get("max_std"))
    else:
        sp = duo_subgroup_text_prefix(task_key)
        if sp:
            k = best_duo_exp_for_task(bucket, sp, task_key, use_returns)
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
    duo: dict[str, dict[str, dict[str, Any]]],
    text_suffixes: Sequence[str],
    *,
    uniso_cell: str = "—",
    latex: bool = False,
    mfull_unified_key: str | None = None,
    mfull_unified_key_ret: str | None = None,
) -> dict[str, Any]:
    """单行：task + UniSO-T（Improved）+ 14 列 max 的 mean±std（Markdown/CSV 用 —；LaTeX 用 latex=True）。
    ``mfull_unified_key*``：全任务 label 列共用的实验键（见 :func:`best_duo_exp_full_multitask_unified`）。"""

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
        fp = duo_full_multitask_prefix()
        if prefix == fp:
            uk = mfull_unified_key_ret if use_ret else mfull_unified_key
            if uk:
                return F(bucket, uk)
        bk = best_duo_exp_for_task(bucket, prefix, task_key, use_ret)
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
    out["c03_gdf_st"] = G(duo, duo_single_task_prefix(task_key), False)
    out["c04_gdf_st_ret"] = G(duo, duo_single_task_prefix(task_key), True)
    out["c05_gdf_msub_l"] = G(
        duo, duo_subgroup_multitask_prefix(task_key), False
    )
    out["c06_gdf_msub_l_ret"] = G(
        duo, duo_subgroup_multitask_prefix(task_key), True
    )
    out["c07_gdf_mfull_l"] = G(duo, duo_full_multitask_prefix(), False)
    out["c08_gdf_mfull_l_ret"] = G(duo, duo_full_multitask_prefix(), True)
    out["c09_gdf_msub_t"] = _matrix12_text_cell(
        duo, task_key, False, text_suffixes, full_multitask=False, latex=latex
    )
    out["c10_gdf_msub_t_ret"] = _matrix12_text_cell(
        duo, task_key, True, text_suffixes, full_multitask=False, latex=latex
    )
    out["c11_gdf_mfull_t"] = _matrix12_text_cell(
        duo,
        task_key,
        False,
        text_suffixes,
        full_multitask=True,
        full_multitask_text_prefix=None,
        latex=latex,
    )
    out["c12_gdf_mfull_t_ret"] = _matrix12_text_cell(
        duo,
        task_key,
        True,
        text_suffixes,
        full_multitask=True,
        full_multitask_text_prefix=None,
        latex=latex,
    )
    out["c13_gdf_mfull_t_imp"] = _matrix12_text_cell(
        duo,
        task_key,
        False,
        text_suffixes,
        full_multitask=True,
        full_multitask_text_prefix=duo_all_improved_text_prefix(),
        latex=latex,
    )
    out["c14_gdf_mfull_t_imp_ret"] = _matrix12_text_cell(
        duo,
        task_key,
        True,
        text_suffixes,
        full_multitask=True,
        full_multitask_text_prefix=duo_all_improved_text_prefix(),
        latex=latex,
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
    ("c11_gdf_mfull_t", "DFGO 全部multi(text) all"),
    ("c12_gdf_mfull_t_ret", "DFGO 全部multi(text) all+ret"),
    ("c13_gdf_mfull_t_imp", "DFGO 全部multi(text) all_improved"),
    ("c14_gdf_mfull_t_imp_ret", "DFGO 全部multi(text) all_improved+ret"),
]


def build_matrix12_rows(
    gtg: dict[str, dict[str, dict[str, Any]]],
    duo: dict[str, dict[str, dict[str, Any]]],
    text_suffixes: Sequence[str],
    uniso_by_task: dict[str, str],
) -> list[dict[str, Any]]:
    fp = duo_full_multitask_prefix()
    task_keys_full = [t for t, _, _ in MAX_TEX_TASK_ROWS]
    ku = best_duo_exp_full_multitask_unified(
        duo, fp, task_keys_full, False
    )
    kur = best_duo_exp_full_multitask_unified(
        duo, fp, task_keys_full, True
    )
    rows: list[dict[str, Any]] = []
    for t in TASK_ORDER:
        if t not in TASK_TO_SUBGROUP_MULTITASK_EXP:
            continue
        u = uniso_by_task.get(t, "")
        ucell = u if u else "—"
        rows.append(
            matrix12_row(
                t,
                gtg,
                duo,
                text_suffixes,
                uniso_cell=ucell,
                mfull_unified_key=ku,
                mfull_unified_key_ret=kur,
            )
        )
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
    duo: dict[str, dict[str, dict[str, Any]]],
    text_suffixes: Sequence[str],
    uniso_by_task: dict[str, str],
) -> None:
    """与 ``write_latex`` 相同版式：含 UniSO-T 列与 Mean rank 行。"""
    _m12_fp = duo_full_multitask_prefix()
    _m12_ku = best_duo_exp_full_multitask_unified(
        duo,
        _m12_fp,
        [t for t, _, _ in MAX_TEX_TASK_ROWS],
        False,
    )
    m12_note = latex_hyper_selection_minipage_lines(
        duo,
        [
            ("DFGO full T (all)", lambda _: duo_all_text_prefix()),
            ("DFGO full T (all imp)", lambda _: duo_all_improved_text_prefix()),
        ],
        [t for t in TASK_ORDER if t in TASK_TO_SUBGROUP_MULTITASK_EXP],
        False,
        unified_full_multitask_labels=("DFGO full L", _m12_fp, _m12_ku),
    )
    lines = [
        r"% Requires: \usepackage{booktabs}, \usepackage{graphicx}, \usepackage{xcolor}.",
        r"% Per row: best / runner-up across UniSO-T + 14 method columns.",
        r"\begin{table*}[t!]",
        rf"\caption{{{caption}}}",
        r"\vspace{0.3em}",
        r"\centering",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{l|*{15}{c}}",
        r"\toprule",
        r"Task & \shortstack{UniSO-T} & \shortstack{GTG\\ST} & \shortstack{GTG\\ST+r} & \shortstack{DFGO\\ST} & \shortstack{DFGO\\ST+r} & "
        r"\shortstack{DFGO\\subL} & \shortstack{DFGO\\subL+r} & \shortstack{DFGO\\fullL} & \shortstack{DFGO\\fullL+r} & "
        r"\shortstack{DFGO\\subT} & \shortstack{DFGO\\subT+r} & \shortstack{DFGO\\fullT\\,(all)} & \shortstack{DFGO\\fullT\\,(all)+r} & "
        r"\shortstack{DFGO\\fullT\\,(imp)} & \shortstack{DFGO\\fullT\\,(imp)+r} \\",
        r"\midrule",
    ]
    task_keys_list = [t for t in TASK_ORDER if t in TASK_TO_SUBGROUP_MULTITASK_EXP]
    n_tasks = len(task_keys_list)
    means_m = np.full((n_tasks, len(MATRIX12_COLUMN_KEYS)), np.nan, dtype=np.float64)
    _wl_fp = duo_full_multitask_prefix()
    _wl_ku = best_duo_exp_full_multitask_unified(
        duo,
        _wl_fp,
        [t for t, _, _ in MAX_TEX_TASK_ROWS],
        False,
    )
    _wl_kur = best_duo_exp_full_multitask_unified(
        duo,
        _wl_fp,
        [t for t, _, _ in MAX_TEX_TASK_ROWS],
        True,
    )
    for ti, task_key in enumerate(task_keys_list):
        u = uniso_by_task.get(task_key, "")
        mr = matrix12_row(
            task_key,
            gtg,
            duo,
            text_suffixes,
            uniso_cell=u if u else "—",
            latex=True,
            mfull_unified_key=_wl_ku,
            mfull_unified_key_ret=_wl_kur,
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
        ]
    )
    lines.extend(m12_note)
    lines.extend(
        [
            rf"\label{{{label}}}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def d_best_cell(
    duo_root: Path,
    task_key: str,
    overrides: dict[str, float],
) -> str:
    if task_key in overrides:
        return f"{overrides[task_key]:.{DECIMALS}f}"
    v = read_dataset_best_from_experiment(duo_root, task_key)
    if v is None:
        return "--"
    return f"{v:.{DECIMALS}f}"


def collect_all_experiment_keys(
    duo: dict[str, dict[str, dict[str, Any]]],
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
    duo_root: Path,
    gtg: dict[str, dict[str, dict[str, Any]]],
    duo: dict[str, dict[str, dict[str, Any]]],
    d_best_overrides: dict[str, float],
    uniso_by_task: dict[str, str],
    caption: str,
    label: str,
) -> None:
    """多列 DFGO：``all_frac*_sigma*/`` 下每个超参子目录一列，列名为子目录名（关键参数）。"""
    exp_keys = collect_all_experiment_keys(duo, duo_root)
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
        db = d_best_cell(duo_root, task_key, d_best_overrides)
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
            c = stats_cell_latex(duo, exp_key, task_key)
            dfgo_cells.append(c)
            col_vals.append(_mean_from_task_stats(duo.get(exp_key, {}).get(task_key)))
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
    duo_root: Path,
    gtg: dict[str, dict[str, dict[str, Any]]],
    duo: dict[str, dict[str, dict[str, Any]]],
    d_best_overrides: dict[str, float],
    caption: str,
    label: str,
    uniso_by_task: dict[str, str],
) -> None:
    ctx = max_short_traj_context()
    _fp_full = duo_full_multitask_prefix()
    k_full_unified = best_duo_exp_full_multitask_unified(
        duo,
        _fp_full,
        [t for t, _, _ in LATEX_TASK_ROWS],
        False,
        allowed_first_hyper=str(ctx["mt_label_hyper"]),
    )
    short_note = latex_hyper_selection_minipage_lines(
        duo,
        [
            (
                "single",
                lambda tk: duo_single_task_prefix(tk),
                lambda b, p, tk: best_duo_exp_for_task(
                    b, p, tk, False, hyper_eq=ctx["st_slug"].get(tk)
                ),
            ),
            (
                "single+text",
                lambda tk: duo_single_task_prefix(tk),
                lambda b, p, tk: best_duo_exp_for_task(
                    b, p, tk, False, hyper_eq=ctx["st_text_slug"].get(tk)
                ),
            ),
            (
                "multi+text",
                lambda _: duo_all_text_prefix(),
                lambda b, p, tk: best_duo_exp_for_multitask_text_maxshort(
                    b, tk, False, str(ctx["mt_core"])
                ),
            ),
        ],
        [t for t, _, _ in LATEX_TASK_ROWS],
        False,
        unified_full_multitask_labels=(
            "multi+label",
            _fp_full,
            k_full_unified,
        ),
    )
    lines = [
        r"% Requires: \usepackage{booktabs}, \usepackage{graphicx}, \usepackage{xcolor}.",
        r"% Per row: best / runner-up among UniSO-T + GTG + four DUO columns (single / single+text / multi+label / multi+text); "
        r"DUO uses trajectories from examples/traj_params_per_task_example2.json (override DUO_MAX_SHORT_TRAJ_JSON).",
        r"\begin{table*}[t!]",
        rf"\caption{{{caption}}}",
        r"\vspace{0.3em}",
        r"\centering",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{l|c|c|c|cccc}",
        r"\toprule",
        r"Task & $\mathcal{D}$(best) & \shortstack{UniSO-T} & GTG & single & single+text & multi+label & multi+text \\",
        r"\midrule",
    ]
    n_rows = len(LATEX_TASK_ROWS)
    n_rank_methods = 6
    means_w = np.full((n_rows, n_rank_methods), np.nan, dtype=np.float64)
    for ri, (task_key, latex_name, exp_name) in enumerate(LATEX_TASK_ROWS):
        db = d_best_cell(duo_root, task_key, d_best_overrides)
        u_raw = uniso_by_task.get(task_key, "")
        u_cell = u_raw if u_raw else "--"
        gtg_c = stats_cell_latex(gtg, exp_name, task_key)
        k1 = best_duo_exp_for_task(
            duo,
            duo_single_task_prefix(task_key),
            task_key,
            False,
            hyper_eq=ctx["st_slug"].get(task_key),
        )
        g1 = stats_cell_latex(duo, k1 or "", task_key)
        k_st_text = best_duo_exp_for_task(
            duo,
            duo_single_task_prefix(task_key),
            task_key,
            False,
            hyper_eq=ctx["st_text_slug"].get(task_key),
        )
        g_st_text = stats_cell_latex(duo, k_st_text or "", task_key)
        g_full = stats_cell_latex(duo, k_full_unified or "", task_key)
        ktext = best_duo_exp_for_multitask_text_maxshort(
            duo, task_key, False, str(ctx["mt_core"])
        )
        g_text = stats_cell_latex(duo, ktext or "", task_key)
        u_m = parse_mean_from_text_cell(u_cell)
        gtg_m = _mean_from_task_stats(gtg.get(exp_name, {}).get(task_key))
        g1_m = _mean_from_task_stats(
            duo.get(k1, {}).get(task_key) if k1 else None
        )
        gst_m = _mean_from_task_stats(
            duo.get(k_st_text, {}).get(task_key) if k_st_text else None
        )
        gfull_m = _mean_from_task_stats(
            duo.get(k_full_unified, {}).get(task_key)
            if k_full_unified
            else None
        )
        gtext_m = _mean_from_task_stats(
            duo.get(ktext, {}).get(task_key) if ktext else None
        )
        means_w[ri, 0] = np.nan if u_m is None else u_m
        means_w[ri, 1] = np.nan if gtg_m is None else gtg_m
        means_w[ri, 2] = np.nan if g1_m is None else g1_m
        means_w[ri, 3] = np.nan if gst_m is None else gst_m
        means_w[ri, 4] = np.nan if gfull_m is None else gfull_m
        means_w[ri, 5] = np.nan if gtext_m is None else gtext_m
        u_show, gtg_c, g1, g_st_text_c, g_full_col, g_text_col = rank_colorize_latex_cells(
            [u_cell, gtg_c, g1, g_st_text, g_full, g_text]
        )
        lines.append(
            f"{latex_name} & {db} & {u_show} & {gtg_c} & {g1} & {g_st_text_c} & {g_full_col} & {g_text_col} \\\\"
        )
    mu_w, sd_w = column_mean_rank_stats(means_w)
    rank_parts = [fmt_mean_pm_rank(mu_w[j], sd_w[j]) for j in range(n_rank_methods)]
    rank_parts = rank_colorize_latex_mean_rank_row(rank_parts)
    lines.extend(
        [
            r"\midrule",
            "Mean rank & / & " + " & ".join(rank_parts) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
        ]
    )
    lines.extend(short_note)
    lines.extend(
        [
            rf"\label{{{label}}}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_nmax_design_bench(
    path: Path,
    _duo_root: Path,
    _gtg: dict[str, dict[str, dict[str, Any]]],
    duo: dict[str, dict[str, dict[str, Any]]],
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
        k1 = best_duo_exp_for_task(
            duo, duo_single_task_prefix(tk), tk, False
        )
        g1m = _mean_nmax_from_task_stats(
            duo.get(k1, {}).get(tk) if k1 else None
        )
        means_all[ti, n_base] = np.nan if g1m is None else g1m
        kfull = best_duo_exp_for_task(
            duo, duo_full_multitask_nmax_prefix(), tk, False
        )
        gfm = _mean_nmax_from_task_stats(
            duo.get(kfull, {}).get(tk) if kfull else None
        )
        means_all[ti, n_base + 1] = np.nan if gfm is None else gfm
    mu_r, _sd_r = column_mean_rank_stats_higher_nan_worst(means_all)
    n_rank_total = nmax_avg_rank_denominator_total(n_base, parsed)
    rank_dfgo_s_plain = nmax_fmt_dfgo_avg_rank(mu_r[n_base], n_rank_total)
    rank_dfgo_m_plain = nmax_fmt_dfgo_avg_rank(mu_r[n_base + 1], n_rank_total)
    g1_cells: list[str] = []
    gm_cells: list[str] = []
    for ti, task_key in enumerate(DESIGN_BENCH_TASK_ORDER):
        k1 = best_duo_exp_for_task(
            duo, duo_single_task_prefix(task_key), task_key, False
        )
        g1_cells.append(stats_cell_latex_nmax(duo, k1 or "", task_key))
        kfull = best_duo_exp_for_task(
            duo, duo_full_multitask_nmax_prefix(), task_key, False
        )
        gm_cells.append(stats_cell_latex_nmax(duo, kfull or "", task_key))
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
        r"\textbf{DFGO} (multi, all tasks, text\_conditioned) & / & "
        + " & ".join(task_cols_plain[ti][n_base + 1] for ti in range(5))
        + r" & "
        + avg_plain[n_base + 1]
        + r" \\"
    )
    nmax_note = latex_hyper_selection_minipage_lines(
        duo,
        [
            ("DFGO ST (nmax)", lambda tk: duo_single_task_prefix(tk)),
            ("DFGO multi (nmax)", lambda _: duo_full_multitask_nmax_prefix()),
        ],
        list(DESIGN_BENCH_TASK_ORDER),
        False,
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
        ]
    )
    lines.extend(nmax_note)
    lines.extend(
        [
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


# 相对本脚本：DUO 仓库根；实验日志在 results/；汇总表与 UniSO 输入在 results/analysis_table/。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUO_RESULTS = _PROJECT_ROOT / "results"
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
    "\\textbf{single} / \\textbf{single+text}: under \\texttt{single\\_task/\\dots/}, only the hyperfolder matching "
    "\\texttt{examples/traj\\_params\\_per\\_task\\_example2.json} for that task (plus \\texttt{\\_textcond} for single+text). "
    "\\textbf{multi+label}: one shared \\texttt{multi\\_task/all\\_<frac\\_sigma>/mt\\_<hex>/} directory for all tasks, "
    "where \\texttt{mt\\_<hex>} is the digest for the same per-task trajectories as example2 (override \\texttt{DUO\\_FULL\\_MULTITASK\\_HYPER} to force a folder). "
    "\\textbf{multi+text}: full multitask with text under \\texttt{text\\_conditioned\\_only/\\dots/} or injected \\texttt{eval\\_sweep\\_w\\_text/\\dots/}, "
    "restricted to runs whose path contains that same \\texttt{mt\\_<hex>}. "
    "Hyperfolder names are listed in the table note below."
)
LATEX_LABEL = "tab:gtg-duo-eval"
LATEX_CAPTION_MATRIX12 = (
    "Un-normalized \\texttt{max\\_ep\\_reward} (mean $\\pm$ std): UniSO-T (Improved) + GTG / DFGO single-task, "
    "subgroup vs full multitask (label / text $\\times$ returns). "
    "\\textbf{DFGO fullL} is \\texttt{multi\\_task/all\\_…} (labels, one shared hyperfolder by mean over tasks); "
    "\\textbf{DFGO fullT (all)} vs \\textbf{(all imp)} are both under \\texttt{text\\_conditioned\\_only} "
    "(\\texttt{all\\_<frac>} vs \\texttt{all\\_improved\\_<frac>}). "
    "Mean rank: per task among UniSO-T + 14 columns; mean $\\pm$ std across tasks. "
    "Best hyperfolder per task is listed in the table note."
)
LATEX_LABEL_MATRIX12 = "tab:gtg-duo-eval-m12"
LATEX_CAPTION_NMAX = (
    "Normalized scores on the five Design-Bench tasks (same setting as \\texttt{uniso\\_nresult.tex}). "
    "Baseline rows are copied from that table; \\textbf{DFGO} rows report \\texttt{nmax\\_ep\\_reward} "
    "(mean $\\pm$ std over runs), i.e.\\ $(y-y_{\\min})/(y_{\\max}-y_{\\min})$ in \\texttt{evaluate.py}, "
    "comparable to the normalized baselines. "
    "\\textbf{DFGO (single)} uses per-task best under \\texttt{single\\_task/…}; "
    "\\textbf{DFGO (multi, all tasks, text\\_conditioned)} scans \\texttt{text\\_conditioned\\_only/all\\_improved\\_<frac>/} "
    "(override with \\texttt{DUO\\_NMAX\\_MULTITASK\\_PREFIX}); not \\texttt{multi\\_task}. "
    "Best hyperfolder per task is in the table note. "
    "Missing \\texttt{nmax} cells are worst when ranking. "
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
LATEX_LABEL_ALL = "tab:gtg-duo-eval-all-text"


def _run_sweep_w_ablation() -> None:
    """Write ``results/analysis_table/max_ablation.{tex,csv}`` from ``eval_sweep_w_text`` (see ``make_sweep_w_ablation_table.py``)."""
    import importlib.util

    p = Path(__file__).resolve().parent / "make_sweep_w_ablation_table.py"
    spec = importlib.util.spec_from_file_location("_mksweep_w", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    sweep_root = DUO_RESULTS / "eval_sweep_w_text"
    override = os.environ.get("SWEEP_W_MODEL_DIR", "").strip()
    if override:
        root = Path(override).resolve()
    else:
        root = mod.discover_default_model_dir(sweep_root)
    mod.write_max_ablation(root, gtg_results=GTG_RESULTS)


def main(mode: str = "all") -> None:
    if mode == "sweep_w":
        _run_sweep_w_ablation()
        return

    d_best_overrides: dict[str, float] = {}
    if D_BEST_JSON.is_file():
        raw = json.loads(D_BEST_JSON.read_text(encoding="utf-8"))
        d_best_overrides = {str(k): float(v) for k, v in raw.items()}

    uniso_by_task = parse_uniso_best_per_task(UNISO_RESULT_TEX)

    duo = scan_results_root(DUO_RESULTS)
    gtg = scan_results_root(GTG_RESULTS)

    sweep_note = merge_eval_sweep_w_text_into_duo(duo, DUO_RESULTS)
    _swp = _env_compat("DUO_SWEEP_W_PREFIX", "GTGDFGO_SWEEP_W_PREFIX", "")
    if sweep_note and _swp:
        print(
            "eval_sweep_w_text: DUO full multitask text columns use "
            f"w={_env_compat('DUO_SWEEP_W_VALUE', 'GTGDFGO_SWEEP_W_VALUE', '')} "
            f"under {_swp}"
        )

    do_short = mode in ("all", "short")
    do_full = mode in ("all", "full")
    do_final = mode in ("all", "final")

    if do_short:
        out_base = OUTPUT_BASE
        rows = enrich_multitask_columns(
            sort_comparison_rows(build_comparison_rows(duo, gtg)), duo
        )
        tex_path = out_base.with_suffix(".tex")
        out_base.parent.mkdir(parents=True, exist_ok=True)

        write_csv(out_base.with_suffix(".csv"), rows)
        write_latex(
            tex_path,
            DUO_RESULTS,
            gtg,
            duo,
            d_best_overrides,
            caption=LATEX_CAPTION + sweep_note,
            label=LATEX_LABEL,
            uniso_by_task=uniso_by_task,
        )
        write_latex_nmax_design_bench(
            NMAX_TEX,
            DUO_RESULTS,
            gtg,
            duo,
            d_best_overrides,
            UNISO_NRESULT_TEX,
            caption=LATEX_CAPTION_NMAX + sweep_note,
            label=LATEX_LABEL_NMAX,
        )
        print(f"Wrote {out_base.with_suffix('.csv')}")
        print(f"Wrote {tex_path}")
        print(f"Wrote {NMAX_TEX}")
        print(
            f"DUO experiments: {len(duo)}, GTG experiments: {len(gtg)}, comparison rows: {len(rows)}"
        )

    if do_final:
        all_base = TEXT_CONDITIONED_RESULT_ANALYSIS_BASE
        all_base.parent.mkdir(parents=True, exist_ok=True)
        write_latex_all_fulltext(
            all_base.with_suffix(".tex"),
            DUO_RESULTS,
            gtg,
            duo,
            d_best_overrides,
            uniso_by_task,
            caption=LATEX_CAPTION_ALL + sweep_note,
            label=LATEX_LABEL_ALL,
        )
        print(f"Wrote {all_base.with_suffix('.tex')}")

    if do_full:
        mbase = MATRIX12_BASE
        mrows = build_matrix12_rows(
            gtg, duo, MATRIX12_TEXT_SUFFIXES, uniso_by_task
        )
        mbase.parent.mkdir(parents=True, exist_ok=True)
        write_matrix12_csv(mbase.with_suffix(".csv"), mrows)
        write_matrix12_latex(
            mbase.with_suffix(".tex"),
            LATEX_CAPTION_MATRIX12 + sweep_note,
            LATEX_LABEL_MATRIX12,
            gtg,
            duo,
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
        choices=("all", "short", "full", "final", "sweep_w"),
        default="all",
        help=(
            "Which outputs to generate: "
            "short = analysis_table/max_short (wide CSV+TeX) and nmax.tex; "
            "full = analysis_table/max_extended (matrix CSV+TeX); "
            "final = analysis_table/text_conditioned_result_analysis (DFGO: one column per hyper under all_frac…); "
            "sweep_w = analysis_table/max_ablation from eval_sweep_w_text (optional SWEEP_W_MODEL_DIR); "
            "all = short + full + final (default). Text-CFG multi-w comparison is max_ablation.tex (not max.tex)."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(mode=args.mode)

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

所有汇总表与 UniSO 输入均位于 ``results/analysis_table/``：宽表 ``max_short.*``、``text_conditioned_result_analysis.*``、``nmax.tex``，以及 ``w_ablation.*``（text CFG 消融：不同 ``w`` 一列，跨 seed 聚合）、``ce_ablation.*``（CE ablation：两列对比）、``uniso_result.tex``、``uniso_nresult.tex``、``d_best.json``（可选）。
``text_conditioned_result_analysis``（``--mode final``）：``text_conditioned_only/all_frac1.0_sigma0.0/``（可用 ``EVAL_ALL_TASK_FRAC_SIG`` 覆盖）下**每个超参子目录一列** DFGO。默认一次生成全部；也可用 ``--mode short|final``（见 ``run_analyze_eval.sh``）。
``max_short``、``nmax`` 中 DUO「全任务 multitask text」列：若存在 ``results/eval_sweep_w_text/<mt_*>/`` 且 ``DUO_SWEEP_W_DISABLE`` 未置 1，则按 ``w_ablation`` 的准则（UniSO、GTG ST 与各 ``w`` 联合排名后，取平均秩最小的 ``w``）选出最优 ``condition_guidance_w_text``，该列数据来自对应 ``eval_w*.log``（可固定到 CE 目录，见 ``SWEEP_W_DEFAULT_CE_DIRNAME``）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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

# Real-world few-shot benchmark tasks (Design-Bench wrapper + oracle_predict).
REAL_TASK_ORDER: list[str] = [
    "lunar_lander",
    "rover",
    "robot_push",
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
    用于 max_short 的 DFGO fullL、``enrich_multitask_columns`` 的 mfull。
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
    # 避免引入 diffuser 包（其 __init__ 会 import torch）。这里内联关键命名逻辑，
    # 保证仅依赖 stdlib，且与训练脚本的路径签名一致。

    def canonical_train_tasks_csv(train_tasks: str) -> str:
        parts = [t.strip() for t in train_tasks.split(",") if t.strip()]
        if len(parts) <= 1:
            return parts[0] if parts else ""
        return ",".join(sorted(parts))

    def multitask_text_only_path_infix(args: Any) -> str:
        if getattr(args, "multitask_text_only", False):
            return os.environ.get("GTG_MTTEXTONLY_PATH_INFIX", "_mttextonly")
        return ""

    def text_cond_path_infix(args: Any) -> str:
        if getattr(args, "use_text_condition", False) or getattr(
            args, "multitask_text_only", False
        ):
            return os.environ.get("GTG_TEXTCOND_PATH_INFIX", "_textcond")
        return ""

    def returns_cond_path_infix(args: Any) -> str:
        rc = getattr(args, "returns_condition", None)
        ir = getattr(args, "include_returns", None)
        if rc is True and ir is True:
            return os.environ.get("GTG_RETCOND_PATH_INFIX", "_retcond")
        return ""

    def _merge_traj_params_json(
        path: str,
        tasks: list[str],
        n_traj: dict[str, int],
        k: dict[str, int],
        eps: dict[str, float],
    ) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return n_traj, k, eps
        defs = raw.get("defaults") or {}
        if isinstance(defs, dict):
            for t in tasks:
                if "n_traj" in defs:
                    n_traj[t] = int(defs["n_traj"])
                if "k" in defs:
                    k[t] = int(defs["k"])
                if "eps" in defs:
                    eps[t] = float(defs["eps"])
        for key, val in raw.items():
            if key == "defaults" or key not in n_traj:
                continue
            if not isinstance(val, dict):
                continue
            if "n_traj" in val:
                n_traj[key] = int(val["n_traj"])
            if "k" in val:
                k[key] = int(val["k"])
            if "eps" in val:
                eps[key] = float(val["eps"])
        return n_traj, k, eps

    def multitask_trajectory_signature(
        tasks: list[str],
        n_traj: dict[str, int],
        k: dict[str, int],
        eps: dict[str, float],
        horizon: int,
    ) -> str:
        ts = sorted(tasks)
        n0, k0, e0 = n_traj[ts[0]], k[ts[0]], eps[ts[0]]
        uniform = all(
            (n_traj[t] == n0)
            and (k[t] == k0)
            and (abs(eps[t] - e0) <= 1e-9)
            for t in ts
        )
        if uniform:
            return f"{n0}x{horizon}_k{k0}_eps{e0:g}"
        parts: list[str] = []
        for t in ts:
            parts.append(f"{t}_n{n_traj[t]}_k{k[t]}_eps{eps[t]:g}")
        return "ptask__" + "__".join(parts)

    def multitask_slug_id(traj_signature: str) -> str:
        h = hashlib.sha256(traj_signature.encode("utf-8")).hexdigest()[:16]
        return f"mt_{h}"

    def multitask_checkpoint_hyper_dir(
        sig: str, ret_infix: str, text_infix: str, mttextonly_infix: str
    ) -> str:
        return f"{multitask_slug_id(sig)}{ret_infix}{text_infix}{mttextonly_infix}"

    def prepare_multitask_traj(
        tasks_list: list[str],
        n_traj_val: int,
        k_val: int,
        eps_val: float,
        horizon: int,
        traj_params_json: str | None,
    ) -> tuple[dict[str, int], dict[str, int], dict[str, float], str]:
        n_d = {t: int(n_traj_val) for t in tasks_list}
        k_d = {t: int(k_val) for t in tasks_list}
        e_d = {t: float(eps_val) for t in tasks_list}
        if traj_params_json:
            n_d, k_d, e_d = _merge_traj_params_json(
                traj_params_json, tasks_list, n_d, k_d, e_d
            )
        sig = multitask_trajectory_signature(tasks_list, n_d, k_d, e_d, horizon)
        return n_d, k_d, e_d, sig

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
    从 ``results/eval_sweep_w_text/<mt_* 或 sweep 归档目录>/`` 的 ``eval_w*.log`` 按与 ``w_ablation`` **Mean rank 行中各 w 列**
    相同的准则（UniSO、GTG ST 与各 w 联合排名后，取平均秩最小的 ``w``）选出最优 ``w``，
    注入虚拟实验键 ``eval_sweep_w_text/<mt>/w_text<w>``（及 ``…_ret``，数值相同供 +returns 列使用），
    并设置 ``DUO_SWEEP_W_PREFIX``，使 ``duo_all_text_prefix`` / ``all_improved`` / nmax multi text 均指向该前缀。

    可用 ``DUO_SWEEP_W_DISABLE=1`` 关闭；``SWEEP_W_MODEL_DIR`` 指定 ``mt_*`` 目录（否则优先固定 CE 目录，见 ``SWEEP_W_DEFAULT_CE_DIRNAME``）。
    返回空串（不向 caption 追加说明；注入成功后控制台仍可打印选用的 ``w``）。
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
            pinned_ce = (sweep_root / SWEEP_W_DEFAULT_CE_DIRNAME).resolve()
            model_dir = pinned_ce if pinned_ce.is_dir() else mod.discover_default_model_dir(sweep_root)
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
    return ""


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

# 与宽表一致的任务行（``w_ablation`` 等同）；**必须包含 Superconductor**（在 D'Kitty 与 TF Bind 8 之间）。
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


def _nresult_midrule_before_duo(last_baseline_method: str) -> bool:
    """最后一行基线为 UniSO-T 时，其与 DUO 之间需 ``\\midrule``（RaM 行后已有 ``\\midrule``，不再重复）。"""
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
    源表 ``/N`` 表示参与排名的方法数（不含 ``$\\mathcal{D}$(best)`` 行）。
    ``nmax`` 在相同方法集合上增加一行 DUO（全任务 text），故分母 = ``(n_base - 1) + 1``
    （首行为 D(best) 时）；否则 ``n_base + 1``。
    """
    if parsed and r"$\mathcal{D}$(best)" in parsed[0][0]:
        return (n_base - 1) + 1
    return n_base + 1


def nmax_fmt_duo_avg_rank(mu: float, n_total: int) -> str:
    """DUO 行 Avg.\\ Rank：``mean / N``（与基线 ``a / N`` 同一分母 ``N``）。"""
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
    """Avg.\\ Rank：越小越好；``/``、``--`` 跳过；``a / b`` 取第一个数（含 DUO ``m / N``）。"""
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


# --- Shared naming helpers (used by w_ablation and CE ablation) ---

RETCOND_INFIX = "_retcond"


def _single_exp_name(task: str, use_returns: bool) -> str:
    return f"{task}_multiple_runs{RETCOND_INFIX if use_returns else ''}"


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
        r"% Blocks: Task+D(best) | UniSO-T | GTG ST/ST+r | DUO (text_conditioned_only/all_frac…).",
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
    # UniSO + GTG ST + GTG ST+r + 每列 DUO
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
    lines = [
        r"% Requires: \usepackage{booktabs}, \usepackage{graphicx}, \usepackage{xcolor}.",
        r"% Per row: best / runner-up among UniSO-T + GTG + four DUO columns.",
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
    归一化分数：基线来自 ``uniso_nresult.tex``；DUO 行为日志中 ``nmax_ep_reward``
    （与 evaluate 中归一化一致）。
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
    means_all = np.full((n_t, n_base + 1), np.nan, dtype=np.float64)
    for ti in range(n_t):
        for j in range(n_base):
            v = parse_mean_from_text_cell(parsed[j][2][ti])
            means_all[ti, j] = np.nan if v is None else v
        tk = DESIGN_BENCH_TASK_ORDER[ti]
        kfull = best_duo_exp_for_task(
            duo, duo_full_multitask_nmax_prefix(), tk, False
        )
        gfm = _mean_nmax_from_task_stats(
            duo.get(kfull, {}).get(tk) if kfull else None
        )
        means_all[ti, n_base] = np.nan if gfm is None else gfm
    mu_r, _sd_r = column_mean_rank_stats_higher_nan_worst(means_all)
    n_rank_total = nmax_avg_rank_denominator_total(n_base, parsed)
    rank_duo_plain = nmax_fmt_duo_avg_rank(mu_r[n_base], n_rank_total)
    gm_cells: list[str] = []
    for ti, task_key in enumerate(DESIGN_BENCH_TASK_ORDER):
        kfull = best_duo_exp_for_task(
            duo, duo_full_multitask_nmax_prefix(), task_key, False
        )
        gm_cells.append(stats_cell_latex_nmax(duo, kfull or "", task_key))
    # 列方向：先剥掉源表行内着色，再与 DUO 一起按数值重算蓝/紫（任务列越大越好，Avg.\ Rank 越小越好）
    task_cols_plain: list[list[str]] = [[] for _ in range(5)]
    for ti in range(5):
        col: list[str] = []
        for j in range(n_base):
            col.append(latex_cell_strip_wrappers(parsed[j][2][ti]))
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
    avg_plain.append(rank_duo_plain)
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
    if parsed and _nresult_midrule_before_duo(parsed[-1][0]):
        lines.append(r"\midrule")
    lines.append(
        r"\textbf{DUO} (multi, all tasks, text) & / & "
        + " & ".join(task_cols_plain[ti][n_base] for ti in range(5))
        + r" & "
        + avg_plain[n_base]
        + r" \\"
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
        ]
    )
    lines.extend(
        [
            rf"\label{{{label}}}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_trajectory_hyperparams(path: Path) -> None:
    """Standalone table: per-task $n_{\mathrm{traj}}$, $k$, $\varepsilon$ from ``max_short_traj_json_path()``."""
    jp = max_short_traj_json_path()
    if not jp.is_file():
        path.write_text(
            rf"% Missing trajectory JSON: {jp}\n",
            encoding="utf-8",
        )
        return
    raw = json.loads(jp.read_text(encoding="utf-8"))
    defaults: dict[str, Any] = dict(raw.get("defaults") or {})
    keys = [k for k in raw if k != "defaults"]
    ordered: list[str] = []
    for t in TASK_ORDER:
        if t in keys:
            ordered.append(t)
    for k in sorted(keys):
        if k not in ordered:
            ordered.append(k)
    hz = os.environ.get("DUO_MAX_SHORT_HORIZON", "64")
    cap = (
        r"Per-task trajectory hyperparameters ($n_{\mathrm{traj}}$, $k$, $\varepsilon$); "
        rf"diffusion horizon $H={hz}$."
    )
    lines = [
        r"% Requires: \usepackage{booktabs}.",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{cap}}}",
        r"\label{tab:duo-traj-hyperparams}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Task & $n_{\mathrm{traj}}$ & $k$ & $\varepsilon$ \\",
        r"\midrule",
    ]
    for task_key in ordered:
        row = {**defaults, **(raw[task_key] if isinstance(raw[task_key], dict) else {})}
        nt = row.get("n_traj", "")
        kk = row.get("k", "")
        eps = row.get("eps", "")
        name = latex_task_display_name(task_key)
        lines.append(f"{name} & {nt} & {kk} & {eps} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
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
NMAX_TEX = ANALYSIS_TABLE_DIR / "nmax.tex"
REAL_BASE = ANALYSIS_TABLE_DIR / "real"
TRAJECTORY_HYPER_TEX = ANALYSIS_TABLE_DIR / "trajectory_hyperparams.tex"
UNISO_RESULT_TEX = ANALYSIS_TABLE_DIR / "uniso_result.tex"
UNISO_NRESULT_TEX = ANALYSIS_TABLE_DIR / "uniso_nresult.tex"
D_BEST_JSON = ANALYSIS_TABLE_DIR / "d_best.json"
W_ABLATION_TEX = ANALYSIS_TABLE_DIR / "w_ablation.tex"
W_ABLATION_CSV = ANALYSIS_TABLE_DIR / "w_ablation.csv"
CE_ABLATION_TEX = ANALYSIS_TABLE_DIR / "ce_ablation.tex"
CE_ABLATION_CSV = ANALYSIS_TABLE_DIR / "ce_ablation.csv"

# 固定 sweep-w 的两个目录名（相对 results/eval_sweep_w_text/）
SWEEP_W_DEFAULT_BASE_DIRNAME = (
    "mt_911054c35daad7e0_textcond_mttextonly_tsbias0.5_lr0.0002"
)
SWEEP_W_DEFAULT_CE_DIRNAME = (
    "mt_911054c35daad7e0_textcond_mttextonly_ce0.005_tsbias0.5_lr0.0002"
)

LATEX_CAPTION = (
    "Un-normalized \\texttt{max\\_ep\\_reward} (mean $\\pm$ std over runs): "
    "UniSO-T (Improved), GTG, and DUO (single / single+text / multitask label / multitask text). "
    "Mean rank averages per-task ranks (higher reward is better). "
    "$\\mathcal{D}$(best) uses offline train subset optima (optional overrides from \\texttt{d\\_best.json})."
)
LATEX_LABEL = "tab:gtg-duo-eval"
LATEX_CAPTION_NMAX = (
    "Normalized Design-Bench scores (baselines aligned with \\texttt{uniso\\_nresult.tex}); "
    "DUO reports multitask text-conditioned \\texttt{nmax\\_ep\\_reward} (mean $\\pm$ std). "
    "Avg.\\ Rank is mean per-task rank divided by pool size~$N$."
)
LATEX_LABEL_NMAX = "tab:duo-nmax-designbench"
TEXT_CONDITIONED_RESULT_ANALYSIS_BASE = (
    ANALYSIS_TABLE_DIR / "text_conditioned_result_analysis"
)
_LATEX_ESC_ALL_SIG = EVAL_ALL_TASK_FRAC_SIG.replace("_", r"\_")
LATEX_CAPTION_ALL = (
    "Full multitask text-conditioned evaluation: one column per hyperparameter subfolder under "
    f"\\texttt{{text\\_conditioned\\_only/{_LATEX_ESC_ALL_SIG}/}}. "
    "UniSO-T (Improved), GTG ST / ST+r, and DUO. Mean rank uses higher reward as better."
)
LATEX_LABEL_ALL = "tab:gtg-duo-eval-all-text"


def _run_w_ablation() -> None:
    """Write ``results/analysis_table/w_ablation.{tex,csv}`` from ``eval_sweep_w_text`` (see ``make_sweep_w_ablation_table.py``)."""
    import importlib.util

    p = Path(__file__).resolve().parent / "make_sweep_w_ablation_table.py"
    spec = importlib.util.spec_from_file_location("_mksweep_w", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    sweep_root = DUO_RESULTS / "eval_sweep_w_text"
    root = (sweep_root / SWEEP_W_DEFAULT_BASE_DIRNAME).resolve()
    if not root.is_dir():
        root = mod.discover_default_model_dir(sweep_root)
    mod.write_w_ablation(root, gtg_results=GTG_RESULTS)


def write_ce_ablation(
    out_tex: Path,
    out_csv: Path,
    base_model_dir: Path,
    ce_model_dir: Path,
) -> None:
    """
    CE ablation: compare two sweep-w directories (max only).

    Each column selects its own best ``w`` using the same criterion as ``w_ablation``:
    ranks are assigned per task jointly over (UniSO, GTG ST, all w), then averaged across tasks
    to pick the minimal mean-rank ``w``.

    Output is a simple 2-column table with mean±std and a Mean rank row (no coloring).
    """
    import importlib.util

    mod_path = Path(__file__).resolve().parent / "make_sweep_w_ablation_table.py"
    spec = importlib.util.spec_from_file_location("_sw_ce", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    base_model_dir = base_model_dir.resolve()
    ce_model_dir = ce_model_dir.resolve()
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    def _collect_one(model_dir: Path) -> tuple[float, dict[tuple[str, float], list[float]]]:
        values, w_list = mod.collect_sweep_max_by_task_from_logs(model_dir)
        if not w_list:
            raise FileNotFoundError(f"no eval_w*.log under {model_dir}")
        best_w = mod.sweep_best_w_match_max_ablation(model_dir, GTG_RESULTS)
        if best_w is None:
            best_w = w_list[0]
        return float(best_w), values

    w_base, v_base = _collect_one(base_model_dir)
    w_ce, v_ce = _collect_one(ce_model_dir)

    # rows: MAX_TEX_TASK_ROWS order; show max(mean±std) pooled over seeds at best w.
    rows_csv: list[list[str]] = []
    rows_csv.append(["Task", "base", "ce0.005"])

    lines_tex: list[str] = []
    lines_tex.append(r"% Requires: \usepackage{booktabs}.")
    lines_tex.append(r"\begin{table}[t]")
    lines_tex.append(
        r"\caption{CE ablation on multitask text-only: max\_ep\_reward at each column's best $w$ (mean $\pm$ std over seeds).}"
    )
    lines_tex.append(r"\label{tab:ce-ablation}")
    lines_tex.append(r"\centering")
    lines_tex.append(r"\small")
    lines_tex.append(r"\begin{tabular}{lcc}")
    lines_tex.append(r"\toprule")
    lines_tex.append(rf"Task & base ($w={w_base:g}$) & ce0.005 ($w={w_ce:g}$) \\")
    lines_tex.append(r"\midrule")

    means_m = np.full((len(MAX_TEX_TASK_ROWS), 2), np.nan, dtype=np.float64)
    for i, (task_key, latex_name, _exp_name) in enumerate(MAX_TEX_TASK_ROWS):
        xs_b = v_base.get((task_key, w_base), [])
        xs_c = v_ce.get((task_key, w_ce), [])
        if xs_b:
            mu_b, sd_b = mean_std([float(x) for x in xs_b])
            cell_b = fmt_pm_latex(mu_b, sd_b)
            means_m[i, 0] = float(mu_b)
        else:
            cell_b = "--"
        if xs_c:
            mu_c, sd_c = mean_std([float(x) for x in xs_c])
            cell_c = fmt_pm_latex(mu_c, sd_c)
            means_m[i, 1] = float(mu_c)
        else:
            cell_c = "--"

        lines_tex.append(f"{latex_name} & {cell_b} & {cell_c} \\\\")
        rows_csv.append(
            [
                latex_name,
                cell_b.replace("$", "").replace(r"\pm", "±"),
                cell_c.replace("$", "").replace(r"\pm", "±"),
            ]
        )

    mu_r, sd_r = column_mean_rank_stats(means_m)
    mean_rank_cells = [fmt_mean_pm_rank(mu_r[j], sd_r[j]) for j in range(2)]
    lines_tex.append(r"\midrule")
    lines_tex.append("Mean rank & " + " & ".join(mean_rank_cells) + r" \\")
    lines_tex.append(r"\bottomrule")
    lines_tex.append(r"\end{tabular}")
    lines_tex.append(r"\end{table}")
    lines_tex.append("")

    out_tex.write_text("\n".join(lines_tex), encoding="utf-8")

    # CSV mean rank row (no LaTeX math).
    rows_csv.append(
        [
            "Mean rank",
            mean_rank_cells[0].replace("$", "").replace(r"\pm", "±"),
            mean_rank_cells[1].replace("$", "").replace(r"\pm", "±"),
        ]
    )
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows_csv)


def _collect_best_real_world_nmax(
    results_root: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Scan ``results/real_task/{zero_shot,few_shot}/**/run*_seed*/evaluate.log`` and aggregate
    normalized metric ``nmax_ep_reward`` per task.

    Returns:
      out[mode][task] = {
        "nmax_mean": float, "nmax_std": float, "runs": int,
        "best_hyper": str, "best_hyper_mean": float,
      }
    where ``best_hyper`` is the relative folder under ``<task>_frac*_sigma*/`` (few-shot may
    contain an extra pool folder like ``fs_k128_worst/``).
    """
    root = results_root / "real_task"
    if not root.is_dir():
        return {}

    # mode -> task -> hyper_rel -> [nmax]
    buckets: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for evaluate_log in root.rglob("evaluate.log"):
        if not _ALL_RUN_DIR_RE.match(evaluate_log.parent.name):
            continue
        # expected layout:
        #   real_task/<mode>/<task>_frac*_sigma*/<hyper...>/run*/evaluate.log
        # mode is the folder right under real_task/
        try:
            rel = evaluate_log.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 5:
            continue
        mode = parts[0]
        task_dir = parts[1]
        task = task_dir.split("_frac", 1)[0].strip()
        if not task:
            continue
        # hyper id = join(parts[2:-2]) to drop run*/evaluate.log
        hyper_rel = "/".join(parts[2:-2]).strip() or "."

        d = parse_evaluate_log(evaluate_log)
        if task not in d:
            # Some real-world logs might be plain (no [task] prefix). Fall back to first entry.
            if len(d) == 1:
                (_, v) = next(iter(d.items()))
            else:
                continue
        else:
            v = d[task]
        nmax = float(v[1])
        if not np.isfinite(nmax):
            continue
        buckets[mode][task][hyper_rel].append(nmax)

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for mode, by_task in buckets.items():
        out[mode] = {}
        for task, by_hyper in by_task.items():
            best_h = ""
            best_mu = float("-inf")
            best_sd = float("nan")
            best_n = 0
            for h, vals in by_hyper.items():
                mu, sd = mean_std(vals)
                if not np.isfinite(mu):
                    continue
                if mu > best_mu:
                    best_mu, best_sd, best_n, best_h = mu, sd, len(vals), h
            if best_h:
                out[mode][task] = {
                    "nmax_mean": best_mu,
                    "nmax_std": best_sd,
                    "runs": best_n,
                    "best_hyper": best_h,
                    "best_hyper_mean": best_mu,
                }
    return out


def write_real_world_tables(
    out_base: Path,
    results_root: Path,
    caption: str = "Real-world tasks (normalized score) for zero-shot and few-shot.",
    label: str = "tab:duo-real-world",
) -> None:
    """
    Write ``real.csv`` + ``real.tex`` under ``results/analysis_table/``.

    CSV columns:
      task, zero_shot_nmax_mean, zero_shot_nmax_std, few_shot_nmax_mean, few_shot_nmax_std,
      zero_shot_best_hyper, few_shot_best_hyper
    """
    agg = _collect_best_real_world_nmax(results_root)
    zs = agg.get("zero_shot", {})
    fs = agg.get("few_shot", {})

    tasks = [t for t in REAL_TASK_ORDER if t in zs or t in fs]
    # append any extra tasks found
    extra = sorted((set(zs) | set(fs)) - set(tasks))
    tasks.extend(extra)

    rows: list[dict[str, Any]] = []
    for t in tasks:
        z = zs.get(t)
        f = fs.get(t)
        rows.append(
            {
                "task": t,
                "zero_shot_nmax_mean": "" if z is None else z.get("nmax_mean", ""),
                "zero_shot_nmax_std": "" if z is None else z.get("nmax_std", ""),
                "few_shot_nmax_mean": "" if f is None else f.get("nmax_mean", ""),
                "few_shot_nmax_std": "" if f is None else f.get("nmax_std", ""),
                "zero_shot_best_hyper": "" if z is None else z.get("best_hyper", ""),
                "few_shot_best_hyper": "" if f is None else f.get("best_hyper", ""),
                "zero_shot_runs": "" if z is None else z.get("runs", ""),
                "few_shot_runs": "" if f is None else f.get("runs", ""),
            }
        )

    out_base.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_base.with_suffix(".csv"), rows)

    # Minimal LaTeX table (no colorization; normalized scores may be negative depending on task wrapper).
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Task & Zero-shot (nmax) & Few-shot (nmax) \\")
    lines.append(r"\midrule")
    for r in rows:
        t = str(r["task"])
        t_esc = t.replace("_", r"\_")
        zs_cell = fmt_pm_latex(r["zero_shot_nmax_mean"], r["zero_shot_nmax_std"])
        fs_cell = fmt_pm_latex(r["few_shot_nmax_mean"], r["few_shot_nmax_std"])
        lines.append(rf"{t_esc} & {zs_cell} & {fs_cell} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_base.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(mode: str = "all") -> None:
    if mode == "w_ablation":
        _run_w_ablation()
        return

    d_best_overrides: dict[str, float] = {}
    if D_BEST_JSON.is_file():
        raw = json.loads(D_BEST_JSON.read_text(encoding="utf-8"))
        d_best_overrides = {str(k): float(v) for k, v in raw.items()}

    uniso_by_task = parse_uniso_best_per_task(UNISO_RESULT_TEX)

    duo = scan_results_root(DUO_RESULTS)
    gtg = scan_results_root(GTG_RESULTS)

    merge_eval_sweep_w_text_into_duo(duo, DUO_RESULTS)
    _swp = _env_compat("DUO_SWEEP_W_PREFIX", "GTGDFGO_SWEEP_W_PREFIX", "")
    _wv = _env_compat("DUO_SWEEP_W_VALUE", "GTGDFGO_SWEEP_W_VALUE", "")
    if _swp.strip():
        print(f"eval_sweep_w_text: DUO multitask text uses w={_wv} under {_swp}")

    do_short = mode in ("all", "short")
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
            caption=LATEX_CAPTION,
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
            caption=LATEX_CAPTION_NMAX,
            label=LATEX_LABEL_NMAX,
        )
        write_real_world_tables(
            REAL_BASE,
            DUO_RESULTS,
            caption="Real-world tasks (normalized score) for zero-shot and few-shot. "
            "Each cell is mean $\\pm$ std over seeds for the best hyperfolder under results/real_task.",
            label="tab:duo-real-world",
        )
        print(f"Wrote {out_base.with_suffix('.csv')}")
        print(f"Wrote {tex_path}")
        write_latex_trajectory_hyperparams(TRAJECTORY_HYPER_TEX)
        print(f"Wrote {NMAX_TEX}")
        print(f"Wrote {REAL_BASE.with_suffix('.tex')}")
        print(f"Wrote {TRAJECTORY_HYPER_TEX}")
        print(
            f"DUO experiments: {len(duo)}, GTG experiments: {len(gtg)}, comparison rows: {len(rows)}"
        )
        # w_ablation + ce_ablation are small and cheap; keep in short/all.
        try:
            _run_w_ablation()
        except Exception as e:
            print(f"[warn] w_ablation skipped: {e}")
        try:
            write_ce_ablation(
                CE_ABLATION_TEX,
                CE_ABLATION_CSV,
                DUO_RESULTS / "eval_sweep_w_text" / SWEEP_W_DEFAULT_BASE_DIRNAME,
                DUO_RESULTS / "eval_sweep_w_text" / SWEEP_W_DEFAULT_CE_DIRNAME,
            )
        except Exception as e:
            print(f"[warn] ce_ablation skipped: {e}")

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
            caption=LATEX_CAPTION_ALL,
            label=LATEX_LABEL_ALL,
        )
        print(f"Wrote {all_base.with_suffix('.tex')}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate evaluate.log and write tables under results/analysis_table/ (see README)."
    )
    p.add_argument(
        "--mode",
        choices=("all", "short", "final", "w_ablation"),
        default="all",
        help=(
            "Which outputs to generate: "
            "short = analysis_table/max_short (wide CSV+TeX), nmax.tex, and trajectory_hyperparams.tex; "
            "final = analysis_table/text_conditioned_result_analysis (DUO: one column per hyper under all_frac…); "
            "w_ablation = analysis_table/w_ablation from eval_sweep_w_text (pinned mt dir when available); "
            "all = short + final (default)."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(mode=args.mode)

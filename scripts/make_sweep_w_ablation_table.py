#!/usr/bin/env python3
"""
从 ``eval_sweep_w_text/<实验目录>/seed*/eval_w*.log`` 汇总 text CFG 消融（目录可为历史 ``mt_*`` 或
``train_eval_sweep_w_text.sh`` 生成的参数 slug）：
**同一 mt_* 目录下所有 seed** 的 ``eval_w*.log`` 全部参与聚合；每个 ``w`` 一列，格内为
该 (task, w) 在所有可用 seed 上的 **max_ep_reward 的 mean ± std**。

兼容旧布局：``<mt_*>/run<N>_seed<SEED>/``（无平铺 ``seed*`` 时仍扫描）、以及 ``run_<时间戳>/seed*/``。

输出：``results/analysis_table/w_ablation.tex``、``w_ablation.csv``。

用法::

    python scripts/make_sweep_w_ablation_table.py
    python scripts/make_sweep_w_ablation_table.py --model-dir results/eval_sweep_w_text/mt_xxx_textcond_mttextonly

默认优先使用固定目录 ``mt_911054c35daad7e0_textcond_mttextonly_tsbias0.5_lr0.0002``（若存在）；
否则在 ``results/eval_sweep_w_text/`` 下选取 **最近修改的** ``mt_*`` 目录；若无则选含 ``seed*/eval_w*.log`` 的最新实验目录；
再否则回退到最新 ``run_*`` 旧目录。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_W_ABLATION_DIRNAME = (
    "mt_911054c35daad7e0_textcond_mttextonly_tsbias0.5_lr0.0002"
)


def _load_analyze():
    p = _ROOT / "scripts" / "analyze_eval_results.py"
    spec = importlib.util.spec_from_file_location("analyze_eval_results", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


ae = _load_analyze()

# Support both:
# - eval_w8p0.log (legacy)
# - eval_w8p0_1400epochs.log (checkpoint sweep; legacy 750epochs also supported)
_EVAL_LOG_RE = re.compile(r"^eval_w([^_]+)(?:_(\d+)epochs)?\.log$")
# 推荐：<mt_*>/seed0/eval_w*.log
_RE_SEED_DIR = re.compile(r"^seed\d+$")
# 曾用：<mt_*>/run0_seed0/
_RE_RUN_SEED_BATCH = re.compile(r"^run(\d+)_seed(\d+)$")
_RE_RUN_LEGACY = re.compile(r"^run_\d{8}_\d{6}$")


def w_from_eval_log_name(name: str) -> float:
    m = _EVAL_LOG_RE.match(name)
    if not m:
        raise ValueError(f"unexpected eval log name: {name}")
    return float(m.group(1).replace("p", "."))


def _desired_epoch_suffix() -> str:
    """
    Optional selection of checkpoint-eval logs by epoch suffix.

    If DUO_EVAL_LOG_EPOCHS is set (e.g. "1400"), prefer files like:
      eval_w8p0_1400epochs.log
    """
    return os.environ.get("DUO_EVAL_LOG_EPOCHS", "").strip()


def _eval_log_patterns() -> list[str]:
    ep = _desired_epoch_suffix()
    if ep:
        return [f"eval_w*_{ep}epochs.log", "eval_w*.log"]
    return ["eval_w*.log"]


def _iter_eval_logs_one_dir(d: Path) -> list[Path]:
    """
    List eval logs under one directory (seed dir or legacy dir).

    If DUO_EVAL_LOG_EPOCHS is set and any *_<E>epochs.log exists, return ONLY those.
    Otherwise return eval_w*.log (legacy) only.
    """
    pats = _eval_log_patterns()
    if len(pats) == 1:
        return sorted(d.glob(pats[0]))
    # epoch-specific first
    ep_logs = sorted(d.glob(pats[0]))
    if ep_logs:
        return ep_logs
    return sorted(d.glob(pats[1]))


def iter_eval_w_logs(model_or_legacy_dir: Path):
    """遍历目录下所有 ``eval_w*.log``（平铺 seed* / 旧 run*_seed* / 顶层 run_时间戳）。"""
    if not model_or_legacy_dir.is_dir():
        return
    children = [c for c in model_or_legacy_dir.iterdir() if c.is_dir()]
    # 若已有平铺的 seed0、seed1…，只扫这些（避免与历史 runN_seedM 重复计数）；迁移时请删掉旧 run*_seed*。
    has_flat_seed = any(_RE_SEED_DIR.match(c.name) for c in children)
    if has_flat_seed:
        for child in sorted(children):
            if _RE_SEED_DIR.match(child.name):
                yield from _iter_eval_logs_one_dir(child)
        return
    for child in sorted(children):
        if _RE_RUN_SEED_BATCH.match(child.name):
            yield from _iter_eval_logs_one_dir(child)
        elif _RE_RUN_LEGACY.match(child.name):
            for sd in sorted(child.glob("seed*")):
                if sd.is_dir():
                    yield from _iter_eval_logs_one_dir(sd)


def collect_sweep_max_by_task_from_logs(
    root: Path,
) -> tuple[dict[tuple[str, float], list[float]], list[float]]:
    values: dict[tuple[str, float], list[float]] = defaultdict(list)
    w_set: set[float] = set()
    n_logs = 0
    for logf in iter_eval_w_logs(root):
        try:
            w = w_from_eval_log_name(logf.name)
        except ValueError:
            continue
        w_set.add(w)
        parsed = ae.parse_evaluate_log(logf)
        if not parsed:
            continue
        n_logs += 1
        for task_key, (mx, _nm) in parsed.items():
            values[(task_key, w)].append(float(mx))
    if n_logs == 0:
        # 兼容：root 下直接放 eval_w（不推荐）
        for logf in _iter_eval_logs_one_dir(root):
            try:
                w = w_from_eval_log_name(logf.name)
            except ValueError:
                continue
            w_set.add(w)
            parsed = ae.parse_evaluate_log(logf)
            if not parsed:
                continue
            for task_key, (mx, _nm) in parsed.items():
                values[(task_key, w)].append(float(mx))
    w_list = sorted(w_set)
    return values, w_list


def collect_sweep_max_nmax_lists(
    root: Path,
) -> tuple[
    dict[tuple[str, float], list[float]],
    dict[tuple[str, float], list[float]],
    list[float],
]:
    """与 :func:`collect_sweep_max_by_task_from_logs` 相同遍历，但同时收集 ``max`` 与 ``nmax``（供注入汇总表）。"""
    max_v: dict[tuple[str, float], list[float]] = defaultdict(list)
    nmax_v: dict[tuple[str, float], list[float]] = defaultdict(list)
    w_set: set[float] = set()
    n_logs = 0

    def _consume(logf: Path) -> None:
        nonlocal n_logs
        try:
            w = w_from_eval_log_name(logf.name)
        except ValueError:
            return
        w_set.add(w)
        parsed = ae.parse_evaluate_log(logf)
        if not parsed:
            return
        n_logs += 1
        for task_key, (mx, nm) in parsed.items():
            max_v[(task_key, w)].append(float(mx))
            nmax_v[(task_key, w)].append(float(nm))

    for logf in iter_eval_w_logs(root):
        _consume(logf)
    if n_logs == 0:
        for logf in _iter_eval_logs_one_dir(root):
            _consume(logf)

    w_list = sorted(w_set)
    return max_v, nmax_v, w_list


def sweep_best_w_mean_rank(
    max_v: dict[tuple[str, float], list[float]],
    w_list: list[float],
    task_keys: Sequence[str],
) -> float | None:
    """
    仅在各 ``w`` 列之间算平均秩后取最优（**不**含 UniSO / GTG 基线）。

    注入 ``nmax`` / ``max_extended`` 时请用 :func:`sweep_best_w_match_max_ablation`，
    与 ``max_ablation`` 表 Mean rank 行中 **w 列** 的数值一致（基线共同参与排名）。
    """
    if not w_list or not task_keys:
        return None
    if len(w_list) == 1:
        return w_list[0]
    n_t, n_w = len(task_keys), len(w_list)
    means = np.full((n_t, n_w), np.nan, dtype=np.float64)
    for wi, w in enumerate(w_list):
        for ti, tk in enumerate(task_keys):
            xs = max_v.get((tk, w), [])
            if xs:
                means[ti, wi] = float(np.mean(xs))
    mu_r, _ = ae.column_mean_rank_stats(means)
    best_j: int | None = None
    best_mu = float("inf")
    for j in range(n_w):
        m = mu_r[j]
        if np.isfinite(m) and float(m) < best_mu:
            best_mu = float(m)
            best_j = j
    if best_j is None:
        return w_list[0]
    return w_list[best_j]


def build_max_ablation_means_matrix(
    model_dir: Path,
    gtg_results: Path | None = None,
) -> tuple[dict[tuple[str, float], list[float]], list[float], np.ndarray]:
    """
    与 ``write_max_ablation`` 中 **Mean rank** 所用矩阵一致：每行对应 ``MAX_TEX_TASK_ROWS`` 一任务；
    列为 ``[UniSO mean, GTG ST mean, w_1 mean, …]``，各 ``w`` 的 mean 为 **该 (task, w) 下所有 seed**
    的 ``max_ep_reward`` 均值。
    """
    gtg_results = gtg_results or (_ROOT.parent / "GTG" / "results")
    model_dir = model_dir.resolve()
    values, w_list = collect_sweep_max_by_task_from_logs(model_dir)
    if not w_list:
        raise ValueError("no eval_w metrics under model_dir")
    uniso_by_task = ae.parse_uniso_best_per_task(ae.UNISO_RESULT_TEX)
    gtg = ae.scan_results_root(gtg_results.resolve())
    n_w = len(w_list)
    n_rows = len(ae.MAX_TEX_TASK_ROWS)
    means_a = np.full((n_rows, 2 + n_w), np.nan, dtype=np.float64)
    for ri, (task_key, _latex_name, _exp_name) in enumerate(ae.MAX_TEX_TASK_ROWS):
        u_raw = uniso_by_task.get(task_key, "")
        u_cell = u_raw if u_raw else "--"
        col_vals: list[float | None] = [
            ae.parse_mean_from_text_cell(u_cell),
            ae._mean_from_task_stats(
                gtg.get(ae._single_exp_name(task_key, False), {}).get(task_key)
            ),
        ]
        for w in w_list:
            key = (task_key, w)
            xs = values.get(key, [])
            if not xs:
                col_vals.append(None)
            else:
                mu, _ = ae.mean_std(xs)
                col_vals.append(mu)
        means_a[ri, :] = [np.nan if v is None else v for v in col_vals]
    return values, w_list, means_a


def sweep_best_w_match_max_ablation(
    model_dir: Path,
    gtg_results: Path | None = None,
) -> float | None:
    """
    与 ``max_ablation`` 表 **Mean rank** 行中各 ``w`` 列一致：先在每个任务上对 **UniSO、GTG ST、
    全部 w** 联合赋秩，再对 **w 列** 取跨任务平均秩，选 **平均秩最小** 的 ``w``。
    """
    try:
        _values, w_list, means_a = build_max_ablation_means_matrix(
            model_dir, gtg_results=gtg_results
        )
    except ValueError:
        return None
    if len(w_list) == 1:
        return w_list[0]
    mu_r, _ = ae.column_mean_rank_stats(means_a)
    n_w = len(w_list)
    best_k: int | None = None
    best_mu = float("inf")
    for k in range(n_w):
        j = 2 + k
        m = mu_r[j]
        if np.isfinite(m) and float(m) < best_mu:
            best_mu = float(m)
            best_k = k
    if best_k is None:
        return w_list[0]
    return w_list[best_k]


def build_sweep_injected_task_stats(
    max_v: dict[tuple[str, float], list[float]],
    nmax_v: dict[tuple[str, float], list[float]],
    w: float,
    task_keys: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """构造与 :func:`scan_results_root` 单实验键下一致的任务统计（用于注入 ``duo``）。"""
    out: dict[str, dict[str, Any]] = {}
    for tk in task_keys:
        mxs = max_v.get((tk, w), [])
        if not mxs:
            continue
        mm, ms = ae.mean_std(mxs)
        nms = nmax_v.get((tk, w), [])
        if nms:
            nm, ns = ae.mean_std(nms)
        else:
            nm, ns = float("nan"), float("nan")
        out[tk] = {
            "n_runs": len(mxs),
            "max_mean": mm,
            "max_std": ms,
            "nmax_mean": nm,
            "nmax_std": ns,
            "runs": [],
        }
    return out


def discover_default_model_dir(sweep_root: Path) -> Path:
    """优先 ``mt_*``；其次含 ``seed*/eval_w*.log`` 的 sweep 归档目录；否则 ``run_<时间戳>`` 旧目录。"""
    if not sweep_root.is_dir():
        raise FileNotFoundError(f"未找到: {sweep_root}")
    pinned = (sweep_root / _DEFAULT_W_ABLATION_DIRNAME).resolve()
    if pinned.is_dir():
        return pinned
    mt_dirs = sorted(
        [p for p in sweep_root.iterdir() if p.is_dir() and p.name.startswith("mt_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if mt_dirs:
        return mt_dirs[0].resolve()

    flat_sweep: list[tuple[float, Path]] = []
    for p in sweep_root.iterdir():
        if not p.is_dir() or p.name.startswith(".") or p.name.startswith("mt_"):
            continue
        if _RE_RUN_LEGACY.match(p.name):
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        for sd in p.iterdir():
            if sd.is_dir() and _RE_SEED_DIR.match(sd.name) and any(
                _iter_eval_logs_one_dir(sd)
            ):
                flat_sweep.append((mtime, p))
                break
    if flat_sweep:
        flat_sweep.sort(key=lambda t: t[0], reverse=True)
        return flat_sweep[0][1].resolve()

    legacy = sorted(
        [
            p
            for p in sweep_root.iterdir()
            if p.is_dir() and _RE_RUN_LEGACY.match(p.name)
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if legacy:
        return legacy[0].resolve()
    raise FileNotFoundError(
        f"{sweep_root} 下无 mt_*、无含 seed*/eval_w*.log 的目录、也无 run_YYYYMMDD_HHMMSS；请指定 --model-dir"
    )


def write_w_ablation(
    model_dir: Path,
    out_tex: Path | None = None,
    out_csv: Path | None = None,
    gtg_results: Path | None = None,
) -> None:
    out_tex = out_tex or (ae.ANALYSIS_TABLE_DIR / "w_ablation.tex")
    out_csv = out_csv or (ae.ANALYSIS_TABLE_DIR / "w_ablation.csv")
    gtg_results = gtg_results or (_ROOT.parent / "GTG" / "results")

    model_dir = model_dir.resolve()
    try:
        values, w_list, means_a = build_max_ablation_means_matrix(model_dir, gtg_results)
    except ValueError:
        raise SystemExit(
            f"未在 {model_dir} 下解析到任何 eval_w*.log 的指标；请确认已跑评估。"
        )

    d_best_overrides: dict[str, float] = {}
    if ae.D_BEST_JSON.is_file():
        raw = json.loads(ae.D_BEST_JSON.read_text(encoding="utf-8"))
        d_best_overrides = {str(k): float(v) for k, v in raw.items()}

    uniso_by_task = ae.parse_uniso_best_per_task(ae.UNISO_RESULT_TEX)
    gtg = ae.scan_results_root(gtg_results.resolve())

    ae.ANALYSIS_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    n_w = len(w_list)
    tab_spec = "l|c|c|c|" + ("c" * n_w)
    w_headers = " & ".join(rf"\shortstack{{\texttt{{w={str(w)}}}}}" for w in w_list)

    try:
        rel = model_dir.relative_to(_ROOT)
    except ValueError:
        rel = model_dir
    rel_tex = str(rel).replace("_", r"\_")
    hdr_row = (
        r"Task & $\mathcal{D}$(best) & \shortstack{UniSO-T\\Improved} & \shortstack{GTG\\ST} & "
        + w_headers
        + r" \\"
    )

    rows_tex: list[str] = [
        r"% Requires: \usepackage{booktabs}, \usepackage{graphicx}, \usepackage{xcolor}.",
        r"% w_ablation: pooled over all seed* (and legacy run*_seed* / run_*/seed*) under model dir; see scripts/make_sweep_w_ablation_table.py",
        r"\begin{table*}[t!]",
        rf"\caption{{Text CFG ablation (\texttt{{condition\_guidance\_w\_text}}): mean $\pm$ std of \texttt{{max\_ep\_reward}} pooled over all \texttt{{seed*}} runs under \texttt{{{rel_tex}}}. "
        r"Baselines: $\mathcal{D}$(best), UniSO-T Improved, GTG ST.}}",
        r"\vspace{0.3em}",
        r"\centering",
        r"\resizebox{\linewidth}{!}{",
        rf"\begin{{tabular}}{{{tab_spec}}}",
        r"\toprule",
        hdr_row,
        r"\midrule",
    ]

    csv_rows: list[list[str]] = []
    csv_header = ["Task", "D_best", "UniSO-T", "GTG_ST"] + [f"w={w}" for w in w_list]
    csv_rows.append(csv_header)

    task_rows = ae.MAX_TEX_TASK_ROWS
    n_rows = len(task_rows)
    n_rank_cols = 2 + n_w

    for ri, (task_key, latex_name, _exp_name) in enumerate(task_rows):
        db = ae.d_best_cell(ae.DUO_RESULTS, task_key, d_best_overrides)
        u_raw = uniso_by_task.get(task_key, "")
        u_cell = u_raw if u_raw else "--"
        gtg_st = ae.stats_cell_latex(gtg, ae._single_exp_name(task_key, False), task_key)

        w_cells: list[str] = []
        for w in w_list:
            key = (task_key, w)
            xs = values.get(key, [])
            if not xs:
                w_cells.append("--")
            else:
                mu, sd = ae.mean_std(xs)
                w_cells.append(ae.fmt_pm_latex(mu, sd))

        row_body = [u_cell, gtg_st] + w_cells
        row_body = ae.rank_colorize_latex_cells(row_body)
        rows_tex.append(f"{latex_name} & {db} & " + " & ".join(row_body) + r" \\")

        csv_rows.append(
            [latex_name, str(db), u_raw or "", gtg_st.replace("$", "").replace("\\pm", "±")]
            + [c.replace("$", "").replace(r"\pm", "±") for c in w_cells]
        )

    mu_r, sd_r = ae.column_mean_rank_stats(means_a)
    rank_parts = [ae.fmt_mean_pm_rank(mu_r[j], sd_r[j]) for j in range(n_rank_cols)]
    rank_parts = ae.rank_colorize_latex_mean_rank_row(rank_parts)
    rows_tex.extend(
        [
            r"\midrule",
            "Mean rank & / & " + " & ".join(rank_parts) + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            rf"\label{{tab:w-ablation-sweep-w}}",
            r"\end{table*}",
            "",
        ]
    )
    out_tex.write_text("\n".join(rows_tex), encoding="utf-8")

    rank_csv_cells = [ae.fmt_mean_pm_rank(mu_r[j], sd_r[j]) for j in range(n_rank_cols)]
    csv_rows.append(["Mean rank", "/"] + rank_csv_cells)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerows(csv_rows)

    print(f"Wrote {out_tex}")
    print(f"Wrote {out_csv}")
    print(f"Source model_dir: {model_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="w_ablation：sweep_w 跨 seed 聚合")
    ap.add_argument(
        "--sweep-root",
        type=Path,
        default=_ROOT / "results" / "eval_sweep_w_text",
        help="eval_sweep_w_text 根目录",
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="指定 ``mt_*_textcond_mttextonly`` 目录；默认在 sweep-root 下自动选最新 mt_*",
    )
    ap.add_argument(
        "--gtg-results",
        type=Path,
        default=_ROOT.parent / "GTG" / "results",
        help="GTG results 根目录",
    )
    args = ap.parse_args()

    if args.model_dir is not None:
        root = args.model_dir.resolve()
    else:
        root = discover_default_model_dir(args.sweep_root.resolve())

    write_w_ablation(root, gtg_results=args.gtg_results)


if __name__ == "__main__":
    main()

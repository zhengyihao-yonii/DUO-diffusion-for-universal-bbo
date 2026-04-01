#!/usr/bin/env python3
"""
Aggregate evaluate.log metrics across runs (run*_seed* / run*) per experiment,
then compare GTGdfgo vs GTG results in one report (CSV + Markdown).

Metrics: max_ep_reward -> max, nmax_ep_reward -> nmax (mean ± std over runs).
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

BRACKET_MAX = re.compile(
    r"\[([^\]]+)\]\s+max_ep_reward:\s*([-\d.eE+]+(?:e[-+]?\d+)?|nan|inf)"
)
BRACKET_NMAX = re.compile(
    r"\[([^\]]+)\]\s+nmax_ep_reward:\s*([-\d.eE+]+(?:e[-+]?\d+)?|nan|inf)"
)
# (?<!n) avoids matching the suffix "max_ep_reward" inside "nmax_ep_reward"
PLAIN_MAX = re.compile(
    r"(?<!n)max_ep_reward:\s*([-\d.eE+]+(?:e[-+]?\d+)?|nan|inf)"
)
PLAIN_NMAX = re.compile(
    r"nmax_ep_reward:\s*([-\d.eE+]+(?:e[-+]?\d+)?|nan|inf)"
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


def scan_results_root(results_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Returns:
      experiment_name -> task -> {
        'n_runs': int,
        'max_mean', 'max_std', 'nmax_mean', 'nmax_std',
        'runs': [ { 'run', 'max', 'nmax' }, ... ]
      }
    """
    agg: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    if not results_root.is_dir():
        return {}

    for evaluate_log in sorted(results_root.glob("*/*/evaluate.log")):
        exp = evaluate_log.parent.parent.name
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
    exp_names = sorted(set(gtgdfgo.keys()) | set(gtg.keys()))
    rows: list[dict[str, Any]] = []
    for exp in exp_names:
        tasks_g = gtgdfgo.get(exp, {})
        tasks_c = gtg.get(exp, {})
        task_names = sorted(set(tasks_g.keys()) | set(tasks_c.keys()))
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


DECIMALS = 3


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


def write_markdown(
    path: Path,
    gtgdfgo_root: Path,
    gtg_root: Path,
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Evaluate 结果汇总（GTGdfgo vs GTG）",
        "",
        f"- GTGdfgo results: `{gtgdfgo_root}`",
        f"- GTG results: `{gtg_root}`",
        "",
        "各实验在多次 run（如 `run*_seed*`）上聚合：**max** / **nmax** 为 `mean ± std`（std 为样本标准差，单次 run 时 std 记为 0）。",
        "",
        "| experiment | task | GTGdfgo n | max (mean±std) | nmax (mean±std) | GTG n | max (mean±std) | nmax (mean±std) |",
        "|------------|------|-----------|----------------|-----------------|-------|----------------|-----------------|",
    ]
    for r in rows:
        lines.append(
            "| {exp} | {task} | {gn} | {gmax} | {gnmax} | {cn} | {cmax} | {cnmax} |".format(
                exp=r["experiment"],
                task=r["task"],
                gn=r.get("gtgdfgo_n_runs", "") or "—",
                gmax=fmt_pm(r.get("gtgdfgo_max_mean"), r.get("gtgdfgo_max_std")),
                gnmax=fmt_pm(r.get("gtgdfgo_nmax_mean"), r.get("gtgdfgo_nmax_std")),
                cn=r.get("gtg_n_runs", "") or "—",
                cmax=fmt_pm(r.get("gtg_max_mean"), r.get("gtg_max_std")),
                cnmax=fmt_pm(r.get("gtg_nmax_mean"), r.get("gtg_nmax_std")),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gtgdfgo",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
        help="GTGdfgo results 目录（内含 <experiment>/<run>/evaluate.log）",
    )
    ap.add_argument(
        "--gtg",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "GTG" / "results",
        help="GTG 对照 results 目录",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出前缀（默认写入 GTGdfgo/results/eval_comparison）",
    )
    args = ap.parse_args()

    out_base = args.output
    if out_base is None:
        out_base = Path(__file__).resolve().parent.parent / "results" / "eval_comparison"

    gtgdfgo = scan_results_root(args.gtgdfgo)
    gtg = scan_results_root(args.gtg)

    rows = build_comparison_rows(gtgdfgo, gtg)
    md_path = out_base.with_suffix(".md")
    csv_path = out_base.with_suffix(".csv")
    md_path.parent.mkdir(parents=True, exist_ok=True)

    write_csv(csv_path, rows)
    write_markdown(md_path, args.gtgdfgo, args.gtg, rows)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"GTGdfgo experiments: {len(gtgdfgo)}, GTG experiments: {len(gtg)}, comparison rows: {len(rows)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate one LaTeX table: real-task few-shot raw max reward by checkpoint epoch.

Layout (one ``table*``, **one tabular per task**):
  Within each task, Ep* columns are the union of checkpoint epochs for **that task only**
  (rover/robot\_push 不会被迫与 lunar 共用 Ep70–100 空列).
  Columns per tabular: Task, lr, tsbias, hyper, Ep10, Ep20, ...
  Rows: one row per hyper folder under that task.

Source logs:
  .../few_shot/<task>_frac.../.../<hyper...>/run*_seed*/evaluate_ep*.log

Metric:
  ``[<task>] max_ep_reward: ...`` → mean ± std over seeds.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import DefaultDict

_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _ROOT / "results" / "analysis_table"

# 与主分析中 real-world 任务顺序一致（仅用于行排序）
_TASK_ORDER: tuple[str, ...] = ("lunar_lander", "rover", "robot_push")

_RE_RUN_SEED = re.compile(r"^run(\d+)_seed(\d+)$")
_RE_EVAL_EP = re.compile(r"^evaluate_ep(\d+)\.log$")
_RE_MAX = re.compile(
    r"^\[([A-Za-z0-9_]+)\]\s+max_ep_reward:\s*([-\d\.eE+]+),\s*median:\s*([-\d\.eE+]+),\s*mean:\s*([-\d\.eE+]+)\s*$"
)
_RE_LR = re.compile(r"(?:^|[_/])lr([\d.eE+-]+)")
_RE_TSB = re.compile(r"tsbias([\d.]+)")


def _task_rank(name: str) -> int:
    try:
        return _TASK_ORDER.index(name)
    except ValueError:
        return len(_TASK_ORDER) + hash(name) % 1000


def _hyper_sort_key(path: Path) -> tuple[float, str]:
    """同一任务下按 lr 数值再按目录名排序。"""
    m = _RE_LR.search(path.name)
    if m:
        try:
            return (float(m.group(1)), path.name)
        except ValueError:
            pass
    return (float("inf"), path.name)


def _leaf_hyper_dirs(task_root: Path) -> list[Path]:
    """
    取「最深层」且含 ``run*_seed*`` 的目录，避免把上层 fs_pool 误当成 hyper。
    """
    cands = [
        p
        for p in task_root.rglob("*")
        if p.is_dir() and any(p.glob("run*_seed*"))
    ]
    cands.sort(key=lambda p: -len(p.parts))
    selected: list[Path] = []
    for p in cands:
        if any(str(s).startswith(str(p) + "/") for s in selected):
            continue
        selected.append(p)
    return sorted(selected, key=lambda p: p.as_posix())


def _iter_latest_run_seed_dirs(model_dir: Path) -> list[Path]:
    by_seed: dict[int, tuple[int, Path]] = {}
    for d in model_dir.iterdir():
        if not d.is_dir():
            continue
        m = _RE_RUN_SEED.match(d.name)
        if not m:
            continue
        run_id = int(m.group(1))
        seed = int(m.group(2))
        prev = by_seed.get(seed)
        if prev is None or run_id > prev[0]:
            by_seed[seed] = (run_id, d)
    return [v[1] for v in sorted(by_seed.values(), key=lambda t: (t[0], t[1].name))]


def _parse_max_from_log(logf: Path) -> tuple[str, float] | None:
    try:
        txt = logf.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for ln in txt.splitlines():
        m = _RE_MAX.match(ln.strip())
        if m:
            return m.group(1), float(m.group(2))
    return None


def _fmt_pm(vals: list[float]) -> str:
    if not vals:
        return "--"
    mu = float(mean(vals))
    sd = float(stdev(vals)) if len(vals) >= 2 else 0.0
    return f"{mu:.3f} $\\pm$ {sd:.3f}"


def _parse_lr_tsbias(hyper_name: str) -> tuple[str, str]:
    lr_s = "--"
    m_lr = _RE_LR.search(hyper_name)
    if m_lr:
        lr_s = m_lr.group(1)
    ts_s = "--"
    m_ts = _RE_TSB.search(hyper_name)
    if m_ts:
        ts_s = m_ts.group(1)
    return lr_s, ts_s


def _aggregate_one_hyper(hyper_dir: Path) -> dict[str, dict[int, list[float]]]:
    """task -> epoch -> seed values。"""
    agg: DefaultDict[str, DefaultDict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rs in _iter_latest_run_seed_dirs(hyper_dir):
        for ep_log in sorted(rs.glob("evaluate_ep*.log")):
            m = _RE_EVAL_EP.match(ep_log.name)
            if not m:
                continue
            ep = int(m.group(1))
            parsed = _parse_max_from_log(ep_log)
            if not parsed:
                continue
            task, mx = parsed
            agg[task][ep].append(mx)
    return {t: dict(ep_m) for t, ep_m in agg.items()}


def _collect_row_pairs(
    fewshot_root: Path,
    *,
    only_task: str | None,
    explicit_hypers: list[Path] | None,
) -> list[tuple[str, Path]]:
    """(logical_task_name, hyper_dir) 列表，已排序。"""
    pairs: list[tuple[str, Path]] = []
    fr = fewshot_root.resolve()

    if explicit_hypers:
        for h in explicit_hypers:
            hr = h.resolve()
            try:
                rel = hr.relative_to(fr)
            except ValueError:
                print(f"[warn] skip (outside --fewshot-root): {hr}", file=sys.stderr)
                continue
            if not rel.parts:
                continue
            task = rel.parts[0].split("_frac", 1)[0]
            if only_task and task != only_task:
                continue
            if hr.is_dir() and any(hr.glob("run*_seed*")):
                pairs.append((task, hr))
        pairs.sort(key=lambda x: (_task_rank(x[0]), _hyper_sort_key(x[1])))
        return pairs

    for task_root in sorted(p for p in fr.iterdir() if p.is_dir()):
        task = task_root.name.split("_frac", 1)[0]
        if only_task and task != only_task:
            continue
        for h in _leaf_hyper_dirs(task_root):
            pairs.append((task, h.resolve()))
    pairs.sort(key=lambda x: (_task_rank(x[0]), _hyper_sort_key(x[1])))
    return pairs


def build_long_table_per_task(
    pairs: list[tuple[str, Path]],
) -> list[tuple[str, list[str], list[list[str]], list[int]]]:
    """按任务分子表；各任务内 epoch 列为该任务行上出现过的 checkpoint 并集。"""
    per_row: list[tuple[str, Path, dict[int, list[float]]]] = []
    for task, hpath in pairs:
        agg_by_task = _aggregate_one_hyper(hpath)
        ep_map = agg_by_task.get(task)
        if ep_map is None and len(agg_by_task) == 1:
            ep_map = next(iter(agg_by_task.values()))
        elif ep_map is None:
            ep_map = {}
        per_row.append((task, hpath, ep_map))

    by_task: dict[str, list[tuple[str, Path, dict[int, list[float]]]]] = defaultdict(list)
    for item in per_row:
        by_task[item[0]].append(item)

    blocks: list[tuple[str, list[str], list[list[str]], list[int]]] = []
    for task in sorted(by_task.keys(), key=_task_rank):
        rows_data = by_task[task]
        epochs = sorted({ep for _, _, em in rows_data for ep in em.keys()})
        headers = ["Task", "lr", "tsbias", "hyper"] + [f"Ep{e}" for e in epochs]
        body: list[list[str]] = []
        for _t, hpath, ep_map in rows_data:
            lr_s, ts_s = _parse_lr_tsbias(hpath.name)
            h_esc = hpath.name.replace("_", r"\_")
            row: list[str] = [
                task.replace("_", r"\_"),
                lr_s.replace("_", r"\_"),
                ts_s.replace("_", r"\_"),
                h_esc,
            ]
            for ep in epochs:
                row.append(_fmt_pm(ep_map.get(ep, [])))
            body.append(row)
        blocks.append((task, headers, body, epochs))
    return blocks


def write_latex_per_task_blocks(
    blocks: list[tuple[str, list[str], list[list[str]], list[int]]],
    out_tex: Path,
    *,
    caption_tail: str,
) -> None:
    lines = [
        r"% Requires: \usepackage{booktabs}, \usepackage{graphicx}.",
        r"\begin{table*}[t!]",
        r"\caption{Real-task few-shot: \texttt{max\_ep\_reward} mean $\pm$ std over seeds; "
        r"one row per hyper folder; lr/tsbias from folder name. "
        r"Epoch columns are defined \emph{per task block} (see \textbf{bold} task headers). "
        + caption_tail
        + "}",
        r"\vspace{0.3em}",
        r"\centering",
    ]
    for bi, (task, headers, body, _epochs) in enumerate(blocks):
        n_cols = len(headers)
        spec = "l|ccc|" + "c" * (n_cols - 4)
        t_show = task.replace("_", r"\_")
        if bi > 0:
            lines.append(r"\vspace{1.0em}")
        lines.append(rf"\textbf{{{t_show}}}")
        lines.append(r"\vspace{0.25em}")
        lines.append(r"\resizebox{\linewidth}{!}{")
        lines.append(rf"\begin{{tabular}}{{{spec}}}")
        lines.append(r"\toprule")
        lines.append(" & ".join(headers) + r" \\")
        lines.append(r"\midrule")
        for r in body:
            lines.append(" & ".join(r) + r" \\")
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
        ]
    lines += [
        r"\label{tab:real-task-fewshot-epoch-max}",
        r"\end{table*}",
        "",
    ]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fewshot-root",
        type=Path,
        default=_ROOT / "results" / "real_task" / "few_shot",
        help="Root containing <task>_frac.../.../hyper/run*_seed*.",
    )
    ap.add_argument(
        "--out-tex",
        type=Path,
        default=_OUT_DIR / "fewshot_epoch_max.tex",
    )
    ap.add_argument(
        "--only-task",
        type=str,
        default="",
        help="Only include rows for this task name (e.g. lunar_lander).",
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Restrict to these hyper directories (can repeat). "
            "Each path must lie under --fewshot-root; task name is inferred from path."
        ),
    )
    args = ap.parse_args()

    fewshot_root = args.fewshot_root.resolve()
    if not fewshot_root.is_dir():
        raise SystemExit(f"Not a directory: {fewshot_root}")

    only_task = args.only_task.strip() or None
    explicit = [p.resolve() for p in args.model_dir if p.exists()]
    if args.model_dir and not explicit:
        raise SystemExit("No existing paths in --model-dir.")

    pairs = _collect_row_pairs(
        fewshot_root,
        only_task=only_task,
        explicit_hypers=explicit if explicit else None,
    )
    if not pairs:
        raise SystemExit("No hyper directories with run*_seed* found.")

    blocks = build_long_table_per_task(pairs)
    if not blocks:
        raise SystemExit("No evaluate_ep*.log metrics found.")
    if not any(b[3] for b in blocks):
        raise SystemExit("No evaluate_ep*.log metrics found.")

    if explicit:
        caption_tail = "Rows from explicit ``--model-dir`` list only."
    elif only_task:
        caption_tail = f"Filtered to task \texttt{{{only_task}}}."
    else:
        caption_tail = "All leaf hyper folders under each task."

    write_latex_per_task_blocks(blocks, args.out_tex.resolve(), caption_tail=caption_tail)
    nrows = sum(len(b[2]) for b in blocks)
    print(f"Wrote {args.out_tex.resolve()} ({nrows} rows in {len(blocks)} task blocks)")
    for task, h in pairs:
        print(f"  {task}: {h}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a LaTeX table of best latent objective value per (task, phase) and per DUO variant.

English doc: Reads ``quality_trace_<variant>_<slug>.npz`` from a Quality run's artifact folder;
``slug`` splits at ``.meta_`` into ``<exp1_*.meta stem>`` and ``<phase>``. Best value =
``proxy_best_f`` when present (proxy-selected decoded trajectory), else legacy min over
all trajectories at the final denoise step.

Merge mode (``--merge_artifacts_dirs``): combine shift rows from all gaps; ``train_domain`` rows
only from ``--reference_train_artifacts_dir`` (often ``.../quality_training/<pkl>_seed*/eval_train_domain``,
separate from per-gap ``artifacts_*``).
Resolves ``exp1_D_test_gap*`` metas under ``<uniso_parent>/data_exp1_<tag>/``.

中文注释: 各 gap 下 ``D_train_*`` 名称相同但 ``run_exp1`` 生成的实例不同，且 quartet 在对应 gap 的 PKL 上训练，
故「按 gap 单表」时 D_train 数值会随 gap 变——这是实验设定而非导出错误。合并表可把 D_train 固定为某一参考 gap。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from comparisonExperiment.experiment1.task_family import LatentObjective

METHODS: tuple[str, ...] = ("st_duo", "st_text", "mt_label", "mt_text")
KNOWN_PHASES: tuple[str, ...] = (
    "train_domain",
    "shift_zero_shot",
    "shift_few_shot",
    "shift_few_shot_tail10p",
    "shift_few_shot_tail20p",
    "shift_few_shot_tail50p",
)
_TRACE_RE = re.compile(
    r"^quality_trace_(st_duo|st_text|mt_label|mt_text)_(.+)\.npz$",
)
_PHASE_ORDER = {
    "train_domain": 0,
    "shift_zero_shot": 1,
    "shift_few_shot": 2,
    "shift_few_shot_tail10p": 3,
    "shift_few_shot_tail20p": 4,
    "shift_few_shot_tail50p": 5,
}
_RE_D_TRAIN = re.compile(r"^D_train_(\d+)$")
_RE_GAP_IN_TASK = re.compile(r"gap(0p\d+)")
_RE_D_TEST_GAP = re.compile(r"^D_test_gap(0p\d+)$")
_RE_D_TEST_SHIFT = re.compile(r"^D_test_(sim_low|sim_mid|sim_high)$")
_SHIFT_ORDER = {"sim_low": 0, "sim_mid": 1, "sim_high": 2}


def _parse_slug(slug: str) -> tuple[str, str]:
    if ".meta_" not in slug:
        raise ValueError(f"cannot parse slug (expected .meta_): {slug}")
    stem_part, phase = slug.split(".meta_", 1)
    if phase not in KNOWN_PHASES:
        raise ValueError(f"unknown phase {phase!r} in slug {slug!r}")
    return stem_part, phase


def _meta_path(uniso_dir: Path, stem_part: str) -> Path:
    return uniso_dir / f"{stem_part}.meta.json"


def _best_f_npz(npz_path: Path, objective: LatentObjective) -> float:
    """English doc: Prefer ``proxy_best_f`` (v5+); fall back to min oracle at last denoise step."""
    data = np.load(npz_path)
    if "proxy_best_f" in data:
        v = float(np.asarray(data["proxy_best_f"]).reshape(-1)[0])
        if np.isfinite(v):
            return v
    z = data["z_steps"][-1].astype(np.float64)
    zt = torch.from_numpy(z.astype(np.float32))
    f = objective.eval(zt).detach().cpu().numpy().reshape(-1)
    return float(np.min(f))


def _phase_tex(phase: str) -> str:
    m = re.match(r"^shift_few_shot_tail(\d+)p$", phase)
    if m:
        return rf"\texttt{{fs{m.group(1)}}}"
    return {
        "train_domain": r"\texttt{train}",
        "shift_zero_shot": r"\texttt{zs}",
        "shift_few_shot": r"\texttt{fs}",
    }.get(phase, phase)


def _escape_tex(s: str) -> str:
    return s.replace("_", r"\_")


def _uniso_dir_for_task_id(
    task_id: str,
    *,
    uniso_parent: Path,
    train_uniso: Path,
) -> Path:
    """English doc: D_test_* metas under ``data_exp1_<shift>*/``; D_train_* under ``train_uniso``."""
    m = _RE_D_TEST_SHIFT.match(task_id)
    if m:
        sh = m.group(1)
        for cand in sorted(uniso_parent.glob(f"data_exp1_{sh}*")):
            if (cand / f"exp1_{task_id}.meta.json").is_file():
                return cand
        return uniso_parent / f"data_exp1_{sh}"
    m = _RE_D_TEST_GAP.match(task_id)
    if m:
        return uniso_parent / f"data_exp1_{m.group(1)}"
    return train_uniso


def _load_dbest(
    task_id: str,
    *,
    uniso_parent: Path,
    train_uniso: Path,
) -> float | None:
    """English doc: ``dbest`` = min raw latent objective on the task point pool (``run_exp1`` meta)."""
    u = _uniso_dir_for_task_id(task_id, uniso_parent=uniso_parent, train_uniso=train_uniso)
    p = u / f"exp1_{task_id}.meta.json"
    if not p.is_file():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    if "dbest" not in raw:
        return None
    return float(raw["dbest"])


def _uniso_dir_for_stem(
    stem_part: str,
    *,
    uniso_parent: Path,
    train_uniso: Path,
) -> Path:
    """English doc: D_test_sim_* / D_test_gap* under data_exp1_*; D_train_* uses train_uniso."""
    m = re.match(r"^exp1_D_test_(sim_low|sim_mid|sim_high)", stem_part)
    if m:
        sh = m.group(1)
        for cand in sorted(uniso_parent.glob(f"data_exp1_{sh}*")):
            if (cand / f"{stem_part}.meta.json").is_file():
                return cand
        return uniso_parent / f"data_exp1_{sh}"
    m = re.match(r"^exp1_D_test_gap(0p\d+)", stem_part)
    if m:
        return uniso_parent / f"data_exp1_{m.group(1)}"
    return train_uniso


def _gap_numeric_from_task_id(task_id: str) -> float:
    m = _RE_D_TEST_SHIFT.match(task_id)
    if m:
        return float(_SHIFT_ORDER.get(m.group(1), 9))
    m = _RE_GAP_IN_TASK.search(task_id)
    if not m:
        return 1e9
    digits = m.group(1)[2:]
    return float(digits) / 1000.0


def _rows_sort_key(item: tuple[str, str]) -> tuple[float | int, ...]:
    """English doc: D_train_1..K train, then D_test by sim_low→high (or legacy gap), zs before fs."""
    task_id, phase = item
    pord = _PHASE_ORDER.get(phase, 9)
    dm = _RE_D_TRAIN.match(task_id)
    if dm:
        return (0, int(dm.group(1)), pord, task_id)
    if task_id.startswith("D_test_"):
        return (1, _gap_numeric_from_task_id(task_id), pord, task_id)
    return (2, 0.0, pord, task_id)


def _ingest_one_npz(
    npz_path: Path,
    cells: dict[tuple[str, str], dict[str, float]],
    uniso_for_meta: Path,
) -> None:
    m = _TRACE_RE.match(npz_path.name)
    if not m:
        return
    method, slug = m.group(1), m.group(2)
    stem_part, phase = _parse_slug(slug)
    meta = _meta_path(uniso_for_meta, stem_part)
    if not meta.is_file():
        print(f"[warn] missing meta {meta}, skip {npz_path.name}")
        return
    raw = json.loads(meta.read_text(encoding="utf-8"))
    obj_name = str(raw.get("objective", "branin"))
    d_z = int(raw.get("d_z", 2))
    from comparisonExperiment.experiment1.branin_standard import normalize_branin_domain

    bd = normalize_branin_domain(str(raw.get("branin_domain", "legacy")))
    objective = LatentObjective(
        name=obj_name, d_z=d_z, branin_domain=bd  # type: ignore[arg-type]
    )
    bf = _best_f_npz(npz_path, objective)
    tid = str(raw.get("task_id", stem_part.replace("exp1_", "", 1)))
    key = (tid, phase)
    row = cells.setdefault(key, {})
    row[method] = min(bf, row.get(method, bf))


def collect_cells(
    artifact_dirs: list[Path],
    *,
    train_uniso: Path,
    uniso_parent: Path,
    reference_train_artifacts_dir: Path | None,
) -> dict[tuple[str, str], dict[str, float]]:
    """English doc: Merge NPZs; train_domain only from reference dir when set.

    If reference dir is not listed in artifact_dirs (e.g. eval_train_domain under quality_training),
    it is still scanned so D_train rows appear in the merged table.
    """
    cells: dict[tuple[str, str], dict[str, float]] = {}
    ref_resolved = (
        reference_train_artifacts_dir.resolve()
        if reference_train_artifacts_dir is not None
        else None
    )

    scan_dirs: list[Path] = []
    seen_resolved: set[Path] = set()
    if ref_resolved is not None and ref_resolved.is_dir():
        scan_dirs.append(ref_resolved)
        seen_resolved.add(ref_resolved)

    for art in artifact_dirs:
        if not art.is_dir():
            print(f"[warn] skip missing artifacts_dir: {art}")
            continue
        a_res = art.resolve()
        if a_res in seen_resolved:
            continue
        scan_dirs.append(a_res)
        seen_resolved.add(a_res)

    for art in scan_dirs:
        a_res = art.resolve()
        for p in sorted(art.glob("quality_trace_*.npz")):
            m = _TRACE_RE.match(p.name)
            if not m:
                continue
            slug = m.group(2)
            stem_part, phase = _parse_slug(slug)
            if phase == "train_domain" and ref_resolved is not None:
                if a_res != ref_resolved:
                    continue
            u = _uniso_dir_for_stem(
                stem_part,
                uniso_parent=uniso_parent,
                train_uniso=train_uniso,
            )
            _ingest_one_npz(p, cells, u)
    return cells


def _emit_tex(
    cells: dict[tuple[str, str], dict[str, float]],
    out_tex: Path,
    *,
    caption: str,
    label: str,
    extra_comment_lines: list[str],
    uniso_parent: Path,
    train_uniso: Path,
) -> None:
    rows_sorted = sorted(cells.keys(), key=_rows_sort_key)
    dbest_cache: dict[str, float | None] = {}
    for tid, _ph in rows_sorted:
        if tid not in dbest_cache:
            dbest_cache[tid] = _load_dbest(
                tid, uniso_parent=uniso_parent, train_uniso=train_uniso
            )

    lines: list[str] = [
        r"% Auto-generated by QualityExperiment.export_quality_latex_table",
        *extra_comment_lines,
        r"% Lower is better. Bold = best among st\_duo / st\_text / mt\_label / mt\_text (not Dbest).",
        r"% Metric: proxy-selected decoded trajectory oracle $f(\mathbf{z})$ (v5); else last-step min.",
        r"% Dbest = min raw $f(\mathbf{z})$ on the task point pool (see run\_exp1 meta).",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        caption,
        label,
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        r"Task id & Phase & Dbest & st\_duo & st\_text & mt\_label & mt\_text \\",
        r"\hline",
    ]

    for task_id, phase in rows_sorted:
        vals = [cells[(task_id, phase)].get(m, float("nan")) for m in METHODS]
        row_min = np.nanmin(np.asarray(vals, dtype=np.float64))
        task_disp = _escape_tex(str(task_id))
        ph_tex = _phase_tex(phase)
        _db = dbest_cache.get(task_id)
        db_tex = f"{float(_db):.4f}" if _db is not None else "---"
        cells_tex: list[str] = []
        for v in vals:
            if np.isnan(v):
                cells_tex.append("---")
                continue
            s = f"{v:.4f}"
            if float(v) <= row_min + 1e-9:
                cells_tex.append(r"\textbf{" + s + "}")
            else:
                cells_tex.append(s)
        lines.append(
            f"{task_disp} & {ph_tex} & {db_tex} & {cells_tex[0]} & {cells_tex[1]} & {cells_tex[2]} & {cells_tex[3]} \\\\"
        )

    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] {out_tex}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export exp1 Quality best-f LaTeX table.")
    ap.add_argument(
        "--artifacts_dir",
        type=str,
        default="",
        help="Single run: directory with quality_trace_*.npz.",
    )
    ap.add_argument(
        "--merge_artifacts_dirs",
        type=str,
        nargs="*",
        default=None,
        help="If non-empty: merge several gap runs; use --reference_train_artifacts_dir for D_train rows.",
    )
    ap.add_argument(
        "--reference_train_artifacts_dir",
        type=str,
        default="",
        help=(
            "Merge mode: directory with train_domain NPZs (e.g. quality_training/.../eval_train_domain). "
            "Required if >1 merge dir; if omitted with 1 dir, defaults to that dir (legacy combined artifacts)."
        ),
    )
    ap.add_argument(
        "--reference_uniso_data_dir",
        type=str,
        default="",
        help="UniSO dir for D_train_* metas (e.g. .../data_exp1_0p000). Defaults to --uniso_data_dir.",
    )
    ap.add_argument(
        "--uniso_data_dir",
        type=str,
        default=str(_PROJECT_ROOT.parent / "UniSO" / "data"),
        help="Default train-domain UniSO root (data_exp1_* parent is parent of this if .../data).",
    )
    ap.add_argument("--out_tex", type=str, required=True, help="Output .tex path.")
    args = ap.parse_args()

    # nargs=*: absent → None; present → list (possibly empty)
    merge_list = args.merge_artifacts_dirs
    if merge_list is not None:
        dirs = [Path(p).resolve() for p in merge_list if str(p).strip()]
        if not dirs:
            raise SystemExit("empty --merge_artifacts_dirs (pass at least one directory)")
        ref_train = Path(args.reference_train_artifacts_dir).resolve() if str(args.reference_train_artifacts_dir).strip() else None
        if len(dirs) > 1 and ref_train is None:
            raise SystemExit("merge with multiple dirs requires --reference_train_artifacts_dir")
        if ref_train is None and len(dirs) == 1:
            ref_train = dirs[0]
        train_uniso = (
            Path(args.reference_uniso_data_dir).resolve()
            if str(args.reference_uniso_data_dir).strip()
            else Path(args.uniso_data_dir).resolve()
        )
        uniso_parent = train_uniso.parent
        cells = collect_cells(
            dirs,
            train_uniso=train_uniso,
            uniso_parent=uniso_parent,
            reference_train_artifacts_dir=ref_train,
        )
        extra = [
            r"% Merge: shift rows from all gap artifact dirs; D_train train rows from reference (e.g. dtrain_universal/eval_train_domain).",
            rf"% reference_train_artifacts_dir={ref_train}",
        ]
        cap = (
            r"\caption{Exp1 Quality v5 (merged gaps): proxy-filtered best $f(\mathbf{z})$; "
            r"\texttt{train} rows from universal D\_train eval; \texttt{zs/fs} per test gap.}"
        )
        lab = r"\label{tab:exp1-quality-best-f-merged}"
    elif str(args.artifacts_dir).strip():
        art = Path(args.artifacts_dir).resolve()
        train_uniso = Path(args.uniso_data_dir).resolve()
        uniso_parent = train_uniso.parent
        cells = collect_cells(
            [art],
            train_uniso=train_uniso,
            uniso_parent=uniso_parent,
            reference_train_artifacts_dir=None,
        )
        extra = [
            r"% Single-gap table: D_train and D_test both for this gap's UniSO + PKL family.",
        ]
        cap = (
            r"\caption{Exp1 Quality v5: proxy-filtered best latent $f(\mathbf{z})$ per model.}"
        )
        lab = r"\label{tab:exp1-quality-best-f}"
    else:
        raise SystemExit("Provide --artifacts_dir or non-empty --merge_artifacts_dirs")

    if not cells:
        raise SystemExit("no quality_trace_*.npz parsed; check artifact paths.")

    _emit_tex(
        cells,
        Path(args.out_tex).resolve(),
        caption=cap,
        label=lab,
        extra_comment_lines=extra,
        uniso_parent=uniso_parent,
        train_uniso=train_uniso,
    )


if __name__ == "__main__":
    main()

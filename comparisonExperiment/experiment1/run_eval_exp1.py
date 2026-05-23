from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from comparisonExperiment.experiment1.evaluator import write_gap_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gap_dir",
        type=str,
        required=True,
        help="One gap directory under DUO/results/comparison1/exp1/ (e.g. .../gap0p500).",
    )
    ap.add_argument(
        "--test_meta_json",
        type=str,
        required=True,
        help="Path to exp1 meta json for the test task (exp1_D_test_gap*.meta.json).",
    )
    ap.add_argument(
        "--candidates_dir",
        type=str,
        required=True,
        help="Directory containing method jsonl files: duo.jsonl, uniso.jsonl, ...",
    )
    ap.add_argument(
        "--methods",
        type=str,
        default="duo,uniso",
        help="Comma-separated method names (each corresponds to <name>.jsonl).",
    )
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument(
        "--family_meta_json",
        type=str,
        default="",
        help="Optional family_meta.json from run_exp1.",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default="summary.json",
        help="Output file name under gap_dir.",
    )
    args = ap.parse_args()

    gap_dir = Path(args.gap_dir)
    gap_dir.mkdir(parents=True, exist_ok=True)
    out_path = gap_dir / str(args.out_json)
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    fm = Path(args.family_meta_json) if str(args.family_meta_json).strip() else None

    write_gap_summary(
        gap_dir=gap_dir,
        test_meta_json=Path(args.test_meta_json),
        methods=methods,
        candidates_dir=Path(args.candidates_dir),
        out_path=out_path,
        topk=int(args.topk),
        family_meta_json=fm,
    )
    print(f"[exp1-eval] wrote {out_path}")


if __name__ == "__main__":
    main()


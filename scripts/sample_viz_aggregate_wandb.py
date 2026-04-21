#!/usr/bin/env python3
"""Average sample_viz JSONL dumps (from --sample_viz_dump_jsonl) across seeds and log one curve per tag to wandb."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def aggregate_tag(dump_dir: Path, tag: str, seeds: list[int]) -> list[dict]:
    paths = [dump_dir / f"{tag}_seed{s}.jsonl" for s in seeds]
    for p in paths:
        if not p.is_file():
            raise SystemExit(f"[aggregate] missing file: {p}")
    per_seed = [load_jsonl(p) for p in paths]
    by_step: list[dict] = [{r["viz_step"]: r for r in rows} for rows in per_seed]
    common = set(by_step[0])
    for m in by_step[1:]:
        common &= set(m.keys())
    if not common:
        raise SystemExit(f"[aggregate] no common viz_step across seeds for tag={tag!r}")
    missing = set(by_step[0].keys()) - common
    if missing:
        print(
            f"[aggregate] warn: tag={tag!r} trimming to common steps only; "
            f"dropped extra steps {sorted(missing)[:8]}{'...' if len(missing) > 8 else ''}"
        )
    out: list[dict] = []
    for step in sorted(common):
        chunk = [m[step] for m in by_step]
        keys = ("mean_y", "max_y", "mean_y_norm", "max_y_norm", "t_index", "denoise_progress")
        rec: dict = {"viz_step": int(step)}
        for k in keys:
            vals = [c[k] for c in chunk]
            rec[k] = float(sum(vals)) / len(vals)
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", type=str, required=True)
    ap.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=["mt_text", "mt_task", "st_text", "st_duo"],
    )
    ap.add_argument("--project", type=str, default=os.environ.get("WANDB_PROJECT", "decdiff-opt"))
    ap.add_argument("--group", type=str, required=True)
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--start_seed", type=int, default=0)
    ap.add_argument("--num_seeds", type=int, required=True)
    args = ap.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    dump_dir = Path(args.dump_dir)

    import wandb

    run_name = args.run_name or f"{args.group}_sample_viz_mean"
    wandb.init(project=args.project, name=run_name, group=args.group, job_type="sample_viz_agg")

    for tag in args.tags:
        rows = aggregate_tag(dump_dir, tag, seeds)
        for r in rows:
            s = int(r["viz_step"])
            wandb.log(
                {
                    f"sample_viz/{tag}/mean_y": r["mean_y"],
                    f"sample_viz/{tag}/max_y": r["max_y"],
                    f"sample_viz/{tag}/mean_y_norm": r["mean_y_norm"],
                    f"sample_viz/{tag}/max_y_norm": r["max_y_norm"],
                    f"sample_viz/{tag}/t_index": int(round(r["t_index"])),
                    f"sample_viz/{tag}/denoise_progress": r["denoise_progress"],
                },
                step=s,
            )

    wandb.finish()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
在 GTOPX 单任务上网格搜索轨迹构造超参 (k, eps)，按 evaluate 日志中的 max_ep_reward 选最优。

用法（在 DUO 根目录）::

    python scripts/search_gtopx_traj_hyperparams.py --k-list 10,20,30,50 --eps-list 0.01,0.05,0.1

默认任务: gtopx2, gtopx3, gtopx4, gtopx6；固定 n_traj、horizon 与 train/eval 一致（见参数）。

说明: 每组 (task,k,eps) 会依次调用 construct_trajectories → train → evaluate，耗时较长；
可先 ``--dry-run`` 查看将执行的命令。
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BRACKET_MAX = re.compile(
    r"\[([a-zA-Z0-9_]+)\]\s+max_ep_reward:\s*([-\d.eE+]+(?:e[-+]?\d+)?|nan|inf)"
)


def _parse_max_from_text(text: str, task: str) -> float | None:
    last: float | None = None
    for m in BRACKET_MAX.finditer(text):
        if m.group(1) == task:
            try:
                last = float(m.group(2))
            except ValueError:
                pass
    return last


def main() -> None:
    p = argparse.ArgumentParser(description="GTOPX 单任务 (k, eps) 网格搜索")
    p.add_argument(
        "--tasks",
        type=str,
        default="gtopx2,gtopx3,gtopx4,gtopx6",
        help="逗号分隔任务名",
    )
    p.add_argument("--n-traj", type=int, default=2000, help="轨迹条数（GTOPX 默认常用 2000）")
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--frac", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--k-list",
        type=str,
        default="10,20,30,50",
        help="逗号分隔 k 候选",
    )
    p.add_argument(
        "--eps-list",
        type=str,
        default="0.01,0.05,0.1",
        help="逗号分隔 eps 候选",
    )
    p.add_argument("--python", type=str, default=os.environ.get("PYTHON", "python"))
    p.add_argument(
        "--no-construct",
        action="store_true",
        help="不跑 construct_trajectories（仅当你已有所需 pkl）",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--out-csv",
        type=str,
        default="results/gtopx_traj_search_results.csv",
        help="结果 CSV（相对项目根）",
    )
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    ks = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    epss = [float(x.strip()) for x in args.eps_list.split(",") if x.strip()]

    rows: list[dict[str, object]] = []
    os.chdir(ROOT)

    for task in tasks:
        best_k, best_eps, best_score = None, None, None
        for k in ks:
            for eps in epss:
                cmd_c = [
                    args.python,
                    str(ROOT / "construct_trajectories.py"),
                    "--tasks",
                    task,
                    "--n_traj",
                    str(args.n_traj),
                    "--k",
                    str(k),
                    "--eps",
                    str(eps),
                    "--horizon",
                    str(args.horizon),
                    "--frac",
                    str(args.frac),
                    "--sigma",
                    str(args.sigma),
                    "--seed",
                    str(args.seed),
                ]
                cmd_t = [
                    args.python,
                    str(ROOT / "train.py"),
                    "--train_tasks",
                    task,
                    "--n_traj",
                    str(args.n_traj),
                    "--k",
                    str(k),
                    "--eps",
                    str(eps),
                    "--horizon",
                    str(args.horizon),
                    "--frac",
                    str(args.frac),
                    "--sigma",
                    str(args.sigma),
                    "--seed",
                    str(args.seed),
                ]
                cmd_e = [
                    args.python,
                    str(ROOT / "evaluate.py"),
                    "--train_tasks",
                    task,
                    "--n_traj",
                    str(args.n_traj),
                    "--k",
                    str(k),
                    "--eps",
                    str(eps),
                    "--horizon",
                    str(args.horizon),
                    "--frac",
                    str(args.frac),
                    "--sigma",
                    str(args.sigma),
                    "--seed",
                    str(args.seed),
                ]
                print("\n===", task, "k=", k, "eps=", eps, "===", flush=True)
                if args.dry_run:
                    print(" ", " ".join(cmd_c))
                    print(" ", " ".join(cmd_t))
                    print(" ", " ".join(cmd_e))
                    continue
                if not args.no_construct:
                    r = subprocess.run(cmd_c, cwd=ROOT)
                    if r.returncode != 0:
                        print(f"[skip eval] construct 失败 return={r.returncode}", flush=True)
                        rows.append(
                            {
                                "task": task,
                                "n_traj": args.n_traj,
                                "k": k,
                                "eps": eps,
                                "max_ep_reward": "",
                                "error": "construct_failed",
                            }
                        )
                        continue
                r = subprocess.run(cmd_t, cwd=ROOT)
                if r.returncode != 0:
                    print(f"[skip eval] train 失败 return={r.returncode}", flush=True)
                    rows.append(
                        {
                            "task": task,
                            "n_traj": args.n_traj,
                            "k": k,
                            "eps": eps,
                            "max_ep_reward": "",
                            "error": "train_failed",
                        }
                    )
                    continue
                r = subprocess.run(
                    cmd_e,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                out = (r.stdout or "") + (r.stderr or "")
                score = _parse_max_from_text(out, task)
                rows.append(
                    {
                        "task": task,
                        "n_traj": args.n_traj,
                        "k": k,
                        "eps": eps,
                        "max_ep_reward": score if score is not None else "",
                        "error": "" if r.returncode == 0 else f"eval_exit_{r.returncode}",
                    }
                )
                print(f"  max_ep_reward ~= {score} (eval return {r.returncode})", flush=True)
                if score is not None and not (
                    isinstance(score, float) and str(score).lower() == "nan"
                ):
                    if best_score is None or score > best_score:
                        best_score, best_k, best_eps = score, k, eps

        if not args.dry_run:
            print(
                f"\n[{task}] 当前网格内最优: k={best_k} eps={best_eps} max_ep_reward={best_score}",
                flush=True,
            )

    out = ROOT / args.out_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run and rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["task", "n_traj", "k", "eps", "max_ep_reward", "error"]
            )
            w.writeheader()
            w.writerows(rows)
        print(f"\n已写入 {out}", flush=True)
    elif args.dry_run:
        print("\n(dry-run，未写 CSV)", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

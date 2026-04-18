#!/usr/bin/env python3
"""
打印与 train.py / evaluate.py 中 ``RUN.prefix`` 一致的中间目录名，例如
``mt_911054c35daad7e0_textcond_mttextonly``（由轨迹签名 ``mt_<16位hex>`` + textcond + mttextonly 组成）。

用于 ``train_eval_sweep_w_text.sh`` 按「同一套模型超参」归档日志目录。
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from diffuser.utils.multitask_canon import (
    canonical_train_tasks_csv,
    multitask_text_only_path_infix,
    returns_cond_path_infix,
    text_cond_path_infix,
)
from diffuser.utils.traj_params import multitask_checkpoint_hyper_dir, prepare_multitask_traj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_tasks", required=True)
    p.add_argument("--frac", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--n_traj", type=int, default=1000)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--traj_params_json", default=None)
    args = p.parse_args()

    train_tasks = canonical_train_tasks_csv(args.train_tasks)
    train_tasks_list = [t.strip() for t in train_tasks.split(",") if t.strip()]

    class NS:
        pass

    ns = NS()
    ns.returns_condition = False
    ns.include_returns = False
    # 与 train_eval_sweep_w_text.sh 中 train/eval 一致
    ns.use_text_condition = True
    ns.multitask_text_only = True

    _ret = returns_cond_path_infix(ns)
    _txt = text_cond_path_infix(ns)
    _mto = multitask_text_only_path_infix(ns)
    _, _, _, sig = prepare_multitask_traj(
        train_tasks_list,
        args.n_traj,
        args.k,
        args.eps,
        args.horizon,
        args.traj_params_json,
    )
    hyper = multitask_checkpoint_hyper_dir(sig, _ret, _txt, _mto)
    print(hyper)


if __name__ == "__main__":
    main()

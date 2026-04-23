#!/usr/bin/env python3
"""
打印与 train.py / evaluate.py 中 ``RUN.prefix`` 一致的中间目录名，例如
``mt_911054c35daad7e0_textcond_mttextonly``、可选 ``--hyper_suffix``（如 ``_ce0.2``）、可选 ``_tsbias*`` / ``_msnr*``、
最后非 32 维时的 ``_latent{d}``（顺序与 train/eval 的 multitask ``RUN.prefix`` 一致）。

可用于核对 ``trained_models/multi_*`` 下与 ``train.py``/``evaluate.py`` 一致的中间目录名（如手工对照路径）。
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _bootstrap_diffuser_utils_light() -> None:
    """Register ``multitask_canon`` / ``traj_params`` without loading ``diffuser.utils`` (pulls torch)."""
    import importlib.util
    import types

    if "diffuser.utils.traj_params" in sys.modules:
        return
    root = _ROOT
    if "diffuser" not in sys.modules:
        d = types.ModuleType("diffuser")
        d.__path__ = [os.path.join(root, "diffuser")]
        sys.modules["diffuser"] = d
    if "diffuser.utils" not in sys.modules:
        u = types.ModuleType("diffuser.utils")
        u.__path__ = [os.path.join(root, "diffuser", "utils")]
        sys.modules["diffuser.utils"] = u

    canon_path = os.path.join(root, "diffuser", "utils", "multitask_canon.py")
    spec_c = importlib.util.spec_from_file_location(
        "diffuser.utils.multitask_canon", canon_path
    )
    canon_mod = importlib.util.module_from_spec(spec_c)
    sys.modules["diffuser.utils.multitask_canon"] = canon_mod
    assert spec_c.loader is not None
    spec_c.loader.exec_module(canon_mod)

    traj_path = os.path.join(root, "diffuser", "utils", "traj_params.py")
    spec_t = importlib.util.spec_from_file_location(
        "diffuser.utils.traj_params", traj_path
    )
    traj_mod = importlib.util.module_from_spec(spec_t)
    sys.modules["diffuser.utils.traj_params"] = traj_mod
    assert spec_t.loader is not None
    spec_t.loader.exec_module(traj_mod)


_bootstrap_diffuser_utils_light()

from diffuser.utils.multitask_canon import (
    canonical_train_tasks_csv,
    diffusion_train_path_suffix,
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
    p.add_argument(
        "--hyper_suffix",
        type=str,
        default="",
        help="Append a suffix to the printed hyper dir (e.g. _ce0.2). "
        "Used to keep sweep results under a separate subfolder.",
    )
    p.add_argument(
        "--latent_dim",
        type=int,
        default=32,
        help="VAE latent dim; non-32 appends _latent{d} like train.py / evaluate.py RUN.prefix.",
    )
    p.add_argument(
        "--train_timestep_bias_power",
        type=float,
        default=0.0,
        help="Match train.py; non-zero appends _tsbias… before _latent{d}.",
    )
    p.add_argument(
        "--train_loss_min_snr_gamma",
        type=float,
        default=0.0,
        help="Match train.py; non-zero appends _msnr… before _latent{d}.",
    )
    args = p.parse_args()

    train_tasks = canonical_train_tasks_csv(args.train_tasks)
    train_tasks_list = [t.strip() for t in train_tasks.split(",") if t.strip()]

    class NS:
        pass

    ns = NS()
    ns.returns_condition = False
    ns.include_returns = False
    # 与 train.py / evaluate.py（multitask text + mttextonly）一致
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
    if args.hyper_suffix:
        hyper = f"{hyper}{args.hyper_suffix}"
    _dtrain = diffusion_train_path_suffix(
        float(args.train_timestep_bias_power),
        float(args.train_loss_min_snr_gamma),
    )
    if _dtrain:
        hyper = f"{hyper}{_dtrain}"
    _ld = int(args.latent_dim)
    if _ld != 32:
        hyper = f"{hyper}_latent{_ld}"
    print(hyper)


if __name__ == "__main__":
    main()

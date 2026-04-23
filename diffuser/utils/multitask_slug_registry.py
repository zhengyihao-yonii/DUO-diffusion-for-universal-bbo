# -*- coding: utf-8 -*-
"""
多任务短路径 ``mt_<hash>`` 与完整轨迹签名 ``ptask__...`` / per-task 参数的映射。

- 写入：``construct_trajectories`` 在 ``generated_datasets/multi_*/`` 生成 ``multitask_slug_manifest.json``。
- 读取：训练/评估通过 :func:`~diffuser.utils.traj_params.resolve_multitask_mixed_path` 解析 mixed 路径；
  人类可读细节见 manifest。

命令行::

    cd DUO && python -m diffuser.utils.multitask_slug_registry show generated_datasets/multi_..._frac1.0_sigma0.0
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Optional

MANIFEST_FILENAME = "multitask_slug_manifest.json"


def multitask_manifest_filename(latent_dim: int = 32) -> str:
    """32 维保持历史文件名；其它维度单独 manifest，避免与 32 维记录互相覆盖。"""
    if int(latent_dim) == 32:
        return MANIFEST_FILENAME
    return f"multitask_slug_manifest_latent{int(latent_dim)}.json"


def write_multitask_manifest(
    multi_generated_datasets_dir,
    *,
    traj_signature: str,
    tasks_list: list[str],
    n_traj: dict[str, int],
    k: dict[str, int],
    eps: dict[str, float],
    frac: float,
    sigma: float,
    horizon: int,
    traj_params_json: str | None,
    latent_dim: int = 32,
) -> Path:
    """写入 ``multitask_slug_manifest.json``（与 ``mixed_mt_*.p`` 同目录）。"""
    from diffuser.utils.traj_params import (
        multitask_mixed_basename,
        multitask_slug_id,
    )

    slug = multitask_slug_id(traj_signature)
    _ld = int(latent_dim)
    _mixed = multitask_mixed_basename(traj_signature, _ld)
    payload: dict[str, Any] = {
        "version": 1,
        "slug_id": slug,
        "traj_signature": traj_signature,
        "latent_dim": _ld,
        "mixed_filename": _mixed,
        "train_tasks_csv": ",".join(tasks_list),
        "frac": frac,
        "sigma": sigma,
        "horizon": horizon,
        "n_traj": {t: int(n_traj[t]) for t in tasks_list},
        "k": {t: int(k[t]) for t in tasks_list},
        "eps": {t: float(eps[t]) for t in tasks_list},
        "traj_params_json": os.path.abspath(traj_params_json)
        if traj_params_json
        else None,
        "path_notes": {
            "mixed_p": f"本目录/{_mixed}",
            "trained_models_hyper": f"{slug} + train.py 中 returns/text/mttextonly 路径片段（见 multitask_checkpoint_hyper_dir）",
            "results_layout": "run_multitask.sh 中 w<W>_mt_<hash>... 与 multi_task 下 mt_<hash>... 与 slug 同源",
        },
    }
    root = Path(multi_generated_datasets_dir)
    root.mkdir(parents=True, exist_ok=True)
    out = root / multitask_manifest_filename(latent_dim)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load_multitask_manifest(
    multi_generated_datasets_dir, latent_dim: int = 32
) -> "Optional[dict[str, Any]]":
    root = Path(multi_generated_datasets_dir)
    p = root / multitask_manifest_filename(latent_dim)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _cli() -> None:
    ap = argparse.ArgumentParser(description="查看 multitask_slug_manifest.json 或从 traj_signature 算 slug")
    sub = ap.add_subparsers(dest="cmd")

    p_show = sub.add_parser("show", help="打印某 multi_* 数据目录下的 manifest")
    p_show.add_argument("multi_dir", type=str, help="generated_datasets/multi_*_frac*_sigma* 路径")

    p_hash = sub.add_parser("hash", help="仅打印某 traj_signature 的 mt_<hex>（与训练一致）")
    p_hash.add_argument("traj_signature", type=str)

    args = ap.parse_args()
    if args.cmd == "show":
        root = Path(args.multi_dir)
        d = load_multitask_manifest(args.multi_dir, 32)
        if not d:
            alt = list(root.glob("multitask_slug_manifest_latent*.json"))
            alt.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if alt:
                d = json.loads(alt[0].read_text(encoding="utf-8"))
        if not d:
            print(
                f"未找到 {MANIFEST_FILENAME} 或 multitask_slug_manifest_latent*.json：{args.multi_dir}"
            )
            raise SystemExit(1)
        print(json.dumps(d, indent=2, ensure_ascii=False))
    elif args.cmd == "hash":
        from diffuser.utils.traj_params import multitask_slug_id

        print(multitask_slug_id(args.traj_signature))
    else:
        ap.print_help()
        raise SystemExit(2)


if __name__ == "__main__":
    _cli()

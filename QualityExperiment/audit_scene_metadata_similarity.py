#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit exp1 scene metadata: MiniLM embeddings, cosine similarity, and A-matrix distances.

English doc: Run after ``run_exp1`` with ``--family_mode scene_aware`` or on a
``family_meta.json`` that lists ``scene_metadata_similarity``.

中文注释: 用于 v3 实验设计验证 — metadata 语义相近度 vs 任务仿射矩阵 A 的 Frobenius 距离。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np


def _frobenius_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), ord="fro"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit scene metadata vs affine A similarity.")
    ap.add_argument(
        "--family_meta",
        type=str,
        default="",
        help="Path to family_meta.json (optional; uses gap/seed to rebuild if missing sim).",
    )
    ap.add_argument("--test_shift", type=str, default="sim_low")
    ap.add_argument("--train_d_x", type=str, default="5,6,7,9,10")
    ap.add_argument("--test_d_x", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--similarity_blend", type=float, default=0.65)
    ap.add_argument(
        "--text_encoder_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    ap.add_argument("--out_csv", type=str, default="")
    args = ap.parse_args()

    from comparisonExperiment.experiment1.scene_metadata import (
        build_scene_correlated_family,
        cosine_similarity_matrix,
        default_train_scenarios,
        embed_scenarios_minilm,
        test_scenario_for_shift,
    )

    train_d_x = tuple(int(x.strip()) for x in str(args.train_d_x).split(",") if x.strip())
    params, scenarios, sim = build_scene_correlated_family(
        train_d_x,
        int(args.test_d_x),
        test_shift=str(args.test_shift),
        seed=int(args.seed),
        similarity_blend=float(args.similarity_blend),
        model_name=str(args.text_encoder_model),
    )
    ab = {k: (v.A, v.b) for k, v in params.items()}

    if str(args.family_meta).strip():
        fm = json.loads(Path(args.family_meta).read_text(encoding="utf-8"))
        stored = fm.get("scene_metadata_similarity")
        if stored is not None:
            sim_stored = np.asarray(stored, dtype=np.float64)
            if sim_stored.shape == sim.shape:
                err = float(np.max(np.abs(sim_stored - sim)))
                print(f"[check] stored sim matrix max |diff| = {err:.6e}")

    trains = default_train_scenarios()
    test = test_scenario_for_shift(str(args.test_shift))
    all_sc = trains + (test,)
    emb = embed_scenarios_minilm(all_sc, model_name=str(args.text_encoder_model))
    sim_emb = cosine_similarity_matrix(emb)

    ids = [s.task_id for s in all_sc]
    n = len(ids)
    print("\n=== MiniLM cosine similarity (scenario text for A blending) ===")
    header = "task_id".ljust(18) + "".join(f"{i:>8d}" for i in range(n))
    print(header)
    for i, tid in enumerate(ids):
        row = tid.ljust(18) + "".join(f"{sim_emb[i, j]:8.3f}" for j in range(n))
        print(row)

    print("\n=== Frobenius distance ||A_i - A_j||_F (padded to max d_x) ===")
    d_pad = max(max(train_d_x), int(args.test_d_x))
    a_pad: dict[str, np.ndarray] = {}
    for tid, (a, _b) in ab.items():
        apad = np.zeros((d_pad, 2), dtype=np.float64)
        apad[: a.shape[0], :] = a
        a_pad[tid] = apad

    dist_a = np.zeros((n, n), dtype=np.float64)
    for i, ti in enumerate(ids):
        for j, tj in enumerate(ids):
            dist_a[i, j] = _frobenius_dist(a_pad[ti], a_pad[tj])

    print(header.replace("task_id", "task_id/A"))
    for i, tid in enumerate(ids):
        row = tid.ljust(18) + "".join(f"{dist_a[i, j]:8.3f}" for j in range(n))
        print(row)

    # 中文注释: 训练任务对 — 期望 sim 高则 A 距离低
    print("\n=== Train pairs: metadata_sim vs A_distance (expect negative correlation) ===")
    pairs: list[tuple[str, str, float, float]] = []
    for i in range(len(trains)):
        for j in range(i + 1, len(trains)):
            ti, tj = ids[i], ids[j]
            pairs.append((ti, tj, float(sim_emb[i, j]), float(dist_a[i, j])))
            print(f"  {ti} <-> {tj}:  cos_sim={sim_emb[i, j]:.3f}  ||A_i-A_j||_F={dist_a[i, j]:.3f}")

    if pairs:
        s_vals = np.array([p[2] for p in pairs])
        d_vals = np.array([p[3] for p in pairs])
        corr = float(np.corrcoef(s_vals, d_vals)[0, 1]) if len(pairs) > 1 else float("nan")
        print(f"\n[summary] Pearson(metadata_sim, A_dist) over train pairs = {corr:.4f}")

    if str(args.out_csv).strip():
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["task_i,task_j,cos_sim_metadata,a_frobenius_dist"]
        for i in range(n):
            for j in range(i + 1, n):
                lines.append(f"{ids[i]},{ids[j]},{sim_emb[i, j]:.6f},{dist_a[i, j]:.6f}")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[save] {out}")


if __name__ == "__main__":
    main()

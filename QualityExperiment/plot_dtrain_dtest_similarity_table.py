#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot D_train × D_test metadata similarity table (MiniLM cosine, same as scene_aware exp1).

English doc: Heatmap + CSV for all train tasks and three test shifts (sim_low/mid/high).
中文注释: 与 scene_metadata 一致的文本嵌入余弦相似度，颜色表示相关性高低。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from comparisonExperiment.experiment1.scene_metadata import (  # noqa: E402
    VALID_TEST_SHIFTS,
    cosine_similarity_matrix,
    default_train_scenarios,
    embed_scenarios_minilm,
    test_scenario_for_shift,
)

# Quality v3 defaults
DEFAULT_TRAIN_D_X: tuple[int, ...] = (5, 6, 7, 9, 10)


def _build_labels(
    train_d_x: tuple[int, ...],
    *,
    include_dx: bool,
) -> tuple[list[str], list[str]]:
    """Row/column labels for train + test scenarios."""
    trains = default_train_scenarios()[: len(train_d_x)]
    train_labels = [
        (
            f"{s.task_id}\n(d_x={train_d_x[i]})"
            if include_dx
            else s.task_id
        )
        for i, s in enumerate(trains)
    ]
    test_labels = []
    for sh in VALID_TEST_SHIFTS:
        sc = test_scenario_for_shift(sh)
        test_labels.append(f"{sc.task_id}\n({sh})" if include_dx else sc.task_id)
    return train_labels, test_labels


def _collect_scenarios(train_d_x: tuple[int, ...]) -> tuple[tuple, tuple]:
    trains = default_train_scenarios()[: len(train_d_x)]
    tests = tuple(test_scenario_for_shift(sh) for sh in VALID_TEST_SHIFTS)
    return trains, tests


def compute_similarity_matrix(
    train_d_x: tuple[int, ...],
    *,
    model_name: str,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """
    Return full (n_train+n_test)² cosine matrix and split labels.

    English doc: Embeddings from ``similarity_text()`` (title + description).
    """
    trains, tests = _collect_scenarios(train_d_x)
    all_sc = trains + tests
    emb = embed_scenarios_minilm(all_sc, model_name=model_name)
    sim = cosine_similarity_matrix(emb)
    train_labels, test_labels = _build_labels(train_d_x, include_dx=True)
    all_labels = train_labels + test_labels
    return sim, train_labels, test_labels, all_labels


def _level_label(v: float, *, tex: bool = False) -> str:
    """中文注释: 分档便于读表；tex=False 用英文避免 matplotlib 缺中文字体。"""
    if v >= 0.55:
        return "高" if tex else "H"
    if v >= 0.35:
        return "中" if tex else "M"
    return "低" if tex else "L"


def save_csv(
    sim: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([""] + [lb.replace("\n", " ") for lb in col_labels])
        for i, rl in enumerate(row_labels):
            w.writerow(
                [rl.replace("\n", " ")]
                + [f"{sim[i, j]:.4f}" for j in range(len(col_labels))]
            )


def save_train_test_block_csv(
    sim: np.ndarray,
    n_train: int,
    train_labels: list[str],
    test_labels: list[str],
    out_csv: Path,
) -> None:
    """D_train rows × D_test cols (held-out block)."""
    block = sim[:n_train, n_train:]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([""] + [lb.replace("\n", " ") for lb in test_labels])
        for i, rl in enumerate(train_labels):
            row = [rl.replace("\n", " ")]
            for j in range(block.shape[1]):
                v = float(block[i, j])
                row.append(f"{v:.4f} ({_level_label(v, tex=True)})")
            w.writerow(row)


def plot_heatmap(
    sim: np.ndarray,
    labels: list[str],
    *,
    n_train: int,
    out_png: Path,
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    n = len(labels)
    fig_w = max(8.0, 0.55 * n + 2.0)
    fig_h = max(6.5, 0.55 * n + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(sim, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MiniLM cosine similarity")

    short = [lb.replace("\n", " ") for lb in labels]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_title(title, fontsize=11)

    for i in range(n):
        for j in range(n):
            v = float(sim[i, j])
            txt = f"{v:.2f}\n{_level_label(v)}"
            color = "white" if v > 0.62 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    # 分隔 train | test
    if n > n_train:
        ax.axhline(n_train - 0.5, color="steelblue", linewidth=2.0)
        ax.axvline(n_train - 0.5, color="steelblue", linewidth=2.0)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_train_vs_test_block(
    sim: np.ndarray,
    n_train: int,
    train_labels: list[str],
    test_labels: list[str],
    *,
    out_png: Path,
    title: str,
) -> None:
    block = sim[:n_train, n_train:]
    nr, nc = block.shape
    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * nc + 2), max(5.5, 0.55 * nr + 2)))
    im = ax.imshow(block, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("cosine similarity")
    ax.set_xticks(range(nc))
    ax.set_yticks(range(nr))
    ax.set_xticklabels([t.replace("\n", " ") for t in test_labels], rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels([t.replace("\n", " ") for t in train_labels], fontsize=9)
    ax.set_xlabel("D_test (held-out shift)")
    ax.set_ylabel("D_train")
    ax.set_title(title, fontsize=11)
    for i in range(nr):
        for j in range(nc):
            v = float(block[i, j])
            ax.text(
                j,
                i,
                f"{v:.2f}\n{_level_label(v)}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > 0.62 else "black",
            )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="D_train + D_test similarity heatmap table.")
    ap.add_argument(
        "--train_d_x",
        type=str,
        default=",".join(map(str, DEFAULT_TRAIN_D_X)),
        help="Comma-separated native dims, default v3: 5,6,7,9,10",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(_PROJECT_ROOT / "results" / "dtrain_dtest_similarity"),
    )
    ap.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    args = ap.parse_args()

    train_d_x = tuple(int(x.strip()) for x in str(args.train_d_x).split(",") if x.strip())
    out_dir = Path(args.out_dir).resolve()
    n_train = len(train_d_x)

    sim, train_labels, test_labels, all_labels = compute_similarity_matrix(
        train_d_x,
        model_name=str(args.model),
    )

    plot_heatmap(
        sim,
        all_labels,
        n_train=n_train,
        out_png=out_dir / "similarity_full_matrix.png",
        title="D_train + D_test metadata similarity (MiniLM cosine)",
    )
    plot_train_vs_test_block(
        sim,
        n_train,
        train_labels,
        test_labels,
        out_png=out_dir / "similarity_train_vs_test.png",
        title="D_train → D_test similarity (scene_aware text; higher = closer shift)",
    )
    save_csv(sim, all_labels, all_labels, out_dir / "similarity_full_matrix.csv")
    save_train_test_block_csv(
        sim,
        n_train,
        train_labels,
        test_labels,
        out_dir / "similarity_train_vs_test.csv",
    )

    print(f"[done] wrote figures and CSV under {out_dir}")
    print("[train vs test] cosine similarity:")
    block = sim[:n_train, n_train:]
    for i, tl in enumerate(train_labels):
        tid = tl.split("\n")[0]
        for j, tsh in enumerate(VALID_TEST_SHIFTS):
            print(f"  {tid} vs {tsh}: {block[i, j]:.3f} ({_level_label(float(block[i, j]), tex=True)})")


if __name__ == "__main__":
    main()

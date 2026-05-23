#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export train-task MiniLM similarity table (scenario text only)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np


def _format_matrix_markdown(
    labels: list[str],
    mat: np.ndarray,
    *,
    title: str,
) -> str:
    header = "| | " + " | ".join(labels) + " |"
    sep = "|---|" + "|".join(["---:"] * len(labels)) + "|"
    rows = [header, sep]
    for i, lab in enumerate(labels):
        cells = " | ".join(f"{mat[i, j]:.3f}" for j in range(len(labels)))
        rows.append(f"| **{lab}** | {cells} |")
    return f"### {title}\n\n" + "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_d_x", type=str, default="5,6,7,9,10")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="results/quality_bundle_3/analysis_table",
    )
    ap.add_argument(
        "--text_encoder_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    args = ap.parse_args()

    from comparisonExperiment.experiment1.scene_metadata import (
        cosine_similarity_matrix,
        default_train_scenarios,
        embed_scenarios_minilm,
    )

    train_d_x = tuple(int(x.strip()) for x in str(args.train_d_x).split(",") if x.strip())
    trains = default_train_scenarios()[: len(train_d_x)]
    labels = [s.task_id for s in trains]
    titles = [s.title for s in trains]

    emb = embed_scenarios_minilm(trains, model_name=str(args.text_encoder_model))
    sim = cosine_similarity_matrix(emb)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "scene_train_metadata_cosine_sim.md"
    title_map = "\n".join(f"- **{labels[i]}**: {titles[i]}" for i in range(len(labels)))
    md_body = _format_matrix_markdown(
        labels,
        sim,
        title="Train MiniLM cosine similarity (5 tasks)",
    )
    md_path.write_text(
        "# Scene train similarity\n\n"
        f"Encoder: `{args.text_encoder_model}`\n\n"
        "## Scenario titles\n\n"
        f"{title_map}\n\n"
        f"{md_body}",
        encoding="utf-8",
    )
    print(f"[save] {md_path}")
    print(md_body)


if __name__ == "__main__":
    main()

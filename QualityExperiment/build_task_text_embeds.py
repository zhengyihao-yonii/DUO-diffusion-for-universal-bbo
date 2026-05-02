#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build ``task_text_embeds.npy`` **[T, E]** from comparison Experiment 1 meta JSON files.

English doc: **Does not create new metadata.** Each row is the sentence-transformer encoding of
the ``metadata_text`` field written by ``comparisonExperiment/experiment1/run_exp1.py`` (same text
as ``exp1_D_train_i.metadata``). Use the **same** ``--text_encoder_model`` as in
``run_quality_suite`` / training.

中文注释: 仅把 comparison1 已写好的 metadata 编成向量，避免与 UniSO/DUO 文本不一致。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from QualityExperiment.metadata_embed import encode_sentence_embedding, load_metadata_text


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Encode exp1_D_train_i metadata_text -> [T,E] numpy for multitask text conditioning."
    )
    ap.add_argument(
        "--uniso_data_dir",
        type=str,
        required=True,
        help="Same as run_exp1 --uniso_data_dir (contains exp1_D_train_*.meta.json).",
    )
    ap.add_argument(
        "--n_train_tasks",
        type=int,
        default=5,
        help="Expect exp1_D_train_1 .. exp1_D_train_<n> (default 5).",
    )
    ap.add_argument(
        "--text_encoder_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    ap.add_argument(
        "--out_npy",
        type=str,
        required=True,
        help="Output path, shape [n_train_tasks, E].",
    )
    args = ap.parse_args()

    root = Path(args.uniso_data_dir).resolve()
    rows: list[np.ndarray] = []
    for i in range(1, int(args.n_train_tasks) + 1):
        meta_path = root / f"exp1_D_train_{i}.meta.json"
        if not meta_path.is_file():
            raise SystemExit(f"Missing {meta_path} — run run_exp1.py first.")
        text = load_metadata_text(meta_path)
        emb = encode_sentence_embedding(text, model_name=str(args.text_encoder_model))
        rows.append(emb)
        print(f"[ok] D_train_{i} embed_dim={emb.shape[0]}")

    stacked = np.stack(rows, axis=0).astype(np.float32)
    out = Path(args.out_npy).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), stacked)
    print(f"[save] {out} shape={stacked.shape}")


if __name__ == "__main__":
    main()

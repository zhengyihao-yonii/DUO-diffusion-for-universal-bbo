# -*- coding: utf-8 -*-
"""
Frozen text embeddings for task metadata (LDM-style conditioning on descriptive text).

Uses sentence-transformers when available; optional cache .npy next to metadata.
Does not replace task one-hot conditioning — it is an additional channel.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Sequence

import numpy as np

# Default matches a small, widely used model (384-d embeddings).
DEFAULT_TEXT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def _metadata_path(metadata_dir: Path, task: str) -> Path:
    return metadata_dir / f"{task}.txt"


def load_task_texts(metadata_dir: str | Path, task_names: Sequence[str]) -> dict[str, str]:
    """Load UTF-8 text files `{task}.txt` for each task name."""
    root = Path(metadata_dir)
    out: dict[str, str] = {}
    missing: list[str] = []
    for t in task_names:
        p = _metadata_path(root, t)
        if not p.is_file():
            missing.append(str(p))
            continue
        out[t] = p.read_text(encoding="utf-8").strip()
    if missing:
        raise FileNotFoundError(
            "Missing task metadata files:\n"
            + "\n".join(missing)
            + "\nSee task_metadata/README.md"
        )
    return out


def _cache_path(metadata_dir: Path, model_name: str, task_names: Sequence[str]) -> Path:
    h = hashlib.sha256(
        (model_name + "\n" + "\n".join(sorted(task_names))).encode("utf-8")
    ).hexdigest()[:16]
    safe = model_name.replace("/", "_").replace(" ", "_")[:80]
    return metadata_dir / ".cache" / f"embeddings_{safe}_{h}.npy"


def build_task_text_embedding_matrix(
    task_names: Sequence[str],
    metadata_dir: str | Path = "task_metadata",
    model_name: str = DEFAULT_TEXT_ENCODER,
    use_cache: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Returns:
        matrix: float32 array of shape [len(task_names), D] (row i = task_names[i])
        embed_dim: D
    """
    metadata_dir = Path(metadata_dir)
    names = list(task_names)
    cache = _cache_path(metadata_dir, model_name, names)
    if use_cache and cache.is_file():
        mat = np.load(cache)
        if mat.shape[0] != len(names):
            raise ValueError(
                f"Cached embedding matrix rows {mat.shape[0]} != len(task_names) {len(names)}; delete {cache}"
            )
        return mat.astype(np.float32), int(mat.shape[1])

    texts = load_task_texts(metadata_dir, names)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "use_text_condition requires `sentence-transformers`. "
            "Install: pip install sentence-transformers"
        ) from e

    model = SentenceTransformer(model_name)
    ordered = [texts[t] for t in names]
    emb = model.encode(
        ordered,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    mat = np.asarray(emb, dtype=np.float32)
    embed_dim = int(mat.shape[1])

    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, mat)

    return mat, embed_dim

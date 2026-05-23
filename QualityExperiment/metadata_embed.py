# -*- coding: utf-8 -*-
"""
Load ``metadata_text`` from exp1 ``*.meta.json`` / ``*.metadata`` and encode for DUO text conditioning.

English doc: Used for shift ZS/FS when multitask or single-task **text** checkpoints need the
held-out task description as an embedding vector (same protocol as ``visualize.sh``).

Environment (optional):
  DUO_SENTENCE_TRANSFORMER_PATH — if set to an existing local directory, load the model from disk
  (avoids Hugging Face Hub / mirror HTTPS during shift phases).
  DUO_ST_LOCAL_FILES_ONLY — if truthy, pass ``local_files_only=True`` to ``SentenceTransformer``
  (use with a populated HF cache and e.g. ``HF_HUB_OFFLINE=1``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# Chinese comment: 同一进程内复用模型，避免每个 wandb run 重复访问 Hub（易触发镜像 SSL 问题）。
_ST_MODEL_CACHE: dict[str, object] = {}


def load_metadata_text(meta_json: Path) -> str:
    """
    Prefer ``metadata_text`` field written by ``run_exp1.py``; else read sibling ``exp1_<task>.metadata``.
    """
    p = Path(meta_json)
    raw = json.loads(p.read_text(encoding="utf-8"))
    mt = raw.get("metadata_text")
    if isinstance(mt, str) and mt.strip():
        return mt.strip()
    tid = raw.get("task_id")
    if tid is not None:
        sib = p.parent / f"exp1_{tid}.metadata"
        if sib.is_file():
            return sib.read_text(encoding="utf-8").strip()
    raise ValueError(
        f"No metadata_text in {p} and no sibling exp1_<task>.metadata; "
        "re-run comparisonExperiment/experiment1/run_exp1.py with current DUO."
    )


def _sentence_transformer_source(model_name: str) -> str:
    raw = os.environ.get("DUO_SENTENCE_TRANSFORMER_PATH", "").strip()
    if raw and Path(raw).is_dir():
        return raw
    return str(model_name)


def _get_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Shift-phase text conditioning requires embeddings: "
            "pip install sentence-transformers\n"
            "Or provide a precomputed vector via --held_out_text_embed_npy."
        ) from e
    src = _sentence_transformer_source(model_name)
    lfo = os.environ.get("DUO_ST_LOCAL_FILES_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    key = f"{src}|lfo={int(lfo)}"
    if key not in _ST_MODEL_CACHE:
        _ST_MODEL_CACHE[key] = SentenceTransformer(src, local_files_only=bool(lfo))
    return _ST_MODEL_CACHE[key]


def encode_sentence_embedding(text: str, *, model_name: str) -> np.ndarray:
    """English doc: Returns ``float32`` vector ``[E]`` (same dtype as training pipelines)."""
    model = _get_sentence_transformer(str(model_name))
    v = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
    out = np.asarray(v[0], dtype=np.float32).reshape(-1)
    return out


def resolve_shift_text_embedding(
    meta_json: Path,
    *,
    explicit_npy: Path | None,
    encoder_model: str,
) -> np.ndarray:
    """
    English doc: Priority — (1) ``explicit_npy`` file if provided and exists;
    (2) encode ``load_metadata_text(meta_json)`` with ``encoder_model``.
    """
    if explicit_npy is not None and explicit_npy.is_file():
        return np.load(str(explicit_npy)).astype(np.float32).reshape(-1)
    text = load_metadata_text(meta_json)
    return encode_sentence_embedding(text, model_name=encoder_model)

# -*- coding: utf-8 -*-
"""Export DUO quality trace npz ``x_last`` rows to comparison jsonl."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _to_jsonl_rows(x: np.ndarray, *, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(int(x.shape[0])):
        rows.append({"method": method, "x": x[i].astype(float).tolist()})
    return rows


def export_npz_to_jsonl(
    npz_path: Path,
    out_jsonl: Path,
    *,
    method: str = "duo_mt_text",
) -> int:
    """Write candidates from ``x_last`` in a quality trace npz."""
    data = np.load(npz_path)
    if "x_last" not in data.files:
        raise KeyError(f"{npz_path} missing x_last; keys={data.files}")
    x = np.asarray(data["x_last"], dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"x_last must be 2D, got shape {x.shape}")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = _to_jsonl_rows(x, method=method)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return int(x.shape[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--method", type=str, default="duo_mt_text")
    args = ap.parse_args()
    n = export_npz_to_jsonl(
        Path(args.npz),
        Path(args.out_jsonl),
        method=str(args.method),
    )
    print(f"[export] wrote {n} rows -> {args.out_jsonl}")


if __name__ == "__main__":
    main()

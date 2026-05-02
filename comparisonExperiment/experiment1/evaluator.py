from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import json

import numpy as np
import torch

from comparisonExperiment.experiment1.task_family import LatentObjective


@dataclass(frozen=True)
class TaskOracle:
    """Reconstruct oracle y(x) from meta.json produced by run_exp1."""

    objective: LatentObjective
    A: torch.Tensor  # [d_x, d_z]
    b: torch.Tensor  # [d_x]
    f_min: float
    f_max: float

    def y(self, x: torch.Tensor) -> torch.Tensor:
        # Chinese comment: 用 pinv(A) 把 x 近似映射回 latent z，再算共享的 f(z)。
        x2 = x.to(dtype=torch.float32)
        A = self.A.to(dtype=torch.float32)
        b = self.b.to(dtype=torch.float32)
        z = (x2 - b) @ torch.linalg.pinv(A).T
        f = self.objective.eval(z).to(dtype=torch.float32)
        lo = float(self.f_min)
        hi = float(self.f_max)
        if abs(hi - lo) < 1e-12:
            return torch.zeros_like(f)
        y = 1.0 - (f - lo) / (hi - lo)
        return torch.clamp(y, 0.0, 1.0)


def load_oracle(meta_path: Path) -> TaskOracle:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    obj = LatentObjective(name=str(meta["objective"]), d_z=int(meta.get("d_z", 2)))
    A = torch.tensor(np.asarray(meta["A"], dtype=np.float32))
    b = torch.tensor(np.asarray(meta["b"], dtype=np.float32))
    return TaskOracle(
        objective=obj,
        A=A,
        b=b,
        f_min=float(meta["f_min"]),
        f_max=float(meta["f_max"]),
    )


def score_candidates(
    *,
    meta_path: Path,
    candidates: np.ndarray,
) -> np.ndarray:
    oracle = load_oracle(meta_path)
    x = torch.tensor(np.asarray(candidates, dtype=np.float32))
    y = oracle.y(x).detach().cpu().numpy().reshape(-1)
    return y


def summarize_scores(y: np.ndarray, *, topk: int = 16) -> Dict[str, Any]:
    a = np.asarray(y, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "max": float("nan"), "mean": float("nan"), "topk_mean": float("nan")}
    k = min(int(topk), int(a.size))
    top = np.sort(a)[-k:]
    return {
        "n": int(a.size),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "topk_mean": float(np.mean(top)),
    }


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def load_candidates_jsonl(path: Path) -> np.ndarray:
    """Load candidates from jsonl rows with key 'x'."""
    xs: list[np.ndarray] = []
    for r in _read_jsonl(path):
        x = np.asarray(r["x"], dtype=np.float32).reshape(1, -1)
        xs.append(x)
    if not xs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(xs, axis=0)


def evaluate_one_method(
    *,
    meta_path: Path,
    candidates_jsonl: Path,
    topk: int = 16,
) -> Dict[str, Any]:
    x = load_candidates_jsonl(candidates_jsonl)
    y = score_candidates(meta_path=meta_path, candidates=x)
    summ = summarize_scores(y, topk=topk)
    return {
        "candidates_jsonl": str(candidates_jsonl),
        "scores": summ,
    }


def write_gap_summary(
    *,
    gap_dir: Path,
    test_meta_json: Path,
    methods: Iterable[str],
    candidates_dir: Path,
    out_path: Path,
    topk: int = 16,
    family_meta_json: Optional[Path] = None,
) -> None:
    """Summarize multiple methods under one gap directory."""
    rec: dict[str, Any] = {
        "gap_dir": str(gap_dir),
        "test_meta_json": str(test_meta_json),
        "topk": int(topk),
    }
    if family_meta_json is not None and family_meta_json.exists():
        rec["family"] = json.loads(family_meta_json.read_text(encoding="utf-8"))

    per: dict[str, Any] = {}
    for m in methods:
        p = candidates_dir / f"{m}.jsonl"
        if not p.exists():
            per[m] = {"missing": True, "path": str(p)}
            continue
        per[m] = evaluate_one_method(meta_path=test_meta_json, candidates_jsonl=p, topk=topk)
    rec["methods"] = per

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")



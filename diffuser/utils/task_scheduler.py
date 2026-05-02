from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass(frozen=True)
class TaskSamplingState:
    """Immutable scheduler state for task sampling.

    - task_probs: shape [num_tasks], sums to 1
    - ema_loss: shape [num_tasks], EMA of observed loss (nan/inf-safe)
    """

    task_probs: torch.Tensor
    ema_loss: torch.Tensor


def _safe_normalize_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pp = torch.clamp(p, min=0.0)
    s = torch.sum(pp)
    if not torch.isfinite(s) or float(s) <= 0.0:
        # fallback to uniform
        return torch.full_like(pp, 1.0 / float(pp.numel()))
    return pp / (s + eps)


def init_task_sampling_state(num_tasks: int, device: str | torch.device = "cpu") -> TaskSamplingState:
    nt = int(num_tasks)
    probs = torch.full((nt,), 1.0 / float(nt), dtype=torch.float32, device=device)
    ema = torch.full((nt,), float("nan"), dtype=torch.float32, device=device)
    return TaskSamplingState(task_probs=probs, ema_loss=ema)


def update_task_sampling_state(
    state: TaskSamplingState,
    *,
    task_ids: torch.Tensor,
    batch_loss: float,
    ema_beta: float,
    min_prob: float,
    temperature: float,
) -> TaskSamplingState:
    """Update EMA loss and recompute task sampling probabilities.

    This is intentionally simple:
    - attribute the (scalar) batch loss to tasks present in this batch,
      weighted by their frequency in the batch.
    - tasks with higher EMA loss get higher sampling probability.

    Note: if you later expose per-sample loss, replace this with a true per-task mean.
    """
    if task_ids.numel() == 0:
        return state

    nt = int(state.task_probs.numel())
    tids = task_ids.detach().to(dtype=torch.long).view(-1)
    tids = tids[(tids >= 0) & (tids < nt)]
    if tids.numel() == 0:
        return state

    loss_v = float(batch_loss)
    if not (loss_v == loss_v) or loss_v == float("inf") or loss_v == float("-inf"):
        return state

    # frequency weights
    counts = torch.bincount(tids, minlength=nt).to(dtype=torch.float32)
    freq = _safe_normalize_probs(counts)

    prev = state.ema_loss
    prev_filled = torch.where(torch.isfinite(prev), prev, torch.full_like(prev, loss_v))
    beta = float(ema_beta)
    new_ema = (1.0 - beta) * prev_filled + beta * (freq * loss_v + (1.0 - freq) * prev_filled)

    # convert EMA -> sampling weights: higher loss => higher prob
    centered = new_ema - torch.nanmin(new_ema)
    centered = torch.where(torch.isfinite(centered), centered, torch.zeros_like(centered))
    temp = float(temperature)
    temp = 1e-6 if temp <= 0.0 else temp
    w = torch.pow(centered + 1e-6, 1.0 / temp)
    p = _safe_normalize_probs(w)

    mp = float(min_prob)
    if mp > 0.0:
        p = _safe_normalize_probs(torch.clamp(p, min=mp))

    return TaskSamplingState(task_probs=p, ema_loss=new_ema)


class MultitaskWindowSampler(torch.utils.data.Sampler[int]):
    """Sample dataset indices by task probability for mixed multitask PKLs.

    This sampler assumes the dataset has:
    - num_trajectories: int
    - size_of_trajectory: int
    - block_size: int
    - task_indices: Tensor/ndarray mapping traj_idx -> task_id
    """

    def __init__(
        self,
        dataset,
        *,
        task_probs_getter: Callable[[], torch.Tensor],
        num_samples: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        super().__init__(dataset)
        self.dataset = dataset
        self.task_probs_getter = task_probs_getter
        self.num_samples = int(num_samples) if num_samples is not None else int(len(dataset))
        self.seed = int(seed)

        n_traj = int(getattr(dataset, "num_trajectories"))
        ti = getattr(dataset, "task_indices", None)
        if ti is None:
            raise ValueError("MultitaskWindowSampler requires dataset.task_indices")
        if torch.is_tensor(ti):
            t_traj = ti
        else:
            t_traj = torch.as_tensor(ti)
        if t_traj.ndim == 2:
            t_traj = t_traj[:, 0]
        if int(t_traj.shape[0]) != n_traj:
            raise ValueError(
                f"task_indices length mismatch: {int(t_traj.shape[0])} vs num_trajectories={n_traj}"
            )
        self.traj_task_ids = t_traj.to(dtype=torch.long, device="cpu")

        self.stride = int(getattr(dataset, "size_of_trajectory")) - int(getattr(dataset, "block_size")) + 1
        if self.stride <= 0:
            raise ValueError("Invalid stride computed from size_of_trajectory and block_size")

    def __len__(self) -> int:  # pragma: no cover
        return int(self.num_samples)

    def __iter__(self):
        gen = torch.Generator(device="cpu")
        # make each epoch deterministic but different
        gen.manual_seed(self.seed)

        task_probs = self.task_probs_getter().detach().to(dtype=torch.float32, device="cpu")
        # map task probs -> per-trajectory probs
        traj_w = task_probs[self.traj_task_ids.clamp(min=0, max=task_probs.numel() - 1)]
        traj_p = _safe_normalize_probs(traj_w)

        traj_idx = torch.multinomial(traj_p, num_samples=self.num_samples, replacement=True, generator=gen)
        ctx = torch.randint(low=0, high=self.stride, size=(self.num_samples,), generator=gen, device="cpu")
        idx = traj_idx * self.stride + ctx
        return iter(idx.tolist())


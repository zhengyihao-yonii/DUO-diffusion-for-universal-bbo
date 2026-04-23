import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Tuple
import pdb
import time

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


def _task_idx_to_one_hot(task_idx, model):
    """
    task 类别索引 -> [batch, num_tasks] one-hot，供 task_mlp(nn.Linear(num_tasks, dim)) 使用。
    已展开的 one-hot 则原样返回。
    """
    if task_idx is None:
        return None
    if not torch.is_tensor(task_idx):
        task_idx = torch.as_tensor(task_idx)
    # 已是 [B, num_tasks]
    if task_idx.dim() == 2 and task_idx.shape[-1] > 1:
        return task_idx.float()
    # batchify / DataLoader 常得到 [B, 1]（对 np.array([idx]) 多了一维）
    if task_idx.dim() == 2 and task_idx.shape[-1] == 1:
        task_idx = task_idx.squeeze(-1)
    if task_idx.dim() == 0:
        task_idx = task_idx.unsqueeze(0)
    if task_idx.dim() == 1:
        task_idx = task_idx.long()
        n_cls = getattr(model, "num_tasks", None)
        if n_cls is None:
            n_cls = int(task_idx.max().item()) + 1
        return F.one_hot(task_idx, num_classes=n_cls).float()
    return task_idx.float()


def _cond_text_embed(cond_or_state):
    """Optional [batch, D] frozen text embedding from conditions dict (task metadata)."""
    if not isinstance(cond_or_state, dict):
        return None
    return cond_or_state.get('text_embed')


def _env_float(key: str, default: float = 0.0) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_truthy(key: str) -> bool:
    """True when env var is set to 1/true/yes/on (case-insensitive)."""
    raw = os.environ.get(key, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _sample_timesteps_biased_small(
    batch_size: int,
    n_timesteps: int,
    bias_power: float,
    device: torch.device,
) -> torch.Tensor:
    """
    训练用：提高离散小 t 的采样概率（与均匀 ``torch.randint`` 并存为可选分支）。
    u ~ Uniform(0,1)，t = floor(u ** (1 + bias_power) * n_timesteps)；bias_power>0 时更常采到小 t。
    """
    u = torch.rand(batch_size, device=device)
    exp = 1.0 + float(bias_power)
    return (u**exp * float(n_timesteps)).long().clamp(0, n_timesteps - 1)


def _p_losses_min_snr_weighted(
    loss_fn: nn.Module,
    x_recon: torch.Tensor,
    targ: torch.Tensor,
    t: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    gamma: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    min-SNR 风格逐样本加权后再对 batch 取均值（仅训练损失；不改变采样）。
    权重 w = min(SNR(t), gamma) / SNR(t)，SNR = alpha_bar / (1 - alpha_bar)。
    """
    elem = loss_fn._loss(x_recon, targ)
    w_spatial = loss_fn.weights
    while w_spatial.dim() < elem.dim():
        w_spatial = w_spatial.unsqueeze(0)
    weighted_elem = elem * w_spatial
    per_sample = weighted_elem.reshape(weighted_elem.shape[0], -1).mean(dim=1)
    ac = extract(alphas_cumprod, t, (t.shape[0],)).reshape(-1).to(dtype=per_sample.dtype)
    snr = ac / (1.0 - ac + 1e-8)
    g = torch.as_tensor(float(gamma), device=per_sample.device, dtype=per_sample.dtype)
    w_t = torch.minimum(snr, g) / (snr + 1e-8)
    loss = (per_sample * w_t).mean()
    if hasattr(loss_fn, "action_dim"):
        ad = int(loss_fn.action_dim)
        a0 = (elem[:, :, -ad:] / loss_fn.weights[0, -ad:]).mean()
    else:
        a0 = elem.mean()
    return loss, {"a0_loss": a0}


def _per_t_bin_weighted_mse_metrics(
    loss_fn: nn.Module,
    n_timesteps: int,
    x_recon: torch.Tensor,
    targ: torch.Tensor,
    t: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Per-discrete-timestep-bin diffusion MSE (same weighting as WeightedL1/WeightedL2).

    Keys ``train/t_loss/b{idx}_t{lo}_{hi}`` are meant for wandb time-series vs training
    step (Trainer logs ``steps``). Smaller ``t`` → smaller bin index (for typical
    schedules, lower ``t`` is closer to clean data).
    """
    if not _env_truthy("DUO_LOG_PER_T_LOSS"):
        return {}
    w = getattr(loss_fn, "weights", None)
    if w is None or not torch.is_tensor(t):
        return {}
    if t.is_floating_point():
        return {}

    raw = F.mse_loss(x_recon, targ, reduction="none")
    w_exp = w
    while w_exp.dim() < raw.dim():
        w_exp = w_exp.unsqueeze(0)
    per_el = raw * w_exp
    per_sample = per_el.reshape(raw.shape[0], -1).mean(dim=1)

    n_t = max(1, int(n_timesteps))
    nb = max(2, min(_env_int("DUO_LOG_PER_T_LOSS_BINS", 20), n_t))
    t_flat = t.long().view(-1)
    out: Dict[str, torch.Tensor] = {}
    for b in range(nb):
        lo = b * n_t // nb
        if b < nb - 1:
            hi = (b + 1) * n_t // nb
            mask = (t_flat >= lo) & (t_flat < hi)
            hi_key = hi - 1
        else:
            mask = t_flat >= lo
            hi_key = n_t - 1
        if mask.any():
            out[f"train/t_loss/b{b:02d}_t{lo}_{hi_key}"] = per_sample[mask].mean()
    return out


def _env_discrete_ce_task_names() -> Set[str]:
    raw = os.environ.get("DUO_DISCRETE_CE_TASK_NAMES", "tfbind8,tfbind10")
    return {s.strip() for s in raw.split(",") if s.strip()}


def _tfbind_obs_flat_dim(task_name: str) -> Optional[int]:
    """TF-Bind design_bench logits: L×(C−1) with C=4 → 3 logits per position."""
    if task_name == "tfbind8":
        return 8 * 3
    if task_name == "tfbind10":
        return 10 * 3
    return None


def _batch_task_names(
    cond: Dict,
    train_tasks_list: Tuple[str, ...],
    batch_size: int,
    device: torch.device,
) -> Optional[List[str]]:
    """
    每条样本的任务名：优先 ``cond['task_idx']``；若无且仅单任务训练则全员该任务。
    """
    if not train_tasks_list:
        return None
    tid = cond.get("task_idx") if isinstance(cond, dict) else None
    if tid is None:
        if len(train_tasks_list) == 1:
            return [train_tasks_list[0]] * batch_size
        return None
    if not torch.is_tensor(tid):
        tid = torch.as_tensor(tid, device=device)
    tid = tid.view(-1).long()
    if tid.shape[0] != batch_size:
        return None
    out: List[str] = []
    for i in range(batch_size):
        ix = int(tid[i].item())
        if ix < 0 or ix >= len(train_tasks_list):
            return None
        out.append(train_tasks_list[ix])
    return out


def _discrete_tfbind_ce(
    x0_pred: torch.Tensor,
    x0_true: torch.Tensor,
    *,
    train_tasks_list: Tuple[str, ...],
    ce_task_names: Set[str],
    batch_task_names: Optional[List[str]],
) -> torch.Tensor:
    """
    辅助 CE：对每个 TF-Bind 样本，将 x0 的前 L×3 维 reshape 为 (H,L,3)，补零得到 4 类 logits，
    目标类为 GT（同空间）在各位置 argmax。若无可计算样本则返回 0。
    """
    device = x0_pred.device
    if batch_task_names is None:
        return torch.zeros((), device=device, dtype=x0_pred.dtype)
    B, H, W = x0_pred.shape
    ces: List[torch.Tensor] = []
    for b in range(B):
        name = batch_task_names[b]
        if name not in ce_task_names:
            continue
        D = _tfbind_obs_flat_dim(name)
        if D is None or D > W:
            continue
        pp = x0_pred[b, :, :D].reshape(H, -1, 3)
        pt = x0_true[b, :, :D].reshape(H, -1, 3)
        # design-bench to_integers：在最后一维左侧 pad 一个 0 再 argmax
        zeros = pp.new_zeros(H, pp.shape[1], 1)
        logits_p = torch.cat([zeros, pp], dim=-1)
        logits_t = torch.cat([zeros, pt], dim=-1)
        targ = logits_t.argmax(dim=-1).reshape(-1)
        logits_flat = logits_p.reshape(-1, 4)
        ces.append(F.cross_entropy(logits_flat, targ, reduction="mean"))
    if not ces:
        return torch.zeros((), device=device, dtype=x0_pred.dtype)
    return torch.stack(ces).mean()


def epsilon_task_text_cfg(
    diffusion: nn.Module,
    x: torch.Tensor,
    cond,
    t,
    returns,
    task_idx,
    text_embed: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Task × text 联合 classifier-free guidance（双线性形式，与 returns 的 CFG 独立）。
    当 condition_guidance_w_task / condition_guidance_w_text 均为 0 时，等价于单次前向（全条件）。
    """
    m = diffusion.model
    tc = getattr(m, "task_condition", False)
    txc = getattr(m, "text_condition", False)
    wt = float(getattr(diffusion, "condition_guidance_w_task", 0.0))
    wx = float(getattr(diffusion, "condition_guidance_w_text", 0.0))
    if not getattr(diffusion, "cfg_apply_task", True):
        wt = 0.0
    if not getattr(diffusion, "cfg_apply_text", True):
        wx = 0.0
    use_t_emb = getattr(diffusion, "sample_with_task_embedding", True)
    use_x_emb = getattr(diffusion, "sample_with_text_embedding", True)
    if not use_t_emb:
        wt = 0.0
    if not use_x_emb:
        wx = 0.0

    def fwd(ft: bool, fx: bool):
        return diffusion.model(
            x,
            cond,
            t,
            returns,
            task_idx=task_idx,
            text_embed=text_embed,
            use_dropout=False,
            force_dropout=False,
            force_task_dropout=ft,
            force_text_dropout=fx,
        )

    if not tc and not txc:
        return fwd(False, False)

    ft_full = True if not use_t_emb else False
    fx_full = True if not use_x_emb else False

    if wt <= 0 and wx <= 0:
        return fwd(ft_full, fx_full)

    if tc and not txc and wt > 0:
        e0 = fwd(True, False)
        e1 = fwd(False, False)
        return e0 + wt * (e1 - e0)

    if not tc and txc and wx > 0:
        e0 = fwd(False, True)
        e1 = fwd(False, False)
        return e0 + wx * (e1 - e0)

    if tc and txc:
        if wt > 0 and wx > 0:
            e00 = fwd(True, True)
            e10 = fwd(False, True)
            e01 = fwd(True, False)
            e11 = fwd(False, False)
            return (
                e00
                + wt * (e10 - e00)
                + wx * (e01 - e00)
                + (wt * wx) * (e11 - e10 - e01 + e00)
            )
        if wt > 0 and wx <= 0:
            e0 = fwd(True, False)
            e1 = fwd(False, False)
            return e0 + wt * (e1 - e0)
        if wt <= 0 and wx > 0:
            e0 = fwd(False, True)
            e1 = fwd(False, False)
            return e0 + wx * (e1 - e0)

    return fwd(ft_full, fx_full)


class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, n_sample_timesteps=200,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
        condition_guidance_w=0.1,
        condition_guidance_w_task=0.0,
        condition_guidance_w_text=0.0,
        cfg_apply_task=True,
        cfg_apply_text=True,
        sample_with_task_embedding=True,
        sample_with_text_embedding=True,
        train_tasks_list: Optional[Tuple[str, ...]] = None,
        train_timestep_bias_power: float = 0.0,
        train_loss_min_snr_gamma: float = 0.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        self.condition_guidance_w_task = float(condition_guidance_w_task)
        self.condition_guidance_w_text = float(condition_guidance_w_text)
        self.cfg_apply_task = bool(cfg_apply_task)
        self.cfg_apply_text = bool(cfg_apply_text)
        self.sample_with_task_embedding = bool(sample_with_task_embedding)
        self.sample_with_text_embedding = bool(sample_with_text_embedding)
        self._train_tasks_list: Tuple[str, ...] = tuple(train_tasks_list or ())

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.n_sample_timesteps = int(n_sample_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)
        # 可选训练策略（默认 0 关闭，不改变原 loss / 均匀 t 采样路径）
        self._train_ts_bias = float(train_timestep_bias_power)
        self._train_min_snr_gamma = float(train_loss_min_snr_gamma)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[:, self.observation_dim:] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.model.calc_energy:
            assert self.predict_epsilon
            x = torch.tensor(x, requires_grad=True)
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns = torch.tensor(returns, requires_grad=True)

        # Extract task_idx from cond if present
        task_idx = cond.get('task_idx') if isinstance(cond, dict) else None
        task_idx = _task_idx_to_one_hot(task_idx, self.model)
        text_embed = _cond_text_embed(cond)

        if self.returns_condition:
            # epsilon could be epsilon or x0 itself（returns CFG；不与此处 task/text 联合 CFG 混用）
            epsilon_cond = self.model(
                x, cond, t, returns, task_idx, text_embed=text_embed,
                use_dropout=False, force_dropout=False,
                force_task_dropout=False, force_text_dropout=False,
            )
            epsilon_uncond = self.model(
                x, cond, t, returns, task_idx, text_embed=text_embed,
                force_dropout=True,
                force_task_dropout=False, force_text_dropout=False,
            )
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = epsilon_task_text_cfg(self, x, cond, t, returns, task_idx, text_embed)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (1.0 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        returns=None,
        verbose=True,
        return_diffusion=False,
        values=None,
        step_callback=None,
        step_callback_stride=1,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, values=values)

        if return_diffusion:
            diffusion = [x]

        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        # for i in reversed(range(0, self.n_timesteps)):
        progress = utils.Progress(self.n_sample_timesteps) if verbose else utils.Silent()
        start_time = time.time()
        stride = max(1, int(step_callback_stride))
        loop_indices = list(reversed(range(0, self.n_sample_timesteps)))
        for step_ord, i in enumerate(loop_indices):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, values=values)

            progress.update({'t': i})

            if return_diffusion:
                diffusion.append(x)

            if step_callback is not None and (
                step_ord % stride == 0 or i == 0
            ):
                step_callback(
                    timestep_index=int(i),
                    step_ordinal=int(step_ord),
                    total_steps=int(self.n_sample_timesteps),
                    x=x,
                )
        end_time = time.time() - start_time
        progress.close()
        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x, end_time

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, values=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond["ctx_len"])
        # batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, returns, values=values, *args, **kwargs)
    
    @torch.no_grad()
    def back_and_forth_sample(self, x_start, cond, returns=None, verbose=True, return_diffusion=False, values=None):
        device = self.betas.device
        batch_size = len(cond["ctx_len"])
        # batch_size = len(cond[0])
        shape = x_start.shape
        
        device = self.betas.device

        batch_size = shape[0]
        # x = 0.5*torch.randn(shape, device=device)
        t = torch.ones((batch_size, )).to(dtype=torch.long, device=device) * (self.n_timesteps // 2)
        noise = torch.randn_like(x_start)

        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x = apply_conditioning(x, cond)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps // 2) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps // 2)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, values=values)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
        
    def grad_p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def grad_p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.grad_p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def grad_conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.grad_p_sample_loop(shape, cond, returns, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        noise = torch.randn_like(x_start)

        if self.predict_epsilon:
            # Cause we condition on obs at t=0
            # noise[:, 0, self.action_dim:] = 0
            noise[:, 0, :] = 0

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond)

        if self.model.calc_energy:
            assert self.predict_epsilon
            x_noisy.requires_grad = True
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns.requires_grad = True
            noise.requires_grad = True

        task_idx = cond.get('task_idx') if isinstance(cond, dict) else None
        task_idx = _task_idx_to_one_hot(task_idx, self.model)
        text_embed = _cond_text_embed(cond)
        x_recon = self.model(
            x_noisy, cond, t, returns, task_idx=task_idx, text_embed=text_embed
        )

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            if self._train_min_snr_gamma > 0.0:
                loss, info = _p_losses_min_snr_weighted(
                    self.loss_fn,
                    x_recon,
                    noise,
                    t,
                    self.alphas_cumprod,
                    float(self._train_min_snr_gamma),
                )
            else:
                loss, info = self.loss_fn(x_recon, noise)
        else:
            if self._train_min_snr_gamma > 0.0:
                loss, info = _p_losses_min_snr_weighted(
                    self.loss_fn,
                    x_recon,
                    x_start,
                    t,
                    self.alphas_cumprod,
                    float(self._train_min_snr_gamma),
                )
            else:
                loss, info = self.loss_fn(x_recon, x_start)

        lam = _env_float("DUO_DISCRETE_CE_LAMBDA", 0.0)
        if lam > 0 and self.predict_epsilon:
            eps_hat = x_recon
            x0_hat = self.predict_start_from_noise(x_noisy, t, eps_hat)
            x0_tgt = x_start
            names = _batch_task_names(cond, self._train_tasks_list, x_start.shape[0], x_start.device)
            ce = _discrete_tfbind_ce(
                x0_hat,
                x0_tgt,
                train_tasks_list=self._train_tasks_list,
                ce_task_names=_env_discrete_ce_task_names(),
                batch_task_names=names,
            )
            loss = loss + lam * ce
            info = dict(info)
            info["ce_loss"] = ce.detach()
            info["discrete_ce_lambda"] = torch.tensor(lam, device=loss.device)

        _t_for_bins = t
        if torch.is_tensor(_t_for_bins) and not _t_for_bins.is_floating_point():
            _targ = noise if self.predict_epsilon else x_start
            _extra = _per_t_bin_weighted_mse_metrics(
                self.loss_fn,
                int(self.n_timesteps),
                x_recon,
                _targ,
                _t_for_bins,
            )
            if _extra:
                info = dict(info)
                info.update(_extra)

        return loss, info

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        if self._train_ts_bias > 0.0:
            t = _sample_timesteps_biased_small(
                batch_size, self.n_timesteps, self._train_ts_bias, x.device
            )
        else:
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, returns)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

class GaussianInvDynDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True, hidden_dim=256,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
        condition_guidance_w=0.1,
        condition_guidance_w_task=0.0,
        condition_guidance_w_text=0.0,
        cfg_apply_task=True,
        cfg_apply_text=True,
        sample_with_task_embedding=True,
        sample_with_text_embedding=True,
        ar_inv=False, train_only_inv=False,
        train_tasks_list: Optional[Tuple[str, ...]] = None,
        train_timestep_bias_power: float = 0.0,
        train_loss_min_snr_gamma: float = 0.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.ar_inv = ar_inv
        self.train_only_inv = train_only_inv
        if self.ar_inv:
            self.inv_model = ARInvModel(hidden_dim=hidden_dim, observation_dim=observation_dim, action_dim=action_dim)
        else:
            self.inv_model = nn.Sequential(
                nn.Linear(2 * self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.action_dim),
            )
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        self.condition_guidance_w_task = float(condition_guidance_w_task)
        self.condition_guidance_w_text = float(condition_guidance_w_text)
        self.cfg_apply_task = bool(cfg_apply_task)
        self.cfg_apply_text = bool(cfg_apply_text)
        self.sample_with_task_embedding = bool(sample_with_task_embedding)
        self.sample_with_text_embedding = bool(sample_with_text_embedding)
        self._train_tasks_list: Tuple[str, ...] = tuple(train_tasks_list or ())

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses['state_l2'](loss_weights)
        self._train_ts_bias = float(train_timestep_bias_power)
        self._train_min_snr_gamma = float(train_loss_min_snr_gamma)

    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = 1
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        # Extract task_idx from cond if present
        task_idx = cond.get('task_idx') if isinstance(cond, dict) else None
        task_idx = _task_idx_to_one_hot(task_idx, self.model)
        text_embed = _cond_text_embed(cond)

        if self.returns_condition:
            epsilon_cond = self.model(
                x, cond, t, returns, task_idx, text_embed=text_embed,
                use_dropout=False, force_dropout=False,
                force_task_dropout=False, force_text_dropout=False,
            )
            epsilon_uncond = self.model(
                x, cond, t, returns, task_idx, text_embed=text_embed,
                force_dropout=True,
                force_task_dropout=False, force_text_dropout=False,
            )
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = epsilon_task_text_cfg(self, x, cond, t, returns, task_idx, text_embed)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        x = apply_conditioning(x, cond)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)
    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, 0)

        # Extract task_idx from cond if present
        task_idx = cond.get('task_idx') if isinstance(cond, dict) else None
        task_idx = _task_idx_to_one_hot(task_idx, self.model)
        text_embed = _cond_text_embed(cond)

        x_recon = self.model(x_noisy, cond, t, returns, task_idx, text_embed=text_embed)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, 0)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            if self._train_min_snr_gamma > 0.0:
                loss, info = _p_losses_min_snr_weighted(
                    self.loss_fn,
                    x_recon,
                    noise,
                    t,
                    self.alphas_cumprod,
                    float(self._train_min_snr_gamma),
                )
            else:
                loss, info = self.loss_fn(x_recon, noise)
        else:
            if self._train_min_snr_gamma > 0.0:
                loss, info = _p_losses_min_snr_weighted(
                    self.loss_fn,
                    x_recon,
                    x_start,
                    t,
                    self.alphas_cumprod,
                    float(self._train_min_snr_gamma),
                )
            else:
                loss, info = self.loss_fn(x_recon, x_start)

        lam = _env_float("DUO_DISCRETE_CE_LAMBDA", 0.0)
        if lam > 0 and self.predict_epsilon:
            eps_hat = x_recon
            x0_hat = self.predict_start_from_noise(x_noisy, t, eps_hat)
            x0_tgt = x_start
            names = _batch_task_names(cond, self._train_tasks_list, x_start.shape[0], x_start.device)
            ce = _discrete_tfbind_ce(
                x0_hat,
                x0_tgt,
                train_tasks_list=self._train_tasks_list,
                ce_task_names=_env_discrete_ce_task_names(),
                batch_task_names=names,
            )
            loss = loss + lam * ce
            info = dict(info)
            info["ce_loss"] = ce.detach()
            info["discrete_ce_lambda"] = torch.tensor(lam, device=loss.device)

        if torch.is_tensor(t) and not t.is_floating_point():
            _targ = noise if self.predict_epsilon else x_start
            _extra = _per_t_bin_weighted_mse_metrics(
                self.loss_fn,
                int(self.n_timesteps),
                x_recon,
                _targ,
                t,
            )
            if _extra:
                info = dict(info)
                info.update(_extra)

        return loss, info

    def loss(self, x, cond, returns=None):
        if self.train_only_inv:
            # Calculating inv loss
            x_t = x[:, :-1, self.action_dim:]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x[:, 1:, self.action_dim:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                loss = self.inv_model.calc_loss(x_comb_t, a_t)
                info = {'a0_loss':loss}
            else:
                pred_a_t = self.inv_model(x_comb_t)
                loss = F.mse_loss(pred_a_t, a_t)
                info = {'a0_loss': loss}
        else:
            batch_size = len(x)
            if self._train_ts_bias > 0.0:
                t = _sample_timesteps_biased_small(
                    batch_size, self.n_timesteps, self._train_ts_bias, x.device
                )
            else:
                t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
            diffuse_loss, info = self.p_losses(x[:, :, self.action_dim:], cond, t, returns)
            # Calculating inv loss
            x_t = x[:, :-1, self.action_dim:]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x[:, 1:, self.action_dim:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                inv_loss = self.inv_model.calc_loss(x_comb_t, a_t)
            else:
                pred_a_t = self.inv_model(x_comb_t)
                inv_loss = F.mse_loss(pred_a_t, a_t)

            loss = (1 / 2) * (diffuse_loss + inv_loss)

        return loss, info

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)


class ARInvModel(nn.Module):
    def __init__(self, hidden_dim, observation_dim, action_dim, low_act=-1.0, up_act=1.0):
        super(ARInvModel, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.action_embed_hid = 128
        self.out_lin = 128
        self.num_bins = 80

        self.up_act = up_act
        self.low_act = low_act
        self.bin_size = (self.up_act - self.low_act) / self.num_bins
        self.ce_loss = nn.CrossEntropyLoss()

        self.state_embed = nn.Sequential(
            nn.Linear(2 * self.observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.lin_mod = nn.ModuleList([nn.Linear(i, self.out_lin) for i in range(1, self.action_dim)])
        self.act_mod = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, self.action_embed_hid), nn.ReLU(),
                                                    nn.Linear(self.action_embed_hid, self.num_bins))])

        for _ in range(1, self.action_dim):
            self.act_mod.append(
                nn.Sequential(nn.Linear(hidden_dim + self.out_lin, self.action_embed_hid), nn.ReLU(),
                              nn.Linear(self.action_embed_hid, self.num_bins)))

    def forward(self, comb_state, deterministic=False):
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        lp_0 = self.act_mod[0](state_d)
        l_0 = torch.distributions.Categorical(logits=lp_0).sample()

        if deterministic:
            a_0 = self.low_act + (l_0 + 0.5) * self.bin_size
        else:
            a_0 = torch.distributions.Uniform(self.low_act + l_0 * self.bin_size,
                                              self.low_act + (l_0 + 1) * self.bin_size).sample()

        a = [a_0.unsqueeze(1)]

        for i in range(1, self.action_dim):
            lp_i = self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](torch.cat(a, dim=1))], dim=1))
            l_i = torch.distributions.Categorical(logits=lp_i).sample()

            if deterministic:
                a_i = self.low_act + (l_i + 0.5) * self.bin_size
            else:
                a_i = torch.distributions.Uniform(self.low_act + l_i * self.bin_size,
                                                  self.low_act + (l_i + 1) * self.bin_size).sample()

            a.append(a_i.unsqueeze(1))

        return torch.cat(a, dim=1)

    def calc_loss(self, comb_state, action):
        eps = 1e-8
        action = torch.clamp(action, min=self.low_act + eps, max=self.up_act - eps)
        l_action = torch.div((action - self.low_act), self.bin_size, rounding_mode='floor').long()
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        loss = self.ce_loss(self.act_mod[0](state_d), l_action[:, 0])

        for i in range(1, self.action_dim):
            loss += self.ce_loss(self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](action[:, :i])], dim=1)),
                                     l_action[:, i])

        return loss/self.action_dim


class ActionGaussianDiffusion(nn.Module):
    # Assumes horizon=1
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
        condition_guidance_w=0.1,
        condition_guidance_w_task=0.0,
        condition_guidance_w_text=0.0,
        cfg_apply_task=True,
        cfg_apply_text=True,
        sample_with_task_embedding=True,
        sample_with_text_embedding=True,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        self.condition_guidance_w_task = float(condition_guidance_w_task)
        self.condition_guidance_w_text = float(condition_guidance_w_text)
        self.cfg_apply_task = bool(cfg_apply_task)
        self.cfg_apply_text = bool(cfg_apply_text)
        self.sample_with_task_embedding = bool(sample_with_task_embedding)
        self.sample_with_text_embedding = bool(sample_with_text_embedding)

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.model.calc_energy:
            assert self.predict_epsilon
            x = torch.tensor(x, requires_grad=True)
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns = torch.tensor(returns, requires_grad=True)

        # Extract task_idx from cond if present
        task_idx = cond.get('task_idx') if isinstance(cond, dict) else None
        task_idx = _task_idx_to_one_hot(task_idx, self.model)
        text_embed = _cond_text_embed(cond)

        if self.returns_condition:
            epsilon_cond = self.model(
                x, cond, t, returns, task_idx, text_embed=text_embed,
                use_dropout=False, force_dropout=False,
                force_task_dropout=False, force_text_dropout=False,
            )
            epsilon_uncond = self.model(
                x, cond, t, returns, task_idx, text_embed=text_embed,
                force_dropout=True,
                force_task_dropout=False, force_text_dropout=False,
            )
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = epsilon_task_text_cfg(self, x, cond, t, returns, task_idx, text_embed)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        shape = (batch_size, self.action_dim)
        cond = cond[0]
        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    def grad_p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def grad_p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def grad_conditional_sample(self, cond, returns=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        shape = (batch_size, self.action_dim)
        cond = cond[0]
        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)
    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, action_start, state, t, returns=None):
        noise = torch.randn_like(action_start)
        action_noisy = self.q_sample(x_start=action_start, t=t, noise=noise)

        if self.model.calc_energy:
            assert self.predict_epsilon
            action_noisy.requires_grad = True
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns.requires_grad = True
            noise.requires_grad = True

        # Extract task_idx from state if present
        task_idx = state.get('task_idx') if isinstance(state, dict) else None
        task_idx = _task_idx_to_one_hot(task_idx, self.model)
        text_embed = _cond_text_embed(state)

        pred = self.model(action_noisy, state, t, returns, task_idx, text_embed=text_embed)

        assert noise.shape == pred.shape

        if self.predict_epsilon:
            loss = F.mse_loss(pred, noise)
        else:
            loss = F.mse_loss(pred, action_start)

        return loss, {'a0_loss':loss}

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        assert x.shape[1] == 1 # Assumes horizon=1
        x = x[:,0,:]
        cond = x[:,self.action_dim:] # Observation
        x = x[:,:self.action_dim] # Action
        return self.p_losses(x, cond, t, returns)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)


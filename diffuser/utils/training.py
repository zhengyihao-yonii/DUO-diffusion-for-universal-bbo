from __future__ import annotations

import os
import copy
import numpy as np
import torch
import einops
import pdb
import diffuser
from copy import deepcopy
from tqdm import tqdm

from .arrays import batch_to_device, to_np, to_device, apply_dict, to_torch
from .timer import Timer
from .cloud import sync_logs
from ml_logger import logger


def _safe_wandb_log(
    metrics: dict,
    step: int | None = None,
    *,
    wandb_step_key: str | None = None,
) -> None:
    """避免未 init 或离线失败时 wandb.log 触发底层 abort。

    - **全局 step**（``wandb.log(..., step=)``）：单段训练时与 ``self.step`` 对齐。
    - **独立横轴**（``wandb_step_key``）：``metrics`` 内须含该键；**不要**再传 ``step=``，供
      ``wandb.define_metric(..., step_metric=...)`` 使用。Proxy 与扩散先后训练时，用
      ``proxy_step`` / ``finetune_step`` 分离坐标，避免单调步冲突且微调曲线从 0 起画。
    """
    try:
        import wandb as _wandb

        if getattr(_wandb, "run", None) is None:
            return
        if wandb_step_key is not None:
            if wandb_step_key not in metrics:
                return
            _wandb.log(metrics)
            return
        st = step
        if st is None and metrics.get("steps") is not None:
            try:
                st = int(metrics["steps"])
            except (TypeError, ValueError):
                st = None
        if st is not None:
            _wandb.log(metrics, step=int(st))
        else:
            _wandb.log(metrics)
    except Exception:
        pass


def configure_wandb_step_axes(*, include_proxy_axis: bool) -> None:
    """在 ``wandb.init`` 之后、首次 ``log`` 之前调用：为 proxy / 扩散登记独立 step 轴。"""
    try:
        import wandb as wb

        if getattr(wb, "run", None) is None:
            return
        wb.define_metric("finetune_step")
        for _m in (
            "loss",
            "lr",
            "steps",
            "a0_loss",
            "epoch",
            "total_epochs",
            "training_mode",
            "training_completed",
            "final_step",
        ):
            wb.define_metric(_m, step_metric="finetune_step")
        wb.define_metric("train/*", step_metric="finetune_step")
        if include_proxy_axis:
            wb.define_metric("proxy_step")
            wb.define_metric("proxy/*", step_metric="proxy_step")
    except Exception:
        pass


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(key: str, default: str = "") -> str:
    v = os.environ.get(key, "").strip()
    return v if v else default


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = float(lr)


def _lr_warmup_cosine(
    *,
    base_lr: float,
    step: int,
    total_steps: int,
    warmup_frac: float,
    min_ratio: float,
) -> float:
    """
    Warmup + cosine decay.

    - step: 0-based global step
    - total_steps: total number of steps for the schedule
    - warmup_frac: fraction of steps for linear warmup (0..1)
    - min_ratio: minimum lr ratio at the end (0..1), so min_lr = base_lr * min_ratio
    """
    if total_steps <= 0:
        return float(base_lr)
    s = max(0, min(int(step), int(total_steps)))
    t = int(total_steps)
    wf = float(warmup_frac)
    wf = 0.0 if wf < 0.0 else (1.0 if wf > 1.0 else wf)
    w_steps = int(round(wf * t))
    min_r = float(min_ratio)
    if min_r < 0.0:
        min_r = 0.0
    if min_r > 1.0:
        min_r = 1.0
    min_lr = float(base_lr) * min_r

    if w_steps > 0 and s < w_steps:
        # Linear warmup from 0 -> base_lr
        return float(base_lr) * (float(s + 1) / float(w_steps))

    # Cosine decay from base_lr -> min_lr
    import math

    denom = max(1, t - w_steps)
    prog = float(s - w_steps) / float(denom)  # 0..1
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return float(min_lr + (float(base_lr) - float(min_lr)) * cos)


def cycle(dl):
    """无限重复遍历 ``dl``。若 ``dl`` 因 ``drop_last=True`` 且样本数不足等原因长度为 0，
    则内层 ``for`` 永不执行，旧实现会在 ``while True`` 中空转占满 CPU 且永不 yield。"""
    while True:
        empty = True
        for data in dl:
            empty = False
            yield data
        if empty:
            raise RuntimeError(
                "DataLoader 未产生任何 batch（空迭代）。常见原因：len(dataset) < batch_size 且 "
                "drop_last=True。请减小 batch_size 或对不足一整批的数据使用 drop_last=False。"
            )

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        proxy_model,
        dataset,
        proxy_dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        proxy_train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        proxy_log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        proxy_save_freq=100,
        label_freq=100000,
        save_parallel=False,
        n_reference=8,
        bucket=None,
        train_device='cuda',
        save_checkpoints=False,
        load_checkpoint=None,
        load_checkpoint_path=None,
        load_proxy_checkpoint=None,
        proxy_save_prefix=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.proxy_model = proxy_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.save_checkpoints = save_checkpoints

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.proxy_log_freq = proxy_log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.proxy_save_freq = proxy_save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.proxy_dataset = proxy_dataset
        
        # 只有当proxy_dataset不为None时，才初始化与proxy model相关的组件
        self.has_proxy = proxy_dataset is not None
        if self.has_proxy:
            ranks = torch.argsort(torch.argsort(-1 * self.proxy_dataset.data_y.flatten()))
            weights = 1.0 / (1e-2 * len(self.proxy_dataset.data_y) + ranks)
            sampler = torch.utils.data.WeightedRandomSampler(
                    weights=weights, num_samples=len(self.proxy_dataset.data_y), replacement=True
                    )
            _n_proxy = int(len(self.proxy_dataset.data_y))
            _drop_proxy = _n_proxy >= train_batch_size
            self.proxy_dataloader = cycle(torch.utils.data.DataLoader(
                self.proxy_dataset, batch_size=train_batch_size, num_workers=0, sampler=sampler, pin_memory=True, drop_last=_drop_proxy,
            ))
        else:
            # 在没有proxy_dataset的情况下，仍然可以训练扩散模型
            self.proxy_dataloader = None

        _n_ds = len(self.dataset)
        _drop_main = _n_ds >= train_batch_size
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True, drop_last=_drop_main,
        ))
        # self.proxy_dataloader = cycle(torch.utils.data.DataLoader(
        #     self.proxy_dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True, drop_last=True,
        # ))
        
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)
        self._base_train_lr = float(train_lr)
        self._half_lr_switched = False
        self._lr_mult_factor = 1.0

        # Optional LR schedule (default constant to preserve historical behavior).
        # Use DUO_TRAIN_LR_SCHEDULE=warmup_cosine to enable.
        self._lr_schedule = _env_str("DUO_TRAIN_LR_SCHEDULE", "constant").lower()
        self._lr_warmup_frac = _env_float("DUO_TRAIN_LR_WARMUP_FRAC", 0.05)
        self._lr_min_ratio = _env_float("DUO_TRAIN_LR_MIN_RATIO", 0.1)
        
        # 只有当proxy_model不为None时，才创建proxy_optimizer
        if self.has_proxy:
            self.proxy_optimizer = torch.optim.Adam(proxy_model.parameters(), lr=proxy_train_lr)
        else:
            self.proxy_optimizer = None

        self.bucket = bucket
        self.n_reference = n_reference
        # 若设置，proxy 权重保存到该目录（用于多任务时为各任务单独训练 proxy）
        self.proxy_save_prefix = proxy_save_prefix

        self.reset_parameters()
        self.step = 0
        self.proxy_step = 0

        self.device = train_device
        
        if load_checkpoint_path is not None:
            self.load_from_path(load_checkpoint_path)
        elif load_checkpoint is not None:
            self.load(epoch=load_checkpoint)
            self.step = load_checkpoint
            
        # 只有当有proxy时才加载proxy checkpoint
        if self.has_proxy and load_proxy_checkpoint is not None:
            self.proxy_load(epoch=load_proxy_checkpoint)
            self.proxy_step = load_proxy_checkpoint

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#
    
    def train_proxy(self, n_train_steps):
        # 只有当有proxy时才训练proxy model
        if not self.has_proxy:
            logger.print("没有proxy_dataset，跳过proxy model训练")
            return
            
        timer = Timer()
        for step in tqdm(range(n_train_steps)):
            x, y = next(self.proxy_dataloader)
            # print(x[:4, :10])
            # print(y[:4])
            # print(self.proxy_dataset.normalizer.unnormalize(x[:4]))
            # print(self.proxy_dataset.normalizer_values.unnormalize(y[:4]))
            # print(kyle)
            # print(batch[0].shape, batch[1].shape)
            # batch = batch_to_device(batch, device=self.device)
            x = x.to(self.device)
            y = y.to(self.device)
            loss, infos = self.proxy_model.loss(x, y)
            loss.backward()

            # Optional grad clipping (disabled by default): set DUO_GRAD_CLIP_NORM, e.g. 1.0
            _clip = _env_float("DUO_GRAD_CLIP_NORM", 2.0)
            if _clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.proxy_model.parameters(), max_norm=_clip
                )
            
            self.proxy_optimizer.step()
            self.proxy_optimizer.zero_grad()
            
            if self.proxy_step % self.proxy_log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                logger.print(f'{self.proxy_step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')
                metrics = {k:v.detach().item() for k, v in infos.items()}
                metrics['proxy_steps'] = self.proxy_step
                metrics['proxy_loss'] = loss.detach().item()
                logger.log_metrics_summary(metrics, default_stats='mean')
                # proxy 使用 proxy_step 轴，避免与后续扩散 finetune_step 全局单调冲突
                wb_metrics = {
                    f"proxy/{k}": float(v.detach().item()) for k, v in infos.items()
                }
                wb_metrics["proxy_step"] = int(self.proxy_step)
                wb_metrics["proxy/loss"] = float(loss.detach().item())
                _safe_wandb_log(wb_metrics, wandb_step_key="proxy_step")

            self.proxy_step += 1

            if self.proxy_step % self.proxy_save_freq == 0:
                self.proxy_save()

    def train(self, n_train_steps):

        timer = Timer()
        # for step in tqdm(range(n_train_steps)):
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                # Drive optional two-stage schedule (tbias & LR switch) without changing signatures.
                if hasattr(self.model, "set_train_schedule_step"):
                    self.model.set_train_schedule_step(
                        int(self.step),
                        int(getattr(self, "_total_train_steps", 0)) or None,
                    )
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            # Optional: if model requests a late-stage LR multiplier, apply once when crossing the split.
            if (not self._half_lr_switched) and hasattr(self.model, "_train_half_ts_bias"):
                hp = float(getattr(self.model, "_train_half_ts_bias", 0.0))
                if hp > 0.0:
                    tot = int(getattr(self, "_total_train_steps", 0))
                    if tot > 0:
                        frac = float(getattr(self.model, "_train_half_ts_frac", 0.7))
                        split = int(frac * float(tot))
                        if int(self.step) >= split:
                            mult = float(getattr(self.model, "_train_half_lr_mult", 0.5))
                            self._lr_mult_factor = float(mult)
                            self._half_lr_switched = True

            # Optional: warmup+cosine LR schedule (big->small). Applied after gradient accumulation.
            if self._lr_schedule in ("warmup_cosine", "warmup-cosine", "cosine"):
                tot = int(getattr(self, "_total_train_steps", 0)) or int(n_train_steps)
                lr = _lr_warmup_cosine(
                    base_lr=self._base_train_lr,
                    step=int(self.step),
                    total_steps=int(tot),
                    warmup_frac=float(self._lr_warmup_frac),
                    min_ratio=float(self._lr_min_ratio),
                )
                lr = float(lr) * float(self._lr_mult_factor)
                _set_optimizer_lr(self.optimizer, lr)
            else:
                # Keep historical constant LR, but still honor optional late-stage multiplier.
                lr = float(self._base_train_lr) * float(self._lr_mult_factor)
                _set_optimizer_lr(self.optimizer, lr)

            # Optional grad clipping (disabled by default): set DUO_GRAD_CLIP_NORM, e.g. 1.0
            _clip = _env_float("DUO_GRAD_CLIP_NORM", 2.0)
            if _clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=_clip)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                logger.print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')
                metrics = {k:v.detach().item() for k, v in infos.items()}
                metrics['steps'] = self.step
                metrics['loss'] = loss.detach().item()
                metrics['lr'] = float(self.optimizer.param_groups[0].get("lr", self._base_train_lr))
                logger.log_metrics_summary(metrics, default_stats='mean')
                wb_metrics = dict(metrics)
                wb_metrics["finetune_step"] = int(self.step)
                _safe_wandb_log(wb_metrics, wandb_step_key="finetune_step")

            # if self.step == 0 and self.sample_freq:
            #     self.render_reference(self.n_reference)

            # if self.sample_freq and self.step % self.sample_freq == 0:
            #     if self.model.__class__ == diffuser.models.diffusion.GaussianInvDynDiffusion:
            #         self.inv_render_samples()
            #     elif self.model.__class__ == diffuser.models.diffusion.ActionGaussianDiffusion:
            #         pass
            #     else:
            #         self.render_samples()

            self.step += 1
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                self.save()
                
    def sample(self, n_samples, topk, horizon, context_length, inpainting, guidance, confidence=False):
        new_trajectories = []
        new_trajectories_confidence_score = []
        for step in tqdm(range(0, n_samples, self.batch_size)):
            batch = next(self.dataloader)
            batch = batch_to_device(batch, device=self.device)
            
            conditions = {i: to_torch(batch.trajectories[:, i], device=self.device) for i in range(context_length)}
            conditions["ctx_len"] = to_torch(np.ones(self.batch_size,), device=self.device) * context_length
           
            if inpainting:
                values = torch.ones(1, ).to(device=self.device).unsqueeze(0)
                values = values.repeat(self.batch_size, horizon-context_length)
                # values = torch.linspace(batch.trajectories[Config.horizon-context_length-1, -1], 1.0, steps=Config.horizon).to(device=device).unsqueeze(0)
            else:
                values = None
            
            if guidance:
                returns = torch.ones(1, ).to(device=self.device).unsqueeze(0)
                returns = returns.repeat(self.batch_size, 1)
                # returns = (to_torch(batch.trajectories[:, :context_length, -1].sum(axis=-1, keepdims=True), device=self.device) + horizon - context_length) / horizon
            else:
                # returns = torch.ones(1, ).to(device=self.device).unsqueeze(0) * 0.0
                # returns = returns.repeat(self.batch_size, 1)
                returns = None

            new_trajectory = self.ema_model.conditional_sample(conditions, values=values, returns=returns)
            # new_trajectory = self.ema_model.back_and_forth_sample(batch.trajectories, conditions, values=values, returns=returns)
            # 使用 VAE 将隐空间轨迹解码回原始设计空间
            with torch.no_grad():
                # 获取隐空间表示
                latent_observation = new_trajectory[..., context_length:, :-1].cpu()
                # 重塑以便通过 VAE（维度与 self.vae.latent_dim 一致）
                _ld = int(getattr(self.vae, "latent_dim", 32))
                latent_observation_reshaped = latent_observation.reshape(-1, _ld).to(self.device)
                # 使用VAE解码到原始空间
                decoded_observation = self.vae.decode(latent_observation_reshaped).cpu()
                # 确保维度正确，dkitty原始观测维度是12
                # 然后进行标准化以便传递给代理模型
                new_observation = self.proxy_dataset.normalizer.normalize(decoded_observation).to(self.device)
            new_trajectory_score, new_trajectory_confidence_score = self.proxy_model(new_observation, confidence=True)
            # print(new_trajectory_score.flatten()[:10])
            new_trajectory_score = self.dataset.normalizer_values.normalize(self.proxy_dataset.unnormalize_values(new_trajectory_score.cpu())).to(self.device)
            # print(new_trajectory[:, context_length:, -1].flatten()[:10])
            # print(new_trajectory_score.flatten()[:10])
            # print(kyle)
            new_trajectory[..., context_length:, -1] = new_trajectory_score.reshape(self.batch_size, horizon-context_length)
            new_trajectory_confidence_score = new_trajectory_confidence_score.reshape(self.batch_size, horizon-context_length)
            
            new_trajectory = new_trajectory.cpu().detach()
            new_trajectory_confidence_score = new_trajectory_confidence_score.cpu().detach()
            
            new_trajectories.append(new_trajectory)
            new_trajectories_confidence_score.append(new_trajectory_confidence_score)
        new_trajectories = torch.cat(new_trajectories, dim=0)
        # print(new_trajectories[:, context_length:, -1].flatten())
        new_trajectories_confidence_score = torch.cat(new_trajectories_confidence_score, dim=0)
        if confidence:
            new_trajectories = new_trajectories[torch.argsort(new_trajectories_confidence_score.sum(axis=-1))[:topk]]
        else:
            new_trajectories = new_trajectories[torch.argsort(new_trajectories[..., context_length:, -1].sum(axis=-1))[-topk:]]
        print(new_trajectories.shape)

        optima = 1.0
        num_trajectories = self.dataset.num_trajectories + new_trajectories.shape[0]
        # print(self.dataset.points[0, :4, :10])
        # print(self.dataset.normalizer.unnormalize(new_trajectories[..., :-1])[0, :4, :10])
        # print(self.dataset.values[0, -10:])
        # print(self.dataset.normalizer_values.unnormalize(new_trajectories[..., -1])[0, -10:])
        # print(kyle)
        points = torch.cat([self.dataset.points, self.dataset.normalizer.unnormalize(new_trajectories[..., :-1])], dim=0)
        values = torch.cat([self.dataset.values, self.dataset.normalizer_values.unnormalize(new_trajectories[..., -1])], dim=0)
        
        # print(self.dataset.points[0, :10], new_trajectories[..., :-1][0, :10], self.dataset.normalizer.unnormalize(new_trajectories[..., :-1][0, :10]))
        # print(self.dataset.values[0, :10], new_trajectories[..., -1][0, :10], self.dataset.normalizer_values.unnormalize(new_trajectories[..., -1][0, :10]))
        
        pointwise_regret = optima - values
        cumulative_regret_to_go = torch.flip(torch.cumsum(torch.flip(pointwise_regret, [1]), 1), [1])
        timesteps = torch.arange(horizon).repeat(num_trajectories, 1)
        
        self.dataset.num_trajectories = num_trajectories
        self.dataset.points = points
        self.dataset.values = values
        self.dataset.pointwise_regret = pointwise_regret
        self.dataset.cumulative_rtg = cumulative_regret_to_go
        self.dataset.timesteps = timesteps
        
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=0, shuffle=True, pin_memory=True, drop_last=True,
        ))
                
    def proxy_save(self):
        data = {
            'step': self.proxy_step,
            'model': self.proxy_model.state_dict(),
        }
        prefix = self.proxy_save_prefix if self.proxy_save_prefix is not None else logger.prefix
        savepath = os.path.join(prefix, 'proxy_checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.proxy_step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        logger.print(f'[ utils/training ] Saved model to {savepath}')

    def save(self):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(logger.prefix, 'checkpoint')
        os.makedirs(savepath, exist_ok=True)
        # logger.save_torch(data, savepath)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f'state_{self.step}.pt')
        else:
            savepath = os.path.join(savepath, 'state.pt')
        torch.save(data, savepath)
        logger.print(f'[ utils/training ] Saved model to {savepath}')
        
    def proxy_load(self, epoch=None, path=None):
        if path is not None:
            loadpath = path
        elif epoch is not None:
            loadpath = os.path.join(self.bucket, logger.prefix, f'proxy_checkpoint/state_{epoch}.pt')
        else:
            # 默认加载最新的state.pt文件
            loadpath = os.path.join(self.bucket, logger.prefix, 'proxy_checkpoint/state.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)
        self.proxy_model.load_state_dict(data['model'])

    def load_from_path(self, loadpath: str):
        """从任意路径加载扩散与 EMA（用于 few-shot 微调，基座 checkpoint 不在当前 RUN.prefix 下）。"""
        data = torch.load(loadpath, map_location=self.device)
        self.step = int(data.get('step', 0))
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state_{epoch}.pt')
        # data = logger.load_torch(loadpath)
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

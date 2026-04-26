import sys

if __name__ == "__main__":
    from diffuser.cpu_threads import maybe_apply_from_argv_and_env

    maybe_apply_from_argv_and_env()

import argparse
from ml_logger import logger, instr, needs_relaunch
from analysis import RUN
import jaynes
from scripts.train import main
from params_proto.neo_hyper import Sweep

if __name__ == '__main__':
    _cli_args = sys.argv[1:]
    sys.argv = [sys.argv[0]]

    try:
        import wandb

        wandb.login(key="cbb3c28f1b21becaa3b185b09464e7f6ba3b84ef")
    except Exception as e:
        print(f"[wandb] login 跳过: {e}", flush=True)
    parser = argparse.ArgumentParser()
    # 多任务训练参数
    parser.add_argument("--train_tasks", type=str, default="dkitty", help="训练数据集列表，用逗号分隔，例如: dkitty,ant,tfbind8")
    parser.add_argument(
        "--eval_task",
        type=str,
        default="",
        help="（单任务可忽略；多任务时仅写入 wandb 元数据，不影响 checkpoint 路径）",
    )
    parser.add_argument("--task", type=str, default="", help="兼容旧版API，将被废弃")
    
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--n_traj", type=int, default=1000)
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=32,
        help="VAE 隐空间维度；须与 construct_trajectories 一致；非 32 时 checkpoint 目录含 _latent{d}",
    )
    parser.add_argument(
        "--fixed_dim",
        type=int,
        default=128,
        help="VAE 输入统一维数（与 construct / train_vae 一致）；非 32 隐空间时 generated_datasets 路径含 _dim128_latent{d}",
    )

    # Task × text 联合 CFG 权重（可选；不指定则沿用 config 中默认值）
    parser.add_argument(
        "--condition_guidance_w_task",
        type=float,
        default=argparse.SUPPRESS,
        help="采样时 task 轴 classifier-free 强度；0 关闭",
    )
    parser.add_argument(
        "--condition_guidance_w_text",
        type=float,
        default=argparse.SUPPRESS,
        help="采样时 text 轴 classifier-free 强度；0 关闭",
    )
    parser.add_argument(
        "--use_text_condition",
        action="store_true",
        default=argparse.SUPPRESS,
        help="启用 task_metadata 文本条件（需锚点 config 含相关字段，如 ant+dkitty 用 ant_config）",
    )
    parser.add_argument(
        "--multitask_text_only",
        action="store_true",
        default=argparse.SUPPRESS,
        help="多任务混合数据上仅训练 text_mlp，不建 task 分支（batch 不含 task_idx）；需配合 --use_text_condition",
    )
    parser.add_argument(
        "--text_encoder_model",
        type=str,
        default=argparse.SUPPRESS,
        help="sentence-transformers 模型：Hub 名或本机目录（离线请传已下载的模型文件夹绝对路径）",
    )
    parser.add_argument(
        "--returns_condition",
        action="store_true",
        default=argparse.SUPPRESS,
        help="显式标量 return 条件（returns_mlp）；GTG 复现需同时 --include_returns",
    )
    parser.add_argument(
        "--include_returns",
        action="store_true",
        default=argparse.SUPPRESS,
        help="训练批包含 RewardBatch.returns，与 returns_condition 配对",
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=None,
        help="限制 CPU 线程数（OpenMP/BLAS/PyTorch）；等价于环境变量 CPU_THREADS（需在进程早期传入，已自动应用）",
    )
    parser.add_argument(
        "--traj_params_json",
        type=str,
        default=None,
        help="多任务：JSON 按任务覆盖 n_traj/k/eps（须与 construct_trajectories 一致）",
    )
    parser.add_argument(
        "--skip_auto_construct_trajectories",
        action="store_true",
        default=False,
        help="多任务：混合轨迹缺失时不自动运行 construct_trajectories（默认会自动生成）",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=None,
        help="扩散训练 epoch 数；n_train_steps = train_epochs * n_steps_per_epoch。",
    )
    parser.add_argument(
        "--real_task_text_only_finetune",
        action="store_true",
        default=False,
        help="真实任务迁移：单任务轨迹 + 仅文本条件（与 multitask_text_only 同架构），"
        "从全任务 multitask text 预训练权重微调（默认 mt 见 --pretrained_mt_hex）",
    )
    parser.add_argument(
        "--fewshot_text_only_finetune",
        action="store_true",
        default=False,
        help="已弃用：请用 --real_task_text_only_finetune",
    )
    parser.add_argument(
        "--pretrained_mt_hex",
        type=str,
        default=None,
        help="预训练 multitask text 的 hyper 段 16 位 hex（默认 911054c35daad7e0；或环境 GTG_REAL_TASK_PRETRAINED_MT_HEX）",
    )
    parser.add_argument(
        "--pretrained_multitask_train_tasks",
        type=str,
        default=None,
        help="预训练模型对应的全任务 CSV（默认 9 任务字典序，与 run_multitask.sh 一致）",
    )
    parser.add_argument(
        "--pretrained_diffusion_seed",
        type=int,
        default=0,
        help="预训练 checkpoint 所在 seed 目录（默认 0）",
    )
    parser.add_argument(
        "--load_diffusion_checkpoint",
        type=str,
        default=None,
        help="预训练扩散 checkpoint 的 .pt 路径（含 model/ema）；不填则按 --pretrained_mt_hex 自动解析",
    )
    parser.add_argument(
        "--load_diffusion_checkpoint_epoch",
        type=int,
        default=None,
        help="若未指定 --load_diffusion_checkpoint，可填 epoch 从当前 RUN.prefix 下加载（一般不用）",
    )
    parser.add_argument(
        "--proxy_filter",
        type=int,
        choices=[0, 1],
        default=argparse.SUPPRESS,
        help="1=训练 proxy（单任务训 proxy；多任务 ensure_multitask_proxies）；0=完全不训。"
        " 也可用环境变量 PROXY_FILTER=0/1（默认 1）",
    )
    parser.add_argument(
        "--run_suffix",
        type=str,
        default="",
        help="Optional suffix appended to RUN.prefix hyper dir (e.g. _ce0.2).",
    )
    parser.add_argument(
        "--train_timestep_bias_power",
        type=float,
        default=0.0,
        help=">0：训练时离散 t 采样偏向小 t（幂偏斜，越大越偏）；0=关闭（默认）。",
    )
    parser.add_argument(
        "--train_loss_min_snr_gamma",
        type=float,
        default=0.0,
        help=">0：对 epsilon 目标启用 min-SNR 逐样本损失加权（可试 5）；0=关闭（默认）。",
    )
    parser.add_argument(
        "--train_half_timestep_bias_frac",
        type=float,
        default=0.7,
        help="两阶段分界点（前段比例，0~1），默认 0.7。",
    )
    parser.add_argument(
        "--train_half_lr_mult",
        type=float,
        default=1.0,
        help="两阶段后段 LR 乘子（与 _halftbiasX 中的 X 对齐）；默认 1.0（关闭）。",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=argparse.SUPPRESS,
        help="扩散 Adam 学习率；不传则用锚点 config 的 learning_rate。传参后 RUN.prefix 含 _lr… 以免混用 checkpoint。",
    )

    args = parser.parse_args(_cli_args)
    if getattr(args, "learning_rate", None) is not None and float(args.learning_rate) <= 0.0:
        raise SystemExit("--learning_rate must be positive")

    # 兼容旧版API - 当指定task时，优先使用task参数覆盖train_tasks和eval_task
    if args.task:
        args.train_tasks = args.task
        args.eval_task = args.task
    
    # 处理任务列表，判断是单任务还是多任务
    from diffuser.utils.multitask_canon import (
        canonical_train_tasks_csv,
        diffusion_train_path_suffix_v2,
        learning_rate_path_suffix,
        multitask_path_token,
        multitask_text_only_path_infix,
        returns_cond_path_infix,
        text_cond_path_infix,
    )

    train_tasks_list = [t.strip() for t in args.train_tasks.split(",") if t.strip()]
    is_multitask = len(train_tasks_list) > 1
    if is_multitask:
        args.train_tasks = canonical_train_tasks_csv(args.train_tasks)
        train_tasks_list = [t.strip() for t in args.train_tasks.split(",") if t.strip()]
        if not args.eval_task:
            args.eval_task = train_tasks_list[0]

    real_task_ft = getattr(args, "real_task_text_only_finetune", False) or getattr(
        args, "fewshot_text_only_finetune", False
    )
    if args.train_epochs is None:
        args.train_epochs = 500
    if real_task_ft:
        args.multitask_text_only = True
        args.use_text_condition = True

    if (
        getattr(args, "multitask_text_only", False)
        and not is_multitask
        and not real_task_ft
    ):
        raise SystemExit(
            "错误: --multitask_text_only 仅适用于多任务（train_tasks 需为逗号分隔的多个任务），"
            "除非使用 --real_task_text_only_finetune（单任务 text-only 迁移）"
        )

    _ret = returns_cond_path_infix(args)
    _txt = text_cond_path_infix(args)
    _mto = multitask_text_only_path_infix(args)

    if real_task_ft and not is_multitask and not getattr(args, "load_diffusion_checkpoint", None):
        from diffuser.utils.real_task_transfer import (
            DEFAULT_PRETRAINED_MULTITASK_CSV,
            resolve_pretrained_diffusion_pt_for_real_task,
        )

        _csv = (
            getattr(args, "pretrained_multitask_train_tasks", None)
            or DEFAULT_PRETRAINED_MULTITASK_CSV
        )
        _csv = canonical_train_tasks_csv(_csv)
        _pt = resolve_pretrained_diffusion_pt_for_real_task(
            multitask_train_tasks_csv=_csv,
            frac=float(args.frac),
            sigma=float(args.sigma),
            mt_hex=getattr(args, "pretrained_mt_hex", None),
            pretrained_seed=int(getattr(args, "pretrained_diffusion_seed", 0)),
            config=None,
            latent_dim=int(getattr(args, "latent_dim", 32)),
        )
        if _pt:
            args.load_diffusion_checkpoint = _pt
            print(f"[real_task] 使用预训练扩散权重: {_pt}", flush=True)
        else:
            raise SystemExit(
                "未找到预训练 checkpoint：请设置 --load_diffusion_checkpoint，"
                "或确认 trained_models/multi_*_frac…/mt_<hex>_textcond_mttextonly/seed*/checkpoint/ 存在"
            )

    if real_task_ft and not is_multitask and args.n_traj == 1000 and args.k == 50:
        args.n_traj = 100
        args.k = 20

    # 根据任务数量设置数据路径
    if is_multitask:
        from diffuser.utils.traj_params import (
            multitask_checkpoint_hyper_dir,
            multitask_mixed_basename,
            prepare_multitask_traj,
        )

        train_tasks_str = multitask_path_token(args.train_tasks)
        n_d, k_d, e_d, sig = prepare_multitask_traj(
            train_tasks_list,
            args.n_traj,
            args.k,
            args.eps,
            args.horizon,
            args.traj_params_json,
        )
        _ld = int(args.latent_dim)
        from diffuser.utils.vae_layout import multitask_generated_candidate_rel_dirs

        _fd = int(getattr(args, "fixed_dim", 128))
        # dirname：latent≠32 时为 multi_*_dim{fd}_latent{ld}（与 train_vae / 用户现有目录一致）
        _rel_root = multitask_generated_candidate_rel_dirs(
            train_tasks_csv=args.train_tasks,
            frac=float(args.frac),
            sigma=float(args.sigma),
            fixed_dim=_fd,
            latent_dim=_ld,
        )[0]
        args.data_path = f"{_rel_root}/{multitask_mixed_basename(sig, _ld)}"
        args.multitask_traj_signature = sig
        args.traj_n_traj_dict = n_d
        args.traj_k_dict = k_d
        args.traj_eps_dict = e_d
        _hyper = multitask_checkpoint_hyper_dir(sig, _ret, _txt, _mto)
        if args.run_suffix:
            _hyper = f"{_hyper}{args.run_suffix}"
        _dtrain = diffusion_train_path_suffix_v2(
            float(getattr(args, "train_timestep_bias_power", 0.0)),
            float(getattr(args, "train_loss_min_snr_gamma", 0.0)),
            float(getattr(args, "train_half_timestep_bias_frac", 0.7)),
            float(getattr(args, "train_half_lr_mult", 1.0)),
        )
        if _dtrain:
            _hyper = f"{_hyper}{_dtrain}"
        _hyper = f"{_hyper}{learning_rate_path_suffix(vars(args).get('learning_rate'))}"
        if _ld != 32:
            _hyper = f"{_hyper}_latent{_ld}"
        RUN.prefix = f"trained_models/multi_{train_tasks_str}_frac{args.frac}_sigma{args.sigma}/{_hyper}/seed{args.seed}/"
    else:
        # 单任务模式，保持原有格式，与dfgo-main一致
        from diffuser.utils.vae_layout import per_task_latent_train_filename

        task_name = train_tasks_list[0]
        if not args.eval_task:
            args.eval_task = task_name
        _ld = int(args.latent_dim)
        args.data_path = (
            f"generated_datasets/{task_name}_frac{args.frac}_sigma{args.sigma}/"
            + per_task_latent_train_filename(
                task_name, args.n_traj, args.horizon, args.k, args.eps, _ld
            )
        )
        _few = "_fewshot_ft" if real_task_ft else ""
        _lat_tag = f"_latent{_ld}" if _ld != 32 else ""
        _dtrain = diffusion_train_path_suffix_v2(
            float(getattr(args, "train_timestep_bias_power", 0.0)),
            float(getattr(args, "train_loss_min_snr_gamma", 0.0)),
            float(getattr(args, "train_half_timestep_bias_frac", 0.7)),
            float(getattr(args, "train_half_lr_mult", 1.0)),
        )
        _lr_suf = learning_rate_path_suffix(vars(args).get("learning_rate"))
        RUN.prefix = f"trained_models/{task_name}_frac{args.frac}_sigma{args.sigma}/{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}{_few}{_ret}{_txt}{_mto}{_dtrain}{_lr_suf}{_lat_tag}/seed{args.seed}/"
    
    logger.print(RUN.prefix, color='green')
    jaynes.config("local")
    thunk = instr(main, **vars(args))
    jaynes.run(thunk)
    

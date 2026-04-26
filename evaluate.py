import sys

if __name__ == "__main__":
    import faulthandler

    faulthandler.enable(all_threads=True)
    from diffuser.cpu_threads import maybe_apply_from_argv_and_env

    maybe_apply_from_argv_and_env()

import argparse
import os
from ml_logger import logger, instr, needs_relaunch
from analysis import RUN
import jaynes
from scripts.evaluate import evaluate
from params_proto.neo_hyper import Sweep


if __name__ == '__main__':
    _cli_args = sys.argv[1:]
    sys.argv = [sys.argv[0]]

    parser = argparse.ArgumentParser()
    
    # 多任务评估支持
    parser.add_argument("--train_tasks", type=str, default="dkitty", help="训练数据集列表，用逗号分隔")
    parser.add_argument("--eval_task", type=str, default="", help="仅评单任务时指定；多任务默认评全部 train_tasks 时可不填（此时若只评一个任务需配合 --eval_only_first）")
    parser.add_argument("--eval_all_tasks", action="store_true", help="多任务时依次评估 train_tasks 中每个任务（多任务时默认开启，可用 --eval_only_first 关闭）")
    parser.add_argument("--eval_only_first", action="store_true", help="多任务时只评估 --eval_task 指定的单个任务，不跑全部训练任务")
    parser.add_argument("--checkpoint_eval_task", type=str, default="", help="已弃用：checkpoint 路径不再含 eval 后缀，忽略即可")
    parser.add_argument("--task", type=str, default="", help="兼容旧版API，将被废弃")
    
    # 其他评估参数
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--ctx_len", type=int, default=32, help="条件上下文长度，需与训练一致")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    
    # 数据生成参数
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--n_traj", type=int, default=1000)
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=32,
        help="VAE 隐空间维度；须与 construct / train 一致；非 32 时 RUN.prefix 含 _latent{d}",
    )

    parser.add_argument(
        "--condition_guidance_w_task",
        type=float,
        default=argparse.SUPPRESS,
        help="评估采样时 task 轴 CFG 权重；不指定则用 config 默认值",
    )
    parser.add_argument(
        "--condition_guidance_w_text",
        type=float,
        default=argparse.SUPPRESS,
        help="评估采样时 text 轴 CFG 权重；不指定则用 config 默认值",
    )
    parser.add_argument(
        "--use_text_condition",
        action="store_true",
        default=argparse.SUPPRESS,
        help="评估时加载 task_metadata 文本嵌入（须与训练一致）",
    )
    parser.add_argument(
        "--multitask_text_only",
        action="store_true",
        default=argparse.SUPPRESS,
        help="与训练一致：多任务仅文本分支（无 task_idx）",
    )
    parser.add_argument(
        "--returns_condition",
        action="store_true",
        default=argparse.SUPPRESS,
        help="与训练一致：采样时 returns 进入 returns_mlp",
    )
    parser.add_argument(
        "--include_returns",
        action="store_true",
        default=argparse.SUPPRESS,
        help="与训练一致：数据集构造含 returns",
    )
    parser.add_argument(
        "--text_encoder_model",
        type=str,
        default=argparse.SUPPRESS,
        help="sentence-transformers 模型：Hub 名或本机目录（离线请传已下载的模型文件夹绝对路径）",
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=None,
        help="限制 CPU 线程数（OpenMP/BLAS/PyTorch）；等价于环境变量 CPU_THREADS",
    )
    parser.add_argument(
        "--traj_params_json",
        type=str,
        default=None,
        help="多任务：JSON 按任务覆盖 n_traj/k/eps（须与 construct / train 一致）",
    )
    parser.add_argument(
        "--skip_auto_construct_trajectories",
        action="store_true",
        default=False,
        help="多任务：混合轨迹缺失时不自动运行 construct_trajectories（默认会自动生成）",
    )
    parser.add_argument(
        "--real_task_zero_shot_eval",
        action="store_true",
        default=False,
        help="单任务：不依赖本目录下已训模型，从全任务 multitask text 预训练 checkpoint 直接评估（text-only）",
    )
    parser.add_argument(
        "--real_task_text_only_finetune",
        action="store_true",
        default=False,
        help="与 train 一致：单任务 text-only 微调后的评估，RUN.prefix 含 _fewshot_ft",
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
        help="与 train 一致：预训练 mt_<hex>_textcond_mttextonly（默认 911054c35daad7e0）",
    )
    parser.add_argument(
        "--pretrained_multitask_train_tasks",
        type=str,
        default=None,
        help="预训练模型对应的全任务 CSV（默认 9 任务）",
    )
    parser.add_argument(
        "--pretrained_diffusion_seed",
        type=int,
        default=0,
        help="预训练 checkpoint 的 seed 目录（默认 0）",
    )
    parser.add_argument(
        "--load_diffusion_checkpoint",
        type=str,
        default=None,
        help="显式指定扩散 state.pt；不填则与 --real_task_zero_shot_eval 联用自动解析",
    )
    parser.add_argument(
        "--proxy_filter",
        type=int,
        choices=[0, 1],
        default=argparse.SUPPRESS,
        help="1=训练/加载 proxy 并用其筛选 queries；0=仅扩散采样后 eval。"
        " Zero-shot（--real_task_zero_shot_eval）在代码中恒关闭 proxy；"
        " few-shot 默认 1，可用 0 或环境变量 PROXY_FILTER=0",
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
        help="须与训练一致：非零时 RUN.prefix 含 _tsbias…（与 train.py 对齐）。",
    )
    parser.add_argument(
        "--train_loss_min_snr_gamma",
        type=float,
        default=0.0,
        help="须与训练一致：非零时 RUN.prefix 含 _msnr…（与 train.py 对齐）。",
    )
    parser.add_argument(
        "--train_half_timestep_bias_frac",
        type=float,
        default=0.7,
        help="须与训练一致：两阶段分界点（前段比例），默认 0.7。",
    )
    parser.add_argument(
        "--train_half_lr_mult",
        type=float,
        default=1.0,
        help="须与训练一致：两阶段后段 LR 乘子；默认 1.0（关闭）。",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=argparse.SUPPRESS,
        help="须与训练一致：显式传参时 RUN.prefix 含 _lr…（与 train.py 对齐）。",
    )
    parser.add_argument(
        "--latent_observation_dim",
        type=int,
        default=None,
        help="真实任务 zero-shot：可选覆盖扩散观测维（通常即 VAE latent；默认读 vae_info 或 32）",
    )
    parser.add_argument(
        "--sample_viz_wandb",
        action="store_true",
        default=False,
        help="在扩散步上按 stride 用 Oracle 评 context 之后轨迹的 y，写入 wandb；可与 --sample_viz_dump_jsonl 联用",
    )
    parser.add_argument(
        "--sample_viz_dump_jsonl",
        type=str,
        default=None,
        help="将每步 Oracle 指标追加写入该目录下 <tag>_seed<seed>.jsonl（可不依赖 wandb；用于 visualize 多 seed 聚合）",
    )
    parser.add_argument(
        "--sample_viz_stride",
        type=int,
        default=10,
        help="每多少步反演（0→T-1 的序号）记一次 Oracle；仍保证 t=0 会记一次",
    )
    parser.add_argument(
        "--sample_viz_tag",
        type=str,
        default="viz",
        help="wandb 键前缀 sample_viz/<tag>/...，用于多实验同图对比（如 mt_text、st_duo）",
    )
    parser.add_argument(
        "--sample_viz_max_queries",
        type=int,
        default=512,
        help="每步 Oracle 最多评多少个点（对尾段展平后子采样，控成本）",
    )

    args = parser.parse_args(_cli_args)

    # 兼容旧版API - 当指定task时，优先使用task参数覆盖train_tasks和eval_task
    if args.task:
        args.train_tasks = args.task
        args.eval_task = args.task
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
    if len(train_tasks_list) > 1:
        args.train_tasks = canonical_train_tasks_csv(args.train_tasks)
        train_tasks_list = [t.strip() for t in args.train_tasks.split(",") if t.strip()]
    if not args.eval_task:
        args.eval_task = train_tasks_list[0]

    zshot = getattr(args, "real_task_zero_shot_eval", False)
    real_task_ft = getattr(args, "real_task_text_only_finetune", False) or getattr(
        args, "fewshot_text_only_finetune", False
    )
    if zshot:
        args.multitask_text_only = True
        args.use_text_condition = True
    elif real_task_ft:
        args.multitask_text_only = True
        args.use_text_condition = True
    train_tasks_str = (
        multitask_path_token(args.train_tasks)
        if len(train_tasks_list) > 1
        else train_tasks_list[0]
    )
    # 多任务时默认评估全部训练任务（与 run_multitask 预期一致）；仅评一个任务时用 --eval_only_first
    if len(train_tasks_list) > 1:
        if args.eval_only_first:
            args.eval_all_tasks = False
        elif not args.eval_all_tasks:
            args.eval_all_tasks = True

    if (
        getattr(args, "multitask_text_only", False)
        and len(train_tasks_list) <= 1
        and not zshot
        and not real_task_ft
    ):
        raise SystemExit(
            "错误: --multitask_text_only 仅适用于多任务（train_tasks 需为逗号分隔的多个任务），"
            "除非使用 --real_task_zero_shot_eval 或 --real_task_text_only_finetune"
        )

    _ret = returns_cond_path_infix(args)
    _txt = text_cond_path_infix(args)
    _mto = multitask_text_only_path_infix(args)

    args.diffusion_checkpoint_dir = None
    if zshot and len(train_tasks_list) == 1:
        from diffuser.utils.real_task_transfer import (
            DEFAULT_PRETRAINED_MT_HEX,
            DEFAULT_PRETRAINED_MULTITASK_CSV,
            resolve_multitask_pretrained_run_dir,
            resolve_diffusion_state_pt,
        )

        _csv = getattr(args, "pretrained_multitask_train_tasks", None) or DEFAULT_PRETRAINED_MULTITASK_CSV
        _csv = canonical_train_tasks_csv(_csv)
        if getattr(args, "load_diffusion_checkpoint", None):
            args.diffusion_checkpoint_dir = os.path.dirname(
                os.path.abspath(args.load_diffusion_checkpoint)
            )
        else:
            _mh = getattr(args, "pretrained_mt_hex", None) or DEFAULT_PRETRAINED_MT_HEX
            run_dir = resolve_multitask_pretrained_run_dir(
                multitask_train_tasks_csv=_csv,
                frac=float(args.frac),
                sigma=float(args.sigma),
                mt_hex=_mh,
                seed=int(getattr(args, "pretrained_diffusion_seed", 0)),
                latent_dim=int(getattr(args, "latent_dim", 32)),
            )
            args.diffusion_checkpoint_dir = os.path.join(run_dir, "checkpoint")
            _pt = resolve_diffusion_state_pt(args.diffusion_checkpoint_dir, None)
            if _pt:
                args.load_diffusion_checkpoint = _pt
            else:
                raise SystemExit(
                    f"未找到预训练 checkpoint：{args.diffusion_checkpoint_dir} 下无 state*.pt"
                )

    # 多任务模式下的数据路径和运行前缀（与 train.py 一致；与评 ant/dkitty 无关）
    if len(train_tasks_list) > 1:
        from diffuser.utils.traj_params import (
            multitask_checkpoint_hyper_dir,
            multitask_mixed_basename,
            prepare_multitask_traj,
        )

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
        # 单任务模式，保持原有逻辑
        from diffuser.utils.vae_layout import per_task_latent_train_filename

        task_name = train_tasks_list[0]
        if zshot and args.n_traj == 1000 and args.k == 50:
            args.n_traj = 100
            args.k = 20
        if zshot:
            _zsuf = "_realtask_zs"
        elif real_task_ft:
            _zsuf = "_fewshot_ft"
        else:
            _zsuf = ""
        _ld = int(args.latent_dim)
        args.data_path = (
            f"generated_datasets/{args.train_tasks}_frac{args.frac}_sigma{args.sigma}/"
            + per_task_latent_train_filename(
                task_name, args.n_traj, args.horizon, args.k, args.eps, _ld
            )
        )
        _lat_tag = f"_latent{_ld}" if _ld != 32 else ""
        _dtrain = diffusion_train_path_suffix_v2(
            float(getattr(args, "train_timestep_bias_power", 0.0)),
            float(getattr(args, "train_loss_min_snr_gamma", 0.0)),
            float(getattr(args, "train_half_timestep_bias_frac", 0.7)),
            float(getattr(args, "train_half_lr_mult", 1.0)),
        )
        _lr_suf = learning_rate_path_suffix(vars(args).get("learning_rate"))
        RUN.prefix = (
            f"trained_models/{args.train_tasks}_frac{args.frac}_sigma{args.sigma}/"
            f"{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}{_zsuf}{_ret}{_txt}{_mto}{_dtrain}{_lr_suf}{_lat_tag}/seed{args.seed}/"
        )
    
    logger.print(RUN.prefix, color='green')
    jaynes.config("local")
    thunk = instr(evaluate, **vars(args))
    jaynes.run(thunk)

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
        default=200,
        help="扩散训练 epoch 数；n_train_steps = train_epochs * n_steps_per_epoch（默认 200）",
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

    args = parser.parse_args(_cli_args)

    # 兼容旧版API - 当指定task时，优先使用task参数覆盖train_tasks和eval_task
    if args.task:
        args.train_tasks = args.task
        args.eval_task = args.task
    
    # 处理任务列表，判断是单任务还是多任务
    from diffuser.utils.multitask_canon import (
        canonical_train_tasks_csv,
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
        # dirname 指向 multi_*；短文件名 mixed_mt_<hash>.p（完整 sig 见 multitask_slug_manifest.json）
        args.data_path = f"generated_datasets/multi_{train_tasks_str}_frac{args.frac}_sigma{args.sigma}/{multitask_mixed_basename(sig)}"
        args.multitask_traj_signature = sig
        args.traj_n_traj_dict = n_d
        args.traj_k_dict = k_d
        args.traj_eps_dict = e_d
        _hyper = multitask_checkpoint_hyper_dir(sig, _ret, _txt, _mto)
        RUN.prefix = f"trained_models/multi_{train_tasks_str}_frac{args.frac}_sigma{args.sigma}/{_hyper}/seed{args.seed}/"
    else:
        # 单任务模式，保持原有格式，与dfgo-main一致
        task_name = train_tasks_list[0]
        if not args.eval_task:
            args.eval_task = task_name
        args.data_path = f'generated_datasets/{task_name}_frac{args.frac}_sigma{args.sigma}/{task_name}_{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}_vae_latent32_train.p'
        _few = "_fewshot_ft" if real_task_ft else ""
        RUN.prefix = f"trained_models/{task_name}_frac{args.frac}_sigma{args.sigma}/{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}{_few}{_ret}{_txt}{_mto}/seed{args.seed}/"
    
    logger.print(RUN.prefix, color='green')
    jaynes.config("local")
    thunk = instr(main, **vars(args))
    jaynes.run(thunk)
    

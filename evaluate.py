import sys

if __name__ == "__main__":
    from diffuser.cpu_threads import maybe_apply_from_argv_and_env

    maybe_apply_from_argv_and_env()

import argparse
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

    args = parser.parse_args(_cli_args)

    # 兼容旧版API - 当指定task时，优先使用task参数覆盖train_tasks和eval_task
    if args.task:
        args.train_tasks = args.task
        args.eval_task = args.task
    from diffuser.utils.multitask_canon import (
        canonical_train_tasks_csv,
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

    if getattr(args, "multitask_text_only", False) and len(train_tasks_list) <= 1:
        raise SystemExit(
            "错误: --multitask_text_only 仅适用于多任务（train_tasks 需为逗号分隔的多个任务）"
        )

    _ret = returns_cond_path_infix(args)
    _txt = text_cond_path_infix(args)
    _mto = multitask_text_only_path_infix(args)
    # 多任务模式下的数据路径和运行前缀（与 train.py 一致；与评 ant/dkitty 无关）
    if len(train_tasks_list) > 1:
        from diffuser.utils.traj_params import prepare_multitask_traj

        n_d, k_d, e_d, sig = prepare_multitask_traj(
            train_tasks_list,
            args.n_traj,
            args.k,
            args.eps,
            args.horizon,
            args.traj_params_json,
        )
        args.data_path = f"generated_datasets/multi_{train_tasks_str}_frac{args.frac}_sigma{args.sigma}/{sig}_vae_latent32_train.p"
        args.multitask_traj_signature = sig
        args.traj_n_traj_dict = n_d
        args.traj_k_dict = k_d
        args.traj_eps_dict = e_d
        RUN.prefix = f"trained_models/multi_{train_tasks_str}_frac{args.frac}_sigma{args.sigma}/{sig}{_ret}{_txt}{_mto}/seed{args.seed}/"
    else:
        # 单任务模式，保持原有逻辑
        task_name = train_tasks_list[0]
        args.data_path = f'generated_datasets/{args.train_tasks}_frac{args.frac}_sigma{args.sigma}/{task_name}_{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}_vae_latent32_train.p'
        RUN.prefix = f"trained_models/{args.train_tasks}_frac{args.frac}_sigma{args.sigma}/{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}{_ret}{_txt}{_mto}/seed{args.seed}/"
    
    logger.print(RUN.prefix, color='green')
    jaynes.config("local")
    thunk = instr(evaluate, **vars(args))
    jaynes.run(thunk)

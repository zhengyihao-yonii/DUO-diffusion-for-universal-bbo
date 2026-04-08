import argparse
import sys
from ml_logger import logger, instr, needs_relaunch
from analysis import RUN
import jaynes
from scripts.train import main
from params_proto.neo_hyper import Sweep
import wandb
    
if __name__ == '__main__':
    wandb.login(key="cbb3c28f1b21becaa3b185b09464e7f6ba3b84ef")
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

    args = parser.parse_args()
    # jaynes + ParamsProto 会再解析 sys.argv；Config 中 bool 与 argparse store_true 冲突，kwargs 已由 instr 传入
    sys.argv = [sys.argv[0]]

    # 兼容旧版API - 当指定task时，优先使用task参数覆盖train_tasks和eval_task
    if args.task:
        args.train_tasks = args.task
        args.eval_task = args.task
    
    # 处理任务列表，判断是单任务还是多任务
    from diffuser.utils.multitask_canon import canonical_train_tasks_csv, multitask_path_token, returns_cond_path_infix

    train_tasks_list = [t.strip() for t in args.train_tasks.split(",") if t.strip()]
    is_multitask = len(train_tasks_list) > 1
    if is_multitask:
        args.train_tasks = canonical_train_tasks_csv(args.train_tasks)
        train_tasks_list = [t.strip() for t in args.train_tasks.split(",") if t.strip()]
        if not args.eval_task:
            args.eval_task = train_tasks_list[0]

    _ret = returns_cond_path_infix(args)
    # 根据任务数量设置数据路径
    if is_multitask:
        # 多任务模式（路径与任务名字典序一致，与 ant,dkitty 与 dkitty,ant 共用同一实验目录）
        train_tasks_str = multitask_path_token(args.train_tasks)
        args.data_path = f'generated_datasets/multi_{train_tasks_str}_frac{args.frac}_sigma{args.sigma}/{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}_vae_latent32_train.p'
        # 与 eval 目标无关：同一套 multitask 权重只存一份
        RUN.prefix = f"trained_models/multi_{train_tasks_str}_frac{args.frac}_sigma{args.sigma}/{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}{_ret}/seed{args.seed}/"
    else:
        # 单任务模式，保持原有格式，与dfgo-main一致
        task_name = train_tasks_list[0]
        if not args.eval_task:
            args.eval_task = task_name
        args.data_path = f'generated_datasets/{task_name}_frac{args.frac}_sigma{args.sigma}/{task_name}_{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}_vae_latent32_train.p'
        RUN.prefix = f"trained_models/{task_name}_frac{args.frac}_sigma{args.sigma}/{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}{_ret}/seed{args.seed}/"
    
    logger.print(RUN.prefix, color='green')
    jaynes.config("local")
    thunk = instr(main, **vars(args))
    jaynes.run(thunk)
    

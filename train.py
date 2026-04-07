import argparse
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

    args = parser.parse_args()
    
    # 兼容旧版API - 当指定task时，优先使用task参数覆盖train_tasks和eval_task
    if args.task:
        args.train_tasks = args.task
        args.eval_task = args.task
    
    # 处理任务列表，判断是单任务还是多任务
    from diffuser.utils.multitask_canon import canonical_train_tasks_csv, multitask_path_token

    train_tasks_list = [t.strip() for t in args.train_tasks.split(",") if t.strip()]
    is_multitask = len(train_tasks_list) > 1
    if is_multitask:
        args.train_tasks = canonical_train_tasks_csv(args.train_tasks)
        train_tasks_list = [t.strip() for t in args.train_tasks.split(",") if t.strip()]
        if not args.eval_task:
            args.eval_task = train_tasks_list[0]

    # 根据任务数量设置数据路径
    if is_multitask:
        # 多任务模式（路径与任务名字典序一致，与 ant,dkitty 与 dkitty,ant 共用同一实验目录）
        train_tasks_str = multitask_path_token(args.train_tasks)
        args.data_path = f'generated_datasets/multi_{train_tasks_str}_frac{args.frac}_sigma{args.sigma}/{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}_vae_latent32_train.p'
        # 与 eval 目标无关：同一套 multitask 权重只存一份
        RUN.prefix = f"trained_models/multi_{train_tasks_str}_frac{args.frac}_sigma{args.sigma}/{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}/seed{args.seed}/"
    else:
        # 单任务模式，保持原有格式，与dfgo-main一致
        task_name = train_tasks_list[0]
        if not args.eval_task:
            args.eval_task = task_name
        args.data_path = f'generated_datasets/{task_name}_frac{args.frac}_sigma{args.sigma}/{task_name}_{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}_vae_latent32_train.p'
        RUN.prefix = f"trained_models/{task_name}_frac{args.frac}_sigma{args.sigma}/{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}/seed{args.seed}/"
    
    logger.print(RUN.prefix, color='green')
    jaynes.config("local")
    thunk = instr(main, **vars(args))
    jaynes.run(thunk)
    

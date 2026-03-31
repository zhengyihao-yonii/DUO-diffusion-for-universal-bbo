import argparse
from ml_logger import logger, instr, needs_relaunch
from analysis import RUN
import jaynes
from scripts.evaluate import evaluate
from params_proto.neo_hyper import Sweep


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 多任务评估支持
    parser.add_argument("--train_tasks", type=str, default="dkitty", help="训练数据集列表，用逗号分隔")
    parser.add_argument("--eval_task", type=str, default="", help="评估任务，如果不指定则使用train_tasks的第一个任务")
    parser.add_argument("--task", type=str, default="", help="兼容旧版API，将被废弃")
    
    # 其他评估参数
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--ctx_len", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    
    # 数据生成参数
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
    if not args.eval_task:
        args.eval_task = args.train_tasks.split(',')[0]
    
    # 解析训练任务列表
    train_tasks_list = args.train_tasks.split(',')
    train_tasks_str = '_'.join(train_tasks_list)
    
    # 多任务模式下的数据路径和运行前缀
    if len(train_tasks_list) > 1:
        # 多任务模式下，确保路径与训练时保持一致，包含_eval{eval_task}后缀
        args.data_path = f'generated_datasets/multi_{train_tasks_str}_frac{args.frac}_sigma{args.sigma}/{sum([args.n_traj] * len(train_tasks_list))}x{args.horizon}_k{args.k}_eps{args.eps}_vae_latent32_train.p'
        RUN.prefix = f"trained_models/multi_{train_tasks_str}_frac{args.frac}_sigma{args.sigma}_eval{args.eval_task}/{sum([args.n_traj] * len(train_tasks_list))}x{args.horizon}_k{args.k}_eps{args.eps}/seed{args.seed}/"
    else:
        # 单任务模式，保持原有逻辑
        task_name = train_tasks_list[0]
        args.data_path = f'generated_datasets/{args.train_tasks}_frac{args.frac}_sigma{args.sigma}/{task_name}_{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}_vae_latent32_train.p'
        RUN.prefix = f"trained_models/{args.train_tasks}_frac{args.frac}_sigma{args.sigma}/{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}/seed{args.seed}/"
    
    logger.print(RUN.prefix, color='green')
    jaynes.config("local")
    thunk = instr(evaluate, **vars(args))
    jaynes.run(thunk)

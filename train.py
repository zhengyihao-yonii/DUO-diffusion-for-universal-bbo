import argparse
from ml_logger import logger, instr, needs_relaunch
from analysis import RUN
import jaynes
from scripts.train import main
from params_proto.neo_hyper import Sweep
import wandb 
    
if __name__ == '__main__':
    wandb.login(key="9f59486bed008c431a4a5804c35bb3c065d0b658")
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="dkitty")
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--n_traj", type=int, default=1000)

    args = parser.parse_args()
    args.data_path = f'generated_datasets/{args.task}_frac{args.frac}_sigma{args.sigma}/{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}_vae_latent32_train.p'
    RUN.prefix = f"trained_models/{args.task}_frac{args.frac}_sigma{args.sigma}/{args.n_traj}x{args.horizon}_k{args.k}_eps{args.eps}/seed{args.seed}/"
    
    logger.print(RUN.prefix, color='green')
    jaynes.config("local")
    thunk = instr(main, **vars(args))
    jaynes.run(thunk)
    

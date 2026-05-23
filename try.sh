#!/usr/bin/env bash
set -euo pipefail
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/wandb_login.sh
source "${_SCRIPT_DIR}/scripts/wandb_login.sh"

TRAIN_TIMESTEP_BIAS_POWER="${TRAIN_TIMESTEP_BIAS_POWER:-0.0}"
TRAIN_LOSS_MIN_SNR_GAMMA="${TRAIN_LOSS_MIN_SNR_GAMMA:-0.0}"

/home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/DUO/train_vae.py --task $1
/home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/DUO/construct_trajectories.py --task $1 --n_traj $2
/home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/DUO/train.py --task $1 --n_traj $2 --eps $4 --k $3 --train_timestep_bias_power "$TRAIN_TIMESTEP_BIAS_POWER" --train_loss_min_snr_gamma "$TRAIN_LOSS_MIN_SNR_GAMMA"
/home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/DUO/evaluate.py --task $1 --n_traj $2 --eps $4 --k $3

# /home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/DUO/evaluate.py --task dkitty --n_traj 4000 --eps 0.01 --k 20
# /home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/DUO/train.py --task tfbind8 --n_traj 1000 --eps 0.05 --k 50
# bash try.sh dkitty 4000 20 0.01
# /home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/universal-offline-bbo-main/src/train_UniSO-T.py experiment=UniSO-T-Improved ++trainer.max_epochs=200

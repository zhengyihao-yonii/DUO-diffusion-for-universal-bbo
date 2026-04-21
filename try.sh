
/home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/DUO/train_vae.py --task $1
/home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/DUO/construct_trajectories.py --task $1 --n_traj $2
/home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/DUO/train.py --task $1 --n_traj $2 --eps $4 --k $3
/home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/DUO/evaluate.py --task $1 --n_traj $2 --eps $4 --k $3

# /home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/DUO/evaluate.py --task dkitty --n_traj 4000 --eps 0.01 --k 20
# /home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/DUO/train.py --task tfbind8 --n_traj 1000 --eps 0.05 --k 50
# bash try.sh dkitty 4000 20 0.01
# /home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/universal-offline-bbo-main/src/train_UniSO-T.py experiment=UniSO-T-Improved ++trainer.max_epochs=200

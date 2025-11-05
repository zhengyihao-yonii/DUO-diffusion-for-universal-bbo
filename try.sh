
/home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/GTGdfgo/train_vae.py --task $1
/home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/GTGdfgo/construct_trajectory.py --task $1
/home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/GTGdfgo/train.py --task $1 --n_traj 1000 --eps 0.05 --k 50
/home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/GTGdfgo/evaluate.py --task $1 --n_traj 1000 --eps 0.05 --k 50

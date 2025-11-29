
/home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/GTGdfgo/train_vae.py --task $1
/home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/GTGdfgo/construct_trajectories.py --task $1
/home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/GTGdfgo/train.py --task $1 --n_traj $2 --eps $4 --k $3
/home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/GTGdfgo/evaluate.py --task $1 --n_traj $2 --eps $4 --k $3

# /home/xk/anaconda3/envs/gtg/bin/python /data/xk/zyh_dfgo/GTGdfgo/evaluate.py --task dkitty --n_traj 4000 --eps 0.01 --k 20
# bash try.sh dkitty 4000 20 0.01


MAX_JOBS=3
AVAILABLE_GPUS="1 2 3"
MAX_RETRIES=0

get_gpu_allocation() {
    local job_number=$1
    local gpus=($AVAILABLE_GPUS)
    local num_gpus=${#gpus[@]}
    local gpu_id=$((job_number % num_gpus))
    echo ${gpus[gpu_id]}
}

check_jobs() {
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

run_with_retry() {
    local script=$1
    local gpu_allocation=$2
    local attempt=0
    echo $gpu_allocation
    while [ $attempt -le $MAX_RETRIES ]; do
        # Run the Python script
        CUDA_VISIBLE_DEVICES=$gpu_allocation python $script
        status=$?
        if [ $status -eq 0 ]; then
            echo "Script $script succeeded."
            break
        else
            echo "Script $script failed on attempt $attempt. Retrying..."
            ((attempt++))
        fi
    done
    if [ $attempt -gt $MAX_RETRIES ]; then
        echo "Script $script failed after $MAX_RETRIES attempts."
    fi
}

# for task in "ant" "superconductor"; do 
#     for seed in {1..8}; do 

#         check_jobs
#         gpu_allocation=$(get_gpu_allocation $job_number)
#         ((job_number++))
#         run_with_retry "train.py \
#             --task $task \
#             --horizon 64 \
#             --k 20 \
#             --eps 0.05 \
#             --n_traj 4000 \
#             --seed $seed" \
#             "$gpu_allocation" & 
#     done 
# done 

# wait


for task in "dkitty"; do 
    for seed in {1..8}; do 

        check_jobs
        gpu_allocation=$(get_gpu_allocation $job_number)
        ((job_number++))
        run_with_retry "train.py \
            --task $task \
            --horizon 64 \
            --k 20 \
            --eps 0.01 \
            --n_traj 4000 \
            --seed $seed" \
            "$gpu_allocation" & 
    done 
done 

wait

for task in "superconductor"; do 
    for seed in {1..8}; do 

        check_jobs
        gpu_allocation=$(get_gpu_allocation $job_number)
        ((job_number++))
        run_with_retry "train.py \
            --task $task \
            --horizon 64 \
            --k 20 \
            --eps 0.05 \
            --n_traj 4000 \
            --seed $seed" \
            "$gpu_allocation" & 
    done 
done 

wait


# for task in "tfbind8" "tfbind10"; do 
#     for seed in {1..8}; do 

#         check_jobs
#         gpu_allocation=$(get_gpu_allocation $job_number)
#         ((job_number++))
#         run_with_retry "train.py \
#             --task $task \
#             --horizon 64 \
#             --k 50 \
#             --eps 0.05 \
#             --n_traj 1000 \
#             --seed $seed" \
#             "$gpu_allocation" & 
#     done 
# done 

# wait
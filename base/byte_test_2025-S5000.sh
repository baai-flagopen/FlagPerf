#!/bin/bash
current_dir=$(pwd)
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
log_file="${current_dir}/byte_${timestamp}.log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$log_file"
}

log "当前目录为：$current_dir"
log "日志文件：$log_file"

dirs=(
    "toolkits/computation-BF16"
    "toolkits/computation-FP16"
    "toolkits/computation-FP32"
    "toolkits/computation-FP64"
    "toolkits/computation-TF32"
    "toolkits/computation-FP8"
    "toolkits/computation-INT8"
    "toolkits/main_memory-bandwidth"
    "toolkits/main_memory-capacity"
)

for device in {0..7}; do
    export MUSA_VISIBLE_DEVICES=$device
    echo " "
    log "========== GPU INDEX=$device =========="
    echo " "
    
    for dir in "${dirs[@]}"; do
        target_dir="${current_dir}/${dir}"
        log "进入目录：$target_dir"
        cd "$target_dir/mthreads/S5000" || { log "目录不存在：$target_dir"; exit 1; }
        
        bash main.sh 2>&1 | tee -a "$log_file"
    done
done

cd "${current_dir}/toolkits/interconnect-MPI_intraserver/mthreads/S5000/" || exit
bash main.sh 2>&1 | tee -a "$log_file"

cd "${current_dir}/toolkits/interconnect-P2P_intraserver/mthreads/S5000/" || exit
bash main.sh 2>&1 | tee -a "$log_file"

log "所有任务完成！日志路径：$log_file"

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

OPENAIKEY=""
OPENAIBASE=""

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python /mnt_new/hanyudong.hyd/Slow-Fast-Vid/llamavid/eval/model_videomme_general.py \
        --model-path /mnt_new/hanyudong.hyd/Slow-Fast-Vid/work_dirs/llama-vid-7b-full-224-video-fps-1-old_best \
        --video_dir /mnt_new/zirui.lgp/datasets/benchmark/VideoMME/data \
        --gt_file /mnt_new/zirui.lgp/datasets/benchmark/VideoMME/videomme_generic_qa.json \
        --output_dir /mnt_new/hanyudong.hyd/Slow-Fast-Vid/work_dirs/eval_videomme_benchmark \
        --output_name pred \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode vicuna_v1 &

done

wait

#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=""
OPENAIKEY=""
OPENAIBASE=""

ln -s /mnt_new/hanyudong.hyd/llama-vid/model_zoo/bert-base-uncased /root/.cache/huggingface/hub/bert-base-uncased

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python /mnt_new/hanyudong.hyd/Slow-Fast-Vid/llamavid/eval/model_msrvtt_qa.py \
    --model-path /mnt_new/hanyudong.hyd/Slow-Fast-Vid/work_dirs/llama-vid-7b-full-224-video-fps-1 \
    --video_dir /mnt_new/public/multimodal_retrieval/datasets/msrvtt/compressed_videos \
    --gt_file_question /mnt_new/fengzipeng.fzp/datasets/video/benchmark_QA/MSRVTT_Zero_Shot_QA/test_q.json \
    --gt_file_answers /mnt_new/fengzipeng.fzp/datasets/video/benchmark_QA/MSRVTT_Zero_Shot_QA/test_a.json \
    --output_dir /mnt_new/hanyudong.hyd/Slow-Fast-Vid/work_dirs/eval_msrvtt \
    --output_name pred \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --conv-mode vicuna_v1 &

done

wait

python /mnt_new/hanyudong.hyd/Slow-Fast-Vid/llamavid/eval/eval_msrvtt_qa.py \
    --pred_path /mnt_new/hanyudong.hyd/Slow-Fast-Vid/work_dirs/eval_msrvtt \
    --output_dir /mnt_new/hanyudong.hyd/Slow-Fast-Vid/work_dirs/eval_msrvtt/results \
    --output_json /mnt_new/hanyudong.hyd/Slow-Fast-Vid/work_dirs/eval_msrvtt/results.json \
    --num_chunks $CHUNKS \
    --num_tasks 16 \
    # --api_key $OPENAIKEY \
    # --api_base $OPENAIBASE




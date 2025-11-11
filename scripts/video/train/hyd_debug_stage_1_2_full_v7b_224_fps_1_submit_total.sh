#!/bin/bash
# ln -s /mnt_new/hanyudong.hyd/llama-vid/model_zoo/bert-base-uncased /root/.cache/huggingface/hub/bert-base-uncased
# wandb disabled
# deepspeed /mnt_new/hanyudong.hyd/Slow-Fast-Vid/llamavid/train/train_mem.py \
#     --deepspeed /mnt_new/hanyudong.hyd/Slow-Fast-Vid/scripts/zero1.json \
#     --model_name_or_path /mnt_new/hanyudong.hyd/llama-vid/model_zoo/LLM/vicuna/7B-V1.5 \
#     --version plain_guided \
#     --data_path /mnt_new/hanyudong.hyd/llama-vid/data/LLaMA-VID-Pretrain/llava_558k_with_webvid_new.json \
#     --image_folder /mnt_vlpt_hy/workspace/duyongchao.dyc/Downloads/datasets/llava_datasets/LLaVA-Pretrain/images \
#     --video_folder pcache://multimodalproxyi-pool.cz50c.alipay.com:39999/mnt/e6907b4b11a9bf76f739ed442b0281ab/webvid/videos \
#     --vision_tower /mnt_new/hanyudong.hyd/llama-vid/model_zoo/LAVIS/eva_vit_g.pth \
#     --image_processor /mnt_new/hanyudong.hyd/llama-vid/llamavid/processor/clip-patch14-224 \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --video_fps 1 \
# 	--bert_type "qformer_pretrain_freeze" \
#     --num_query 32 \
#     --pretrain_qformer /mnt_new/hanyudong.hyd/llama-vid/model_zoo/LAVIS/instruct_blip_vicuna7b_trimmed.pth \
#     --compress_type "mean" \
#     --bf16 True \
#     --output_dir /mnt_new/hanyudong.hyd/Slow-Fast-Vid/work_dirs/llama-vid-7b-pretrain-224-video-fps-1 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 1000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True

# deepspeed /mnt_new/hanyudong.hyd/Slow-Fast-Vid/llamavid/train/train_mem.py \
#     --deepspeed /mnt_new/hanyudong.hyd/Slow-Fast-Vid/scripts/zero1.json \
#     --model_name_or_path /mnt_new/hanyudong.hyd/llama-vid/model_zoo/LLM/vicuna/7B-V1.5 \
#     --version imgsp_v1 \
#     --data_path /mnt_new/hanyudong.hyd/llama-vid/data/LLaMA-VID-Finetune/new_mixed_dataset.json \
#     --image_folder /mnt_new/hanyudong.hyd/llama-vid/data/LLaMA-VID-Finetune \
#     --video_folder /mnt_new/hanyudong.hyd/llama-vid/data/LLaMA-VID-Finetune \
#     --vision_tower /mnt_new/hanyudong.hyd/llama-vid/model_zoo/LAVIS/eva_vit_g.pth \
#     --image_processor /mnt_new/hanyudong.hyd/llama-vid/llamavid/processor/clip-patch14-224 \
#     --pretrain_mm_mlp_adapter /mnt_new/hanyudong.hyd/Slow-Fast-Vid/work_dirs/llama-vid-7b-pretrain-224-video-fps-1/mm_projector.bin \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length False \
#     --video_fps 1 \
#     --bert_type "qformer_pretrain" \
#     --num_query 32 \
#     --compress_type "mean" \
#     --bf16 True \
#     --output_dir /mnt_new/hanyudong.hyd/Slow-Fast-Vid/work_dirs/llama-vid-7b-full-224-video-fps-1  \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 1000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \

deepspeed /mnt_new/hanyudong.hyd/Slow-Fast-Vid/llamavid/train/train_mem.py \
    --deepspeed /mnt_new/hanyudong.hyd/Slow-Fast-Vid/scripts/zero1.json \
    --model_name_or_path /mnt_new/hanyudong.hyd/Slow-Fast-Vid/work_dirs/llama-vid-7b-full-224-video-fps-1 \
    --version imgsp_v1 \
    --data_path /mnt_new/hanyudong.hyd/llama-vid/data/LLaMA-VID-Finetune/llava_v1_5_mix665k_with_only_video_modify_fmt.json \
    --image_folder /mnt_new/hanyudong.hyd/llama-vid/data/LLaMA-VID-Finetune \
    --video_folder /mnt_new/hanyudong.hyd/llama-vid/data/LLaMA-VID-Finetune \
    --vision_tower /mnt_new/hanyudong.hyd/llama-vid/model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor /mnt_new/hanyudong.hyd/llama-vid/llamavid/processor/clip-patch14-224 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --video_fps 1 \
    --bert_type "qformer_pretrain" \
    --num_query 32 \
    --compress_type "mean" \
    --bf16 True \
    --output_dir /mnt_new/hanyudong.hyd/Slow-Fast-Vid/work_dirs/llama-vid-7b-full-224-video-fps-1-post-training  \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1.5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
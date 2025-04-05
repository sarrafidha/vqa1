#!/bin/bash

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########

deepspeed changechat/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path hf-models/llavav1.5-7b \
    --version plain \
    --data_path /root/autodl-tmp/LEVIR-MCI-dataset/ChangeChat_instruct_34k.json \
    --image_folder /root/autodl-tmp/LEVIR-MCI-dataset/images  \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type single_stream_attention \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir experiments/changechat-pretrain-0801b \
    --num_train_epochs 10 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
#!/bin/bash

current_date=$(date +'%m%d_%H_%M')

export WANDB_PROJECT="pmod"
export WANDB_JOB_TYPE="finetune"
export WANDB_TAGS="v1.5"
export WANDB_MODE=online #offline

###########################Configurations############
mod_target_layers="shiftedcos_decay_0.85_0.15"
mod_target_token_type="vision_all"

mod_router_factor=0.5
mod_weight_norm="tanh_0.2_1" #{activation}_{scale}_{bias}:  f(x) = scale*activation(x)+bias

mod_bias_enabled=True
mod_special_init=True

image_aspect_ratio="pad"

run_name="$current_date-llava-v1.5-7b-ft-mod-$mod_target_layers-$mod_target_token_type-$mod_router_factor-$mod_weight_norm-$image_aspect_ratio-pretrain-square"

pretrained_checkpoint_path="./checkpoints/llava-v1.5-7b-pretrain/llava-official-checkpoint/mm_projector.bin"

per_device_train_batch_size=4
gradient_accumulation_steps=4

# run evaluation after training?
eval=true

output_dir="./checkpoints/llava-v1.5-7b/${run_name}"

deepspeed \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero1.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k_cleaned.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter $pretrained_checkpoint_path \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $output_dir\
    --num_train_epochs 1 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $run_name \
    --image_aspect_ratio $image_aspect_ratio \
    \
    --mod_target_layers $mod_target_layers \
    --mod_target_token_type $mod_target_token_type \
    --mod_router_factor $mod_router_factor \
    --mod_bias_enabled $mod_bias_enabled \
    --mod_special_init $mod_special_init \
    --mod_weight_norm $mod_weight_norm \
    \
    --log_level info \
    --log_level_replica error #\
    #--max_steps 20

# save the command to rerun the evaluation
echo "
export WANDB_TAGS="v1.5"

bash ./scripts/lmms-eval/eval.sh --ckpt $output_dir --run_name $run_name --project_name $WANDB_PROJECT
" > $output_dir/rerun_eval.sh

chmod +x $output_dir/rerun_eval.sh

# if eval is true, run evaluation
if $eval; then
    echo "---------------------------start evaluation---------------------------"
    bash ./scripts/lmms-eval/eval.sh \
        --ckpt $output_dir \
        --run_name $run_name \
        --project_name $WANDB_PROJECT #\
        #--eval_tasks ai2d
fi

echo "command for rerun evaluation: "
echo "bash $output_dir/rerun_eval.sh"

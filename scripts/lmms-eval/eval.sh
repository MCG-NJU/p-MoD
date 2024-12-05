#!/bin/bash

export HF_HUB_ENABLE_HF_TRANSFER="0" #https://lmms-lab.github.io/posts/lmms-eval-0.2/

eval_tasks_list=(
"ok_vqa"
"scienceqa_img"
"mme"
"seedbench"
"pope_full"
"realworldqa"
"gqa"
"ai2d"
"textvqa_val"
"chartqa"
"docvqa_val"
"infovqa_val"
"mmbench_en_dev"
"mmmu_val"
)

# Default values for keyword arguments
eval_tasks=$(IFS=,; echo "${eval_tasks_list[*]}")
ckpt="MCG-NJU/p-MoD-LLaVA-NeXT-7B"
conv_template="vicuna_v1"
GPUS=`nvidia-smi -L | wc -l` #count all GPUs
master_port=12345
run_name="pmod-llava-next-7b-ft"
project_name="pmod"
# accelerate config
accelerate_config=$HF_HOME/accelerate/default_config.yaml

# Parse keyword arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --eval_tasks) eval_tasks=("$2"); shift ;;
        --ckpt) ckpt="$2"; shift ;;
        --conv_template) conv_template="$2"; shift ;;
        --GPUS) GPUS="$2"; shift ;;
        --master_port) master_port="$2"; shift ;;
        --run_name) run_name="$2"; shift ;;
        --project_name) project_name="$2"; shift ;;
        --accelerate_config) accelerate_config="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check for commas in $ckpt and $conv_template, which will raise error
if [[ "$ckpt" == *","* ]]; then
    echo "Error: pretrained=($ckpt) contains a comma."
    exit 1
fi
if [[ "$conv_template" == *","* ]]; then
    echo "Error: conv_template=($conv_template) contains a comma."
    exit 1
fi

echo "Running script with:"
echo "Checkpoint: $ckpt"
echo "Conversation Template: $conv_template"
echo "Evaluation Tasks: $eval_tasks"
echo "GPUs: $GPUS"
echo "Master Port: $master_port"
echo "Project Name: $project_name"
echo "Run Name: $run_name"

# Check if the acclerate config file does not exist
if [ ! -f "$accelerate_config" ]; then
    echo "Accelerate config file does not exist. Will use \$HF_HOME/accelerate/default_config.yaml if it exists.
    You can modify default_config.yaml or create your own config with 'accelerate config --config_file $accelerate_config' "

    python3 -m accelerate.commands.launch  --num_processes=$GPUS --main_process_port=${master_port} \
      -m lmms_eval \
      --model llava   \
      --model_args="pretrained=$ckpt,conv_template=$conv_template" \
      --tasks=$eval_tasks  \
      --batch_size 1 \
      --log_samples \
      --log_samples_suffix lmms_eval \
      --output_path="$ckpt/logs/" \
      --wandb_args="project=$project_name,job_type=eval,name=$run_name"
else
    echo "Will use $accelerate_config as accelerate config."

    python3 -m accelerate.commands.launch --config_file $accelerate_config --num_processes=$GPUS --main_process_port=${master_port} \
        -m lmms_eval \
        --model llava   \
        --model_args="pretrained=$ckpt,conv_template=$conv_template" \
        --tasks=$eval_tasks  \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix lmms_eval \
        --output_path="$ckpt/logs/" \
        --wandb_args="project=$project_name,job_type=eval,name=$run_name"
fi


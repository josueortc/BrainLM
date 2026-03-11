#!/bin/bash
#SBATCH --job-name=brainlm-finetune
#SBATCH --output=./logs/log_finetune_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00

# Path to pretrained model (local dir or Hugging Face Hub id, e.g. josueortc/brainlm)
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"./training-runs/brainlm_run"}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-"./sample_data/train"}
VAL_DATASET_PATH=${VAL_DATASET_PATH:-"./sample_data/val"}
COORDS_DATASET_PATH=${COORDS_DATASET_PATH:-"./sample_data/coords"}

mkdir -p logs
cd "$(dirname "$0")/.."

python finetune.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --train_dataset_path "$TRAIN_DATASET_PATH" \
  --val_dataset_path "$VAL_DATASET_PATH" \
  --coords_dataset_path "$COORDS_DATASET_PATH" \
  --variable_of_interest_col_name Age.At.MHQ \
  --output_dir ./training-runs/finetune_run \
  --num_train_epochs 10 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 2e-5 \
  --wandb_logging false

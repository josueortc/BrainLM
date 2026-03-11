#!/bin/bash
#SBATCH --job-name=brainlm
#SBATCH --output=./logs/log_brainlm_%j.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

# Set paths to your Arrow datasets (train, val, coords).
# Generate sample data with: python generate_sample_data.py --output_dir ./sample_data
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-"./sample_data/train"}
VAL_DATASET_PATH=${VAL_DATASET_PATH:-"./sample_data/val"}
COORDS_DATASET_PATH=${COORDS_DATASET_PATH:-"./sample_data/coords"}

mkdir -p logs
cd "$(dirname "$0")/.."

python train.py \
  --output_dir ./training-runs/brainlm_run \
  --train_dataset_path "$TRAIN_DATASET_PATH" \
  --val_dataset_path "$VAL_DATASET_PATH" \
  --coords_dataset_path "$COORDS_DATASET_PATH" \
  --recording_col_name Voxelwise_RobustScaler_Normalized_Recording \
  --num_timepoints_per_voxel 200 \
  --timepoint_patching_size 20 \
  --hidden_size 256 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --intermediate_size 1024 \
  --mask_ratio 0.75 \
  --forward_mask_prob 0.5 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 64 \
  --num_train_epochs 20 \
  --logging_steps 50 \
  --eval_steps 50 \
  --save_steps 250 \
  --save_total_limit 3 \
  --loss_fn mse \
  --wandb_logging false \
  --fp16 true

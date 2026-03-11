# Training

## Pretraining from scratch

```bash
python train.py \
  --output_dir ./runs/my_run \
  --train_dataset_path /path/to/train \
  --val_dataset_path /path/to/val \
  --coords_dataset_path /path/to/coords \
  --num_timepoints_per_voxel 200 \
  --timepoint_patching_size 20 \
  --hidden_size 256 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --mask_ratio 0.75 \
  --forward_mask_prob 0.5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 64 \
  --num_train_epochs 20 \
  --loss_fn mse
```

Effective batch size is `per_device_train_batch_size * gradient_accumulation_steps * num_gpus`. Learning rate is set as `base_learning_rate * total_batch_size / 256` (default base 1e-3).

## Resume from checkpoint

If `output_dir` already contains a checkpoint, the script will resume automatically. To force a specific checkpoint:

```bash
python train.py ... --resume_from_checkpoint /path/to/checkpoint-XXXX
```

## Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `recording_col_name` | `Voxelwise_RobustScaler_Normalized_Recording` | Column name for the fMRI matrix |
| `num_timepoints_per_voxel` | 200 | Timepoints per sample (crop length) |
| `timepoint_patching_size` | 20 | Patch size (tokens per parcel = num_timepoints_per_voxel / this) |
| `mask_ratio` | 0.75 | Fraction of tokens masked (random masking) |
| `forward_mask_prob` | 0.5 | Probability of using forward masking instead of random |
| `loss_fn` | mse | `mse` or `mae` |
| `wandb_logging` | false | Set true to log to Weights & Biases |

## SLURM

Use `scripts/train.sh` after setting environment variables or editing the paths at the top. Submit with `sbatch scripts/train.sh`.

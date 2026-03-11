# Finetuning

Finetuning adds an MLP head on the **CLS token** and trains it to predict a scalar target (e.g. age, clinical score) from the same fMRI input format as pretraining.

## Requirements

- A pretrained BrainLM checkpoint (local directory or Hugging Face Hub id).
- Train and validation Arrow datasets that include the target column (e.g. `Age.At.MHQ`).
- The same coordinates dataset used for pretraining.

## Command

```bash
python finetune.py \
  --model_name_or_path ./runs/pretrained_run \
  --train_dataset_path /path/to/train \
  --val_dataset_path /path/to/val \
  --coords_dataset_path /path/to/coords \
  --variable_of_interest_col_name Age.At.MHQ \
  --output_dir ./runs/finetune_run \
  --num_train_epochs 10 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5
```

## Load from Hugging Face Hub

```bash
python finetune.py \
  --model_name_or_path josueortc/brainlm \
  ...
```

## Supported targets

Any numeric column in your dataset can be used. Examples: `Age.At.MHQ`, `PHQ9.Severity`, `PCL.Score`, `GAD7.Severity`, `Neuroticism`, etc. Rows with missing/NaN targets are filtered out.

## Loss

Regression loss is MSE or MAE, controlled by `--loss_fn` (must match the config’s `loss_fn` in the pretrained model).

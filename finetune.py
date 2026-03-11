"""
Finetune BrainLM on a scalar regression target (e.g. age, clinical score).
Uses the CLS token representation and an MLP head (BrainLMForFinetuning).
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from random import randint
from typing import Dict, Optional

import torch
import wandb
from datasets import load_from_disk

import transformers
from transformers import HfArgumentParser, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from brainlm_mae.configuration_brainlm import BrainLMConfig
from brainlm_mae.finetuning import BrainLMForFinetuning
from utils.trainer import BrainLMTrainer

logger = logging.getLogger(__name__)

check_min_version("4.28.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install datasets>=1.8.0")


@dataclass
class DataTrainingArguments:
    train_dataset_path: str = field(metadata={"help": "Path to train Arrow dataset."})
    val_dataset_path: str = field(metadata={"help": "Path to validation Arrow dataset."})
    variable_of_interest_col_name: str = field(
        default="Age.At.MHQ",
        metadata={"help": "Column name for regression target (e.g. Age.At.MHQ, PCL.Score)."},
    )
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Max train samples for debugging."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Max eval samples for debugging."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path or Hugging Face Hub model id (e.g. josueortc/brainlm)."}
    )
    config_name: Optional[str] = field(default=None, metadata={"help": "Config path if different."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache dir for downloads."})
    model_revision: str = field(default="main", metadata={"help": "Model revision."})


@dataclass
class CustomTrainingArguments(TrainingArguments):
    report_to: str = field(default="none", metadata={"help": "Use manual wandb; set report_to to none."})
    remove_unused_columns: bool = field(default=False)
    wandb_logging: bool = field(default=False, metadata={"help": "Log to Weights & Biases."})
    wandb_path: str = field(default="./")
    recording_col_name: str = field(
        default="Voxelwise_RobustScaler_Normalized_Recording",
        metadata={"help": "Column containing fMRI recording (timepoints x parcels)."},
    )
    num_timepoints_per_voxel: int = field(default=200, metadata={"help": "Timepoints per sample."})
    timepoint_patching_size: int = field(default=20, metadata={"help": "Patch size."})
    coords_dataset_path: str = field(default="./", metadata={"help": "Path to coordinates Arrow dataset."})
    loss_fn: str = field(default="mse", metadata={"help": "Regression loss: mse or mae."})


def collate_fn(examples):
    signal_vectors = torch.stack([e["signal_vectors"] for e in examples], dim=0)
    xyz_vectors = torch.stack([e["xyz_vectors"] for e in examples])
    labels = torch.tensor([e["label"] for e in examples], dtype=torch.float32)
    return {
        "signal_vectors": signal_vectors,
        "xyz_vectors": xyz_vectors,
        "labels": labels,
    }


class FinetuneMetricsCalculator:
    """MSE, MAE, R2 for scalar regression finetuning."""

    def __init__(self) -> None:
        self.current_epoch = 0

    def __call__(self, eval_pred) -> Dict:
        import numpy as np
        from sklearn.metrics import r2_score

        preds = eval_pred.predictions
        if isinstance(preds, tuple):
            logits = preds[0]
        else:
            logits = preds
        logits = np.squeeze(np.asarray(logits))
        labels = np.squeeze(np.asarray(eval_pred.label_ids))
        mse = float(((logits - labels) ** 2).mean())
        mae = float(np.abs(logits - labels).mean())
        r2 = r2_score(labels, logits)
        r2 = max(0.0, r2)
        return {"mse": mse, "mae": mae, "r2": r2}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert training_args.num_timepoints_per_voxel % training_args.timepoint_patching_size == 0

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    if training_args.wandb_logging:
        wandb.init(project="BrainLM", name=os.path.basename(training_args.output_dir.rstrip("/")))

    train_ds = load_from_disk(data_args.train_dataset_path)
    val_ds = load_from_disk(data_args.val_dataset_path)
    coords_ds = load_from_disk(training_args.coords_dataset_path)

    target_col = data_args.variable_of_interest_col_name
    train_ds = train_ds.filter(lambda x: x[target_col] is not None and not (isinstance(x[target_col], float) and (x[target_col] != x[target_col])))
    val_ds = val_ds.filter(lambda x: x[target_col] is not None and not (isinstance(x[target_col], float) and (x[target_col] != x[target_col])))
    if data_args.max_train_samples is not None:
        train_ds = train_ds.shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
    if data_args.max_eval_samples is not None:
        val_ds = val_ds.shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))

    config = BrainLMConfig.from_pretrained(
        model_args.model_name_or_path or model_args.config_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    config.loss_fn = training_args.loss_fn

    model = BrainLMForFinetuning.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        ignore_mismatched_sizes=True,
    )

    recording_col_name = training_args.recording_col_name
    num_timepoints_per_voxel = training_args.num_timepoints_per_voxel
    window_xyz_list = []
    for i in range(424):
        window_xyz_list.append(
            torch.tensor(
                [coords_ds[i]["X"], coords_ds[i]["Y"], coords_ds[i]["Z"]],
                dtype=torch.float32,
            )
        )
    window_xyz_list = torch.stack(window_xyz_list)

    def preprocess(examples):
        signal_vector = []
        xyz_list = []
        labels_out = []
        for idx in range(len(examples[recording_col_name])):
            signal_window = torch.tensor(examples[recording_col_name][idx])
            start_idx = randint(0, signal_window.shape[0] - num_timepoints_per_voxel)
            end_idx = start_idx + num_timepoints_per_voxel
            signal_window = signal_window[start_idx:end_idx, :]
            signal_window = torch.movedim(signal_window, 0, 1)
            signal_vector.append(signal_window)
            xyz_list.append(window_xyz_list)
            labels_out.append(float(examples[target_col][idx]))
        examples["signal_vectors"] = signal_vector
        examples["xyz_vectors"] = xyz_list
        examples["label"] = labels_out
        return examples

    train_ds.set_transform(preprocess)
    val_ds.set_transform(preprocess)

    metrics_calculator = FinetuneMetricsCalculator()
    last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None

    trainer = BrainLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=metrics_calculator,
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()

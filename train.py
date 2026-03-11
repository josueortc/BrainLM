"""
Hugging Face training script for BrainLM masked autoencoder pretraining.
Decoder-only transformer with block-causal attention.
"""

import logging
import os
import sys

from dataclasses import dataclass, field
from pathlib import Path
from random import randint
from typing import Optional

import torch
import wandb
from datasets import DatasetDict, load_from_disk

import transformers
from transformers import HfArgumentParser, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from brainlm_mae.configuration_brainlm import BrainLMConfig
from brainlm_mae.modeling_brainlm import BrainLMForPretraining
from utils.trainer import BrainLMTrainer
from utils.metrics import MetricsCalculator

logger = logging.getLogger(__name__)

check_min_version("4.28.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install datasets>=1.8.0")


@dataclass
class DataTrainingArguments:
    """Arguments for data input to the model."""

    train_dataset_path: str = field(
        metadata={"help": "Path to saved train Arrow dataset."}
    )
    val_dataset_path: str = field(
        metadata={"help": "Path to saved val Arrow dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging, truncate the number of training examples."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging, truncate the number of evaluation examples."},
    )


@dataclass
class ModelArguments:
    """Arguments for model/config initialization."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model checkpoint for weight initialization. Leave unset to train from scratch. "
            "Can be a local path or a Hugging Face Hub model id (e.g. josueortc/brainlm)."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if different from model_name_or_path."},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override default config settings when training from scratch. "
            "Example: n_embd=10,resid_pdrop=0.2"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store pretrained models downloaded from hub."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Specific model version to use (branch, tag, or commit id)."},
    )
    hidden_size: int = field(default=256, metadata={"help": "Transformer hidden size."})
    num_hidden_layers: int = field(default=8, metadata={"help": "Number of transformer layers."})
    num_attention_heads: int = field(
        default=8, metadata={"help": "Number of attention heads per layer."}
    )
    intermediate_size: int = field(
        default=1024, metadata={"help": "Intermediate size in FFN layers."}
    )
    hidden_dropout_prob: float = field(
        default=0.0, metadata={"help": "Dropout probability for layer activations."}
    )
    attention_probs_dropout_prob: float = field(
        default=0.1, metadata={"help": "Dropout probability for attention coefficients."}
    )
    mask_ratio: float = field(
        default=0.75, metadata={"help": "Ratio of masked tokens per voxel."}
    )
    forward_mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "Probability of using forward temporal masking instead of random masking. "
            "0.0 = always random, 1.0 = always forward."
        },
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Extended training arguments for BrainLM."""

    report_to: str = field(
        default="none",
        metadata={"help": "Disable HuggingFace built-in reporting; we use manual wandb logging."},
    )
    remove_unused_columns: bool = field(
        default=False, metadata={"help": "Don't remove unused columns."}
    )
    do_train: bool = field(default=True, metadata={"help": "Whether to train."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to evaluate."})
    base_learning_rate: float = field(
        default=1e-3,
        metadata={"help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."},
    )
    lr_scheduler_type: str = field(
        default="cosine_with_restarts",
        metadata={"help": "Learning rate scheduler type."},
    )
    weight_decay: float = field(
        default=1e-5, metadata={"help": "Weight decay (L2 regularization)."}
    )
    num_train_epochs: int = field(default=20, metadata={"help": "Number of training epochs."})
    warmup_ratio: float = field(
        default=0.05, metadata={"help": "Warmup ratio for learning rate scheduler."}
    )
    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "Training batch size per device."}
    )
    per_device_eval_batch_size: int = field(
        default=4, metadata={"help": "Evaluation batch size per device."}
    )
    logging_strategy: str = field(default="steps", metadata={"help": "Logging strategy."})
    logging_steps: int = field(default=50, metadata={"help": "Log training metrics every X steps."})
    evaluation_strategy: str = field(default="steps", metadata={"help": "Evaluation strategy."})
    eval_steps: int = field(default=50, metadata={"help": "Evaluate every X steps."})
    save_strategy: str = field(default="steps", metadata={"help": "Save strategy."})
    save_steps: int = field(default=250, metadata={"help": "Save checkpoint every X steps."})
    load_best_model_at_end: bool = field(
        default=True, metadata={"help": "Load best model at end of training."}
    )
    save_total_limit: int = field(default=3, metadata={"help": "Max checkpoints to save."})
    seed: int = field(default=42, metadata={"help": "Random seed."})
    wandb_logging: bool = field(
        default=False, metadata={"help": "Whether to log metrics to Weights & Biases."}
    )
    wandb_path: str = field(default="./", metadata={"help": "Path for wandb cache."})
    include_inputs_for_metrics: bool = field(
        default=True,
        metadata={"help": "Include model inputs in metrics calculation."},
    )
    loss_fn: str = field(default="mse", metadata={"help": "Loss function: 'mse' or 'mae'."})
    recording_col_name: str = field(
        default="Voxelwise_RobustScaler_Normalized_Recording",
        metadata={"help": "Column in dataset containing the fMRI recording (timepoints x parcels)."},
    )
    num_timepoints_per_voxel: int = field(
        default=200,
        metadata={
            "help": "Number of timepoints per voxel in one input sample. "
            "Must be divisible by timepoint_patching_size."
        },
    )
    timepoint_patching_size: int = field(
        default=20,
        metadata={"help": "Number of consecutive timepoints grouped into one patch token."},
    )
    coords_dataset_path: str = field(
        default="./",
        metadata={"help": "Path to saved Arrow dataset of brain region coordinates (X, Y, Z)."},
    )
    fp16: bool = field(default=False, metadata={"help": "Whether to use fp16 training."})


def collate_fn(examples):
    """Stack a batch of examples into model input format."""
    signal_vectors = torch.stack([example["signal_vectors"] for example in examples], dim=0)
    xyz_vectors = torch.stack([example["xyz_vectors"] for example in examples])
    labels = torch.stack([example["label"] for example in examples])
    return {
        "signal_vectors": signal_vectors,
        "xyz_vectors": xyz_vectors,
        "input_ids": signal_vectors,
        "labels": labels,
    }


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert training_args.num_timepoints_per_voxel % training_args.timepoint_patching_size == 0, (
        "num_timepoints_per_voxel must be divisible by timepoint_patching_size."
    )

    send_example_telemetry("run_brainlm", model_args, data_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming at {last_checkpoint}.")

    if training_args.wandb_logging:
        wandb.init(
            project="BrainLM",
            name=os.path.basename(training_args.output_dir.rstrip("/")),
            dir=training_args.wandb_path,
        )

    train_ds = load_from_disk(data_args.train_dataset_path)
    val_ds = load_from_disk(data_args.val_dataset_path)
    ds = DatasetDict({"train": train_ds, "validation": val_ds})
    coords_ds = load_from_disk(training_args.coords_dataset_path)

    config_kwargs = {"cache_dir": model_args.cache_dir, "revision": model_args.model_revision}
    if model_args.config_name:
        config = BrainLMConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = BrainLMConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        config.update(
            {
                "mask_ratio": model_args.mask_ratio,
                "forward_mask_prob": model_args.forward_mask_prob,
                "attention_probs_dropout_prob": model_args.attention_probs_dropout_prob,
                "timepoint_patching_size": training_args.timepoint_patching_size,
                "num_timepoints_per_voxel": training_args.num_timepoints_per_voxel,
                "loss_fn": training_args.loss_fn,
            }
        )
    else:
        config = BrainLMConfig(
            hidden_size=model_args.hidden_size,
            num_hidden_layers=model_args.num_hidden_layers,
            num_attention_heads=model_args.num_attention_heads,
            intermediate_size=model_args.intermediate_size,
            hidden_dropout_prob=model_args.hidden_dropout_prob,
            attention_probs_dropout_prob=model_args.attention_probs_dropout_prob,
            num_timepoints_per_voxel=training_args.num_timepoints_per_voxel,
            mask_ratio=model_args.mask_ratio,
            forward_mask_prob=model_args.forward_mask_prob,
            timepoint_patching_size=training_args.timepoint_patching_size,
            loss_fn=training_args.loss_fn,
        )
        if model_args.config_overrides:
            config.update_from_string(model_args.config_overrides)

    if model_args.model_name_or_path:
        model = BrainLMForPretraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            ignore_mismatched_sizes=True,
        )
    else:
        model = BrainLMForPretraining(config)

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

    def preprocess_fmri(examples):
        label = torch.tensor(1, dtype=torch.int64)
        signal_vector = []
        xyz_list = []
        labels = []
        for idx in range(len(examples[recording_col_name])):
            signal_window = torch.tensor(examples[recording_col_name][idx])
            start_idx = randint(0, signal_window.shape[0] - num_timepoints_per_voxel)
            end_idx = start_idx + num_timepoints_per_voxel
            signal_window = signal_window[start_idx:end_idx, :]
            signal_window = torch.movedim(signal_window, 0, 1)
            signal_vector.append(signal_window)
            xyz_list.append(window_xyz_list)
            labels.append(label)
        examples["signal_vectors"] = signal_vector
        examples["xyz_vectors"] = xyz_list
        examples["label"] = labels
        return examples

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            ds["train"] = ds["train"].shuffle(seed=training_args.seed).select(
                range(data_args.max_train_samples)
            )
        ds["train"].set_transform(preprocess_fmri)
    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            ds["validation"] = ds["validation"].shuffle(seed=training_args.seed).select(
                range(data_args.max_eval_samples)
            )
        ds["validation"].set_transform(preprocess_fmri)

    total_train_batch_size = (
        training_args.train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = (
            training_args.base_learning_rate * total_train_batch_size / 256
        )

    metrics_calculator = MetricsCalculator()
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    trainer = BrainLMTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["validation"] if training_args.do_eval else None,
        data_collator=collate_fn,
        compute_metrics=metrics_calculator,
    )

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()

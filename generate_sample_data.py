"""
Generate synthetic Arrow datasets for BrainLM pretraining and finetuning.
Output format matches the expected schema: recording (timepoints x 424 parcels),
plus optional metadata columns for finetuning.
"""

import argparse
import os

import numpy as np
from datasets import Dataset


RECORDING_COL = "Voxelwise_RobustScaler_Normalized_Recording"
NUM_PARCELS = 424
DEFAULT_NUM_TIMEPOINTS = 500
METADATA_COLUMNS = [
    "Age.At.MHQ",
    "PHQ9.Severity",
    "PCL.Score",
    "GAD7.Severity",
    "Neuroticism",
    "Depressed.At.Baseline",
    "Self.Harm.Ever",
    "Not.Worth.Living",
]


def make_recording(rng: np.random.Generator, num_timepoints: int = DEFAULT_NUM_TIMEPOINTS) -> np.ndarray:
    """Random recording of shape (num_timepoints, 424), roughly z-scored."""
    x = rng.standard_normal((num_timepoints, NUM_PARCELS)).astype(np.float32)
    return x


def make_metadata_row(rng: np.random.Generator) -> dict:
    """One row of synthetic metadata for finetuning."""
    return {
        "Age.At.MHQ": float(rng.integers(18, 80)),
        "PHQ9.Severity": float(rng.integers(0, 4)),
        "PCL.Score": float(rng.uniform(0, 40)),
        "GAD7.Severity": float(rng.integers(0, 4)),
        "Neuroticism": float(rng.uniform(0, 24)),
        "Depressed.At.Baseline": float(rng.integers(0, 2)),
        "Self.Harm.Ever": float(rng.integers(0, 2)),
        "Not.Worth.Living": float(rng.integers(0, 2)),
    }


def build_train_val(
    num_train: int = 100,
    num_val: int = 20,
    num_timepoints: int = DEFAULT_NUM_TIMEPOINTS,
    seed: int = 42,
) -> tuple:
    """Build train and val datasets with recording + metadata."""
    rng = np.random.default_rng(seed)
    train_recordings = [make_recording(rng, num_timepoints) for _ in range(num_train)]
    val_recordings = [make_recording(rng, num_timepoints) for _ in range(num_val)]
    train_meta = [make_metadata_row(rng) for _ in range(num_train)]
    val_meta = [make_metadata_row(rng) for _ in range(num_val)]

    train_data = {
        RECORDING_COL: train_recordings,
        **{k: [m[k] for m in train_meta] for k in METADATA_COLUMNS},
    }
    val_data = {
        RECORDING_COL: val_recordings,
        **{k: [m[k] for m in val_meta] for k in METADATA_COLUMNS},
    }

    train_ds = Dataset.from_dict(train_data)
    val_ds = Dataset.from_dict(val_data)
    return train_ds, val_ds


def build_coords(seed: int = 42) -> Dataset:
    """Build coordinates dataset: 424 rows with X, Y, Z (MNI-like)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-80, 80, NUM_PARCELS).astype(np.float64)
    y = rng.uniform(-120, 80, NUM_PARCELS).astype(np.float64)
    z = rng.uniform(-60, 80, NUM_PARCELS).astype(np.float64)
    return Dataset.from_dict({"X": x.tolist(), "Y": y.tolist(), "Z": z.tolist()})


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic BrainLM Arrow datasets.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sample_data",
        help="Directory to write train/val/coords subdirs.",
    )
    parser.add_argument("--num_train", type=int, default=100, help="Number of train samples.")
    parser.add_argument("--num_val", type=int, default=20, help="Number of validation samples.")
    parser.add_argument(
        "--num_timepoints",
        type=int,
        default=DEFAULT_NUM_TIMEPOINTS,
        help="Timepoints per recording (must be >= num_timepoints_per_voxel used in training).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_ds, val_ds = build_train_val(
        num_train=args.num_train,
        num_val=args.num_val,
        num_timepoints=args.num_timepoints,
        seed=args.seed,
    )
    coords_ds = build_coords(seed=args.seed)

    train_path = os.path.join(args.output_dir, "train")
    val_path = os.path.join(args.output_dir, "val")
    coords_path = os.path.join(args.output_dir, "coords")

    train_ds.save_to_disk(train_path)
    val_ds.save_to_disk(val_path)
    coords_ds.save_to_disk(coords_path)

    print(f"Saved train to {train_path} ({len(train_ds)} samples)")
    print(f"Saved val to {val_path} ({len(val_ds)} samples)")
    print(f"Saved coords to {coords_path} ({len(coords_ds)} rows)")
    print(f"Recording shape per sample: ({args.num_timepoints}, {NUM_PARCELS})")
    print("Use with train.py:")
    print(f"  --train_dataset_path {train_path} --val_dataset_path {val_path} --coords_dataset_path {coords_path}")


if __name__ == "__main__":
    main()

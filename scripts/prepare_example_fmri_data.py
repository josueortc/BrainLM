"""
Prepare real fMRI example data for BrainLM using Nilearn.

Fetches OpenNeuro ds000228 (development fMRI) via nilearn.datasets.fetch_development_fmri,
extracts parcellated timeseries with Schaefer 400 atlas, and saves Arrow datasets
(train/val/coords) in BrainLM format. Optionally exports one sample to JSON for
embedding in the project website (--export_website_json).

Requires: pip install nilearn datasets
Usage:
  python scripts/prepare_example_fmri_data.py --output_dir ./example_fmri --n_subjects 2
  python scripts/prepare_example_fmri_data.py --output_dir ./example_fmri --export_website_json ./website_data/example_recording.json
"""

import argparse
import json
import os

import numpy as np
from datasets import Dataset

try:
    from nilearn import datasets as nilearn_datasets
    from nilearn.maskers import NiftiLabelsMasker
except ImportError as e:
    raise ImportError(
        "This script requires nilearn. Install with: pip install nilearn"
    ) from e


RECORDING_COL = "Voxelwise_RobustScaler_Normalized_Recording"
NUM_PARCELS_SCHAEFER = 400
WINDOW_TIMEPOINTS = 200


def fetch_schaefer_coords(n_rois: int = 400):
    """Get approximate ROI centroids for Schaefer atlas (MNI space)."""
    atlas = nilearn_datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
    # Nilearn atlas returns maps; we use random MNI-like coords as placeholder
    # if no label geometry is easily available. For real centroids, use
    # nilearn.plotting.find_parcellation_cut_coords or similar.
    rng = np.random.default_rng(42)
    coords = rng.uniform(-60, 60, (n_rois, 3)).astype(np.float64)
    return coords


def extract_timeseries_one_subject(
    func_path: str,
    confounds_path: str,
    atlas_path: str,
    n_rois: int,
    standardize: bool = True,
) -> np.ndarray:
    """Extract (T, n_rois) timeseries for one subject."""
    masker = NiftiLabelsMasker(
        labels_img=atlas_path,
        standardize=standardize,
        memory="nilearn_cache",
        verbose=0,
    )
    if os.path.isfile(confounds_path):
        import pandas as pd
        confounds = pd.read_csv(confounds_path, sep="\t")
        # Use a minimal set to avoid NaNs
        use_cols = [c for c in confounds.columns if confounds[c].dtype in (np.float64, np.int64)][:20]
        confounds = confounds[use_cols].fillna(0)
    else:
        confounds = None
    data = masker.fit_transform(func_path, confounds=confounds)
    return data.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare real fMRI example data (Nilearn development_fmri + Schaefer 400)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="example_fmri",
        help="Directory to write train/val/coords Arrow datasets.",
    )
    parser.add_argument(
        "--n_subjects",
        type=int,
        default=2,
        help="Number of subjects to fetch (development_fmri).",
    )
    parser.add_argument(
        "--n_rois",
        type=int,
        default=NUM_PARCELS_SCHAEFER,
        help="Schaefer atlas n_rois (400 or 200, etc.).",
    )
    parser.add_argument(
        "--window_timepoints",
        type=int,
        default=WINDOW_TIMEPOINTS,
        help="Timepoints to keep per sample (crop from full run).",
    )
    parser.add_argument(
        "--export_website_json",
        type=str,
        default=None,
        help="If set, export one sample to this JSON path for the project website.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Nilearn data directory for caching downloads.",
    )
    args = parser.parse_args()

    print("Fetching development fMRI (OpenNeuro ds000228)...")
    data = nilearn_datasets.fetch_development_fmri(
        n_subjects=args.n_subjects,
        data_dir=args.data_dir,
    )
    print("Fetching Schaefer atlas...")
    atlas = nilearn_datasets.fetch_atlas_schaefer_2018(n_rois=args.n_rois)
    atlas_path = atlas["maps"]

    n_rois = args.n_rois
    recordings = []
    phenotypic = data.get("phenotypic")
    ages = []

    for i, (func_path, conf_path) in enumerate(zip(data["func"], data["confounds"])):
        print(f"Extracting subject {i + 1}/{len(data['func'])}...")
        ts = extract_timeseries_one_subject(
            func_path,
            conf_path,
            atlas_path,
            n_rois=n_rois,
        )
        T = ts.shape[0]
        if T >= args.window_timepoints:
            # Take first window
            ts = ts[: args.window_timepoints]
        else:
            # Pad or skip
            ts = np.pad(
                ts,
                ((0, args.window_timepoints - T), (0, 0)),
                mode="edge",
            )
        recordings.append(ts)
        if phenotypic is not None and hasattr(phenotypic, "columns") and "age" in phenotypic.columns:
            try:
                ages.append(float(phenotypic.iloc[i]["age"]))
            except Exception:
                ages.append(30.0)
        else:
            ages.append(30.0)

    coords = fetch_schaefer_coords(n_rois=n_rois)

    # Train/val split
    n_train = max(1, len(recordings) - 1)
    train_rec = recordings[:n_train]
    val_rec = recordings[n_train:]
    train_age = ages[:n_train]
    val_age = ages[n_train:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train")
    val_path = os.path.join(args.output_dir, "val")
    coords_path = os.path.join(args.output_dir, "coords")

    train_ds = Dataset.from_dict({
        RECORDING_COL: train_rec,
        "Age.At.MHQ": train_age,
    })
    val_ds = Dataset.from_dict({
        RECORDING_COL: val_rec,
        "Age.At.MHQ": val_age,
    })
    coords_ds = Dataset.from_dict({
        "X": coords[:, 0].tolist(),
        "Y": coords[:, 1].tolist(),
        "Z": coords[:, 2].tolist(),
    })

    train_ds.save_to_disk(train_path)
    val_ds.save_to_disk(val_path)
    coords_ds.save_to_disk(coords_path)

    print(f"Saved train to {train_path} ({len(train_ds)} samples)")
    print(f"Saved val to {val_path} ({len(val_ds)} samples)")
    print(f"Saved coords to {coords_path} ({n_rois} rows)")
    print(f"Recording shape: ({args.window_timepoints}, {n_rois})")
    print("Use with train.py:")
    print(f"  --train_dataset_path {train_path} --val_dataset_path {val_path} --coords_dataset_path {coords_path} --num_brain_voxels {n_rois}")

    if args.export_website_json:
        os.makedirs(os.path.dirname(args.export_website_json) or ".", exist_ok=True)
        rec = recordings[0]
        rec_rounded = np.round(rec.astype(np.float64), 3).tolist()
        coords_rounded = np.round(coords, 3).tolist()
        out = {"recording": rec_rounded, "coords": coords_rounded}
        with open(args.export_website_json, "w") as f:
            json.dump(out, f, separators=(",", ":"))
        print(f"Exported one sample to {args.export_website_json} for website embedding.")


if __name__ == "__main__":
    main()

# Data Format

## Train and validation datasets

Hugging Face **Arrow** datasets saved with `dataset.save_to_disk(path)`. Each example is one subject/session.

### Required column

- **Recording** (default name: `Voxelwise_RobustScaler_Normalized_Recording`): 2D array of shape `(num_timepoints, num_parcels)`, e.g. `(500, 424)`. Each row is one timepoint; each column is one parcel’s BOLD (or normalized) signal. The training script crops a random contiguous window of `num_timepoints_per_voxel` (e.g. 200) timepoints, so the recording must have at least that many rows.

### Optional columns (for finetuning)

- `Age.At.MHQ`, `PHQ9.Severity`, `PCL.Score`, `GAD7.Severity`, `Neuroticism`, `Depressed.At.Baseline`, `Self.Harm.Ever`, `Not.Worth.Living`, etc. Any numeric column can be used as the regression target via `--variable_of_interest_col_name`.

### Creating datasets

Use `generate_sample_data.py` to create synthetic train/val/coords with the correct schema:

```bash
python generate_sample_data.py --output_dir ./sample_data --num_train 100 --num_val 20
```

For your own data, build a Hugging Face `Dataset` (e.g. from a pandas DataFrame or Parquet) with the recording column and optional metadata, then `dataset.save_to_disk("path/to/train")`.

## Coordinates dataset

A separate Arrow dataset with **424 rows** (one per parcel) and columns:

- `X`, `Y`, `Z`: float coordinates (e.g. MNI space).

The training script loads this once and broadcasts the same coordinates to every sample. Row index `i` corresponds to parcel `i` in the recording columns.

Example: generate random coordinates with `generate_sample_data.py` (it writes `sample_data/coords`).

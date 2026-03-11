# Architecture

## Overview

BrainLM is a **decoder-only** transformer: all tokens (masked and unmasked) pass through a single stack of layers. There is no separate encoder or decoder.

## Tokenization

- Each brain parcel has a time series of length `T` (e.g. 200). It is split into `T / P` patches of size `P` (e.g. 20), so each parcel yields 10 tokens.
- Each patch is projected to hidden size `H` (e.g. 256) and receives:
  - **Spatial embedding**: learned linear projection of the parcel’s 3D coordinates (x, y, z).
  - **Temporal encoding**: sinusoidal positional encoding along the time axis.
- Tokens are arranged in **timepoint-major** order: all parcels at t=0, then all parcels at t=1, etc. A learned CLS token is prepended.

## Block-causal attention

- Parcels at the same timepoint can attend to each other (bidirectional within the block).
- Parcels at time `t` can attend to all parcels at times `0 .. t-1`.
- No token can attend to future timepoints.
- The CLS token attends to and is attended by all tokens.

This is implemented with a block lower-triangular mask passed to `scaled_dot_product_attention`.

## Masking

- **Random**: a fraction (e.g. 75%) of tokens per parcel is replaced with a learned `[MASK]` embedding.
- **Forward**: only the last temporal token of each parcel is masked (simulating next-timepoint prediction).
- The model predicts the original signal at masked positions; loss is MSE or MAE on those positions only.

## Output

- **Pretraining**: reconstruction loss and predicted patch values (voxel-major shape `[B, V, num_tokens, P]`).
- **Finetuning**: a 3-layer MLP on the CLS token output for scalar regression (e.g. age, clinical score).

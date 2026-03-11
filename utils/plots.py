import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None


def plot_masked_pred_trends_one_sample(
    pred_logits: np.ndarray,
    signal_vectors: np.ndarray,
    mask: np.ndarray,
    sample_idx: int,
    node_idxs: list,
    dataset_split: str,
    epoch: int,
) -> None:
    """Plot timeseries of predictions vs ground truth for one sample and selected voxels."""
    fig, axes = plt.subplots(nrows=len(node_idxs), ncols=1, sharex=True, sharey=True)
    if len(node_idxs) == 1:
        axes = [axes]
    fig.set_figwidth(25)
    fig.set_figheight(3 * len(node_idxs))

    _, _, num_tokens, time_patch_preds = pred_logits.shape

    for row_idx, node_idx in enumerate(node_idxs):
        ax = axes[row_idx]
        input_data_vals = []
        input_data_timepoints = []
        for token_idx in range(signal_vectors.shape[2]):
            input_data_vals += signal_vectors[sample_idx, node_idx, token_idx].tolist()
            start_timepoint = time_patch_preds * token_idx
            end_timepoint = start_timepoint + time_patch_preds
            input_data_timepoints += list(range(start_timepoint, end_timepoint))
            if mask[sample_idx, node_idx, token_idx] == 1:
                model_pred_vals = pred_logits[sample_idx, node_idx, token_idx].tolist()
                model_pred_timepoints = list(range(start_timepoint, end_timepoint))
                ax.plot(
                    model_pred_timepoints,
                    model_pred_vals,
                    marker=".",
                    markersize=3,
                    label="Masked Predictions",
                    color="orange",
                )
        ax.plot(
            input_data_timepoints,
            input_data_vals,
            marker=".",
            markersize=3,
            label="Input Data",
            color="green",
        )
        ax.set_title(f"Sample {sample_idx}, Voxel {node_idx}")
        ax.axhline(y=0.0, color="gray", linestyle="--", markersize=2)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.9))
    plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.95])
    fig.supxlabel("Timepoint")
    fig.supylabel("Prediction Value")
    plt.suptitle(f"Ground Truth Signal vs Masked Prediction ({dataset_split} split, epoch {epoch})")
    if wandb is not None and wandb.run is not None:
        wandb.log(
            {"trends/pred_trend_{}_sample{}".format(dataset_split, sample_idx): wandb.Image(fig)},
            commit=False,
        )
    plt.close()

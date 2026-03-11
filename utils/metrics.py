from typing import Dict

import numpy as np
from sklearn.metrics import r2_score
from transformers.trainer_utils import EvalPrediction

from utils.plots import plot_masked_pred_trends_one_sample


class MetricsCalculator:
    """Metric calculator for BrainLM pretraining (per-parcel decoder-only model)."""

    def __init__(self) -> None:
        self.current_epoch = 0

    def __call__(self, eval_pred_obj: EvalPrediction) -> Dict:
        predictions = eval_pred_obj.predictions
        logits_tuple = predictions[0]
        mask = predictions[1]
        return self._base_metrics(logits_tuple, mask, eval_pred_obj)

    def _base_metrics(self, logits_tuple, mask, eval_pred_obj: EvalPrediction) -> Dict:
        pred_logits, _ = logits_tuple
        signal_vectors_padded = eval_pred_obj.inputs
        signal_vectors = signal_vectors_padded[: pred_logits.shape[0], :]
        signal_vectors = np.reshape(signal_vectors, pred_logits.shape)

        mse = self.calculate_mse(pred_logits, signal_vectors, mask)
        mae = self.calculate_mae(pred_logits, signal_vectors, mask)
        mask_bool = mask.astype(bool)
        unadjusted_r2 = self.calculate_r_squared_masked(pred_logits, signal_vectors, mask_bool)

        plot_masked_pred_trends_one_sample(
            pred_logits=pred_logits,
            signal_vectors=signal_vectors,
            mask=mask,
            sample_idx=0,
            node_idxs=[0, 100, 200],
            dataset_split="val",
            epoch=self.current_epoch,
        )
        plot_masked_pred_trends_one_sample(
            pred_logits=pred_logits,
            signal_vectors=signal_vectors,
            mask=mask,
            sample_idx=1,
            node_idxs=[0, 100, 200],
            dataset_split="val",
            epoch=self.current_epoch,
        )

        return {"mse": mse, "mae": mae, "r2": unadjusted_r2}

    @staticmethod
    def calculate_mse(pred_values, signal_values, mask):
        mask = np.expand_dims(mask, axis=-1).repeat(pred_values.shape[-1], axis=-1)
        if mask.sum() == 0:
            mask = 1 - mask
        return float((((pred_values - signal_values) ** 2) * mask).sum() / mask.sum())

    @staticmethod
    def calculate_mae(pred_values, signal_values, mask):
        mask = np.expand_dims(mask, axis=-1).repeat(pred_values.shape[-1], axis=-1)
        if mask.sum() == 0:
            mask = 1 - mask
        return float((np.abs((pred_values - signal_values) * mask).sum() / mask.sum()))

    @staticmethod
    def calculate_r_squared_masked(pred_values, signal_values, mask):
        gt_list = []
        pred_vals_list = []
        if mask.sum() == 0:
            mask = ~mask
        for sample_idx in range(signal_values.shape[0]):
            for voxel_idx in range(signal_values.shape[1]):
                gt_list += list(
                    signal_values[sample_idx, voxel_idx][mask[sample_idx, voxel_idx]].flatten()
                )
                pred_vals_list += list(
                    pred_values[sample_idx, voxel_idx][mask[sample_idx, voxel_idx]].flatten()
                )
        r_squared = r2_score(y_true=gt_list, y_pred=pred_vals_list)
        return max(0.0, r_squared)

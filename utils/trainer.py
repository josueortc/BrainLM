from typing import Dict, Union

import torch
import wandb
from transformers import Trainer

from utils.plots import plot_masked_pred_trends_one_sample


class BrainLMTrainer(Trainer):
    """Custom HuggingFace Trainer with wandb logging and prediction trend plots."""

    def log(self, logs: Dict[str, float]) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if "epoch" in logs:
            logs["epoch"] = int(logs["epoch"])
            if hasattr(self.compute_metrics, "current_epoch") and logs["epoch"] > self.compute_metrics.current_epoch:
                self.compute_metrics.current_epoch = logs["epoch"]
        output = {**logs, **{"step": self.state.global_step}}
        if getattr(self.args, "wandb_logging", False):
            wandb.log(output)
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )

    def training_step(
        self, model, inputs: Dict[str, Union[torch.Tensor, object]], num_items_in_batch: int = None
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        logits = outputs.get("logits")
        if (
            self.state.global_step % getattr(self.args, "eval_steps", 50) == 0
            and isinstance(logits, (list, tuple))
            and len(logits) >= 1
        ):
            signal_vectors = inputs["signal_vectors"]
            mask = outputs["mask"]
            pred_logits = logits[0]
            signal_vectors_reshaped = torch.reshape(signal_vectors, pred_logits.shape)
            plot_masked_pred_trends_one_sample(
                pred_logits=pred_logits.detach().cpu().numpy(),
                signal_vectors=signal_vectors_reshaped.cpu().numpy(),
                mask=mask.cpu().numpy(),
                sample_idx=0,
                node_idxs=[0, 100, 200],
                dataset_split="train",
                epoch=getattr(self.state, "epoch", 0),
            )
        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps

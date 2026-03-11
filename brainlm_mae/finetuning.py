"""BrainLM finetuning model with an MLP prediction head on top of the CLS token."""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from brainlm_mae.configuration_brainlm import BrainLMConfig
from brainlm_mae.modeling_brainlm import BrainLMModel, BrainLMPreTrainedModel, BrainLMPretrainingOutput

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Simple 3-layer MLP with dropout and ReLU activations."""

    def __init__(self, in_features: int, hidden_dim: int, out_features: int, dropout_rate: float):
        super().__init__()
        self.lin1 = nn.Linear(in_features, hidden_dim)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.lin3 = nn.Linear(hidden_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.drop1(self.lin1(x)))
        x = torch.relu(self.drop2(self.lin2(x)))
        return self.lin3(x)


class BrainLMForFinetuning(BrainLMPreTrainedModel):
    """Finetuning wrapper: frozen/unfrozen BrainLMModel backbone + MLP head on CLS token."""

    def __init__(self, config: BrainLMConfig):
        super().__init__(config)
        self.model = BrainLMModel(config)
        self.mlp_pred_head = MLP(
            in_features=config.hidden_size,
            hidden_dim=config.hidden_size // 2,
            out_features=1,
            dropout_rate=config.hidden_dropout_prob,
        )
        self.post_init()

    def _forward_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute regression loss between scalar predictions and labels.

        Args:
            logits: ``[B]``
            labels: ``[B]``
        """
        assert logits.shape == labels.shape
        if self.config.loss_fn == "mse":
            return ((logits - labels) ** 2).mean()
        if self.config.loss_fn == "mae":
            return torch.abs(logits - labels).mean()
        raise NotImplementedError(f"Unknown loss function: {self.config.loss_fn}")

    def forward(
        self,
        signal_vectors: torch.Tensor = None,
        xyz_vectors: torch.Tensor = None,
        labels: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BrainLMPretrainingOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states, mask = self.model(signal_vectors, xyz_vectors)

        cls_token = hidden_states[:, 0, :]  # [B, H]
        logits = self.mlp_pred_head(cls_token).squeeze(-1)  # [B]

        loss = self._forward_loss(logits, labels) if labels is not None else None

        if loss is not None and loss.item() > 5.0:
            logger.warning("Loss %.5f is unusually high, check batch", loss.item())

        if not return_dict:
            output = (logits, mask)
            return ((loss,) + output) if loss is not None else output

        return BrainLMPretrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
        )

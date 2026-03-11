from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BrainLMConfig(PretrainedConfig):
    r"""
    Configuration class for the decoder-only BrainLM model with block-causal attention.

    Args:
        hidden_size (`int`, *optional*, defaults to 256):
            Dimensionality of the transformer layers.
        num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of transformer layers.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads per layer.
        intermediate_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the feed-forward layer.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            Non-linear activation function.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            Dropout probability for fully connected layers.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            Dropout ratio for attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            Epsilon for layer normalization.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add bias to query, key, and value projections.
        num_brain_voxels (`int`, *optional*, defaults to 424):
            Number of brain voxels (ROIs / parcels) in the fMRI data.
        num_timepoints_per_voxel (`int`, *optional*, defaults to 200):
            Number of timepoints per voxel in one input sample.
        mask_ratio (`float`, *optional*, defaults to 0.75):
            Fraction of tokens to mask during random masking.
        timepoint_patching_size (`int`, *optional*, defaults to 20):
            Number of consecutive timepoints grouped into one patch token.
        loss_fn (`str`, *optional*, defaults to `"mse"`):
            Loss function for pretraining. One of `"mse"` or `"mae"`.
        forward_mask_prob (`float`, *optional*, defaults to 0.5):
            Probability of using forward masking (mask only the last temporal
            token per parcel) instead of random masking during training.

    Example:

    ```python
    >>> from brainlm_mae.configuration_brainlm import BrainLMConfig
    >>> from brainlm_mae.modeling_brainlm import BrainLMForPretraining

    >>> config = BrainLMConfig()
    >>> model = BrainLMForPretraining(config)
    >>> config = model.config
    ```"""

    model_type = "brainlm"
    auto_map = {
        "AutoConfig": "brainlm_mae.configuration_brainlm.BrainLMConfig",
        "AutoModelForPreTraining": "brainlm_mae.modeling_brainlm.BrainLMForPretraining",
    }

    def __init__(
        self,
        hidden_size=256,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=1024,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        qkv_bias=True,
        num_brain_voxels=424,
        num_timepoints_per_voxel=200,
        mask_ratio=0.75,
        timepoint_patching_size=20,
        loss_fn="mse",
        forward_mask_prob=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.num_brain_voxels = num_brain_voxels
        self.num_timepoints_per_voxel = num_timepoints_per_voxel
        self.mask_ratio = mask_ratio
        self.timepoint_patching_size = timepoint_patching_size
        self.loss_fn = loss_fn
        self.forward_mask_prob = forward_mask_prob

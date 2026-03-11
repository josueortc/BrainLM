"""Tests for BrainLM pretraining model: shapes, loss, gradients, save/load."""

import tempfile
import pytest
import torch

from brainlm_mae.configuration_brainlm import BrainLMConfig
from brainlm_mae.modeling_brainlm import BrainLMForPretraining, BrainLMModel


@pytest.fixture
def config():
    return BrainLMConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        num_brain_voxels=424,
        num_timepoints_per_voxel=200,
        timepoint_patching_size=20,
    )


@pytest.fixture
def batch(config):
    B, V, T = 2, config.num_brain_voxels, config.num_timepoints_per_voxel
    signal = torch.randn(B, V, T)
    xyz = torch.randn(B, V, 3)
    return signal, xyz


def test_forward_shapes(config, batch):
    model = BrainLMForPretraining(config)
    signal, xyz = batch
    out = model(signal_vectors=signal, xyz_vectors=xyz)
    assert out.loss is not None
    assert out.loss.dim() == 0
    logits, hidden_states = out.logits
    P = config.timepoint_patching_size
    num_tokens = config.num_timepoints_per_voxel // P
    assert logits.shape == (2, 424, num_tokens, P)
    assert hidden_states.shape[0] == 2
    assert hidden_states.shape[1] == 1 + num_tokens * 424
    assert hidden_states.shape[2] == config.hidden_size
    assert out.mask.shape == (2, 424, num_tokens)


def test_loss_finite(config, batch):
    model = BrainLMForPretraining(config)
    signal, xyz = batch
    out = model(signal_vectors=signal, xyz_vectors=xyz)
    assert torch.isfinite(out.loss).all()
    assert out.loss.item() >= 0


def test_backward(config, batch):
    model = BrainLMForPretraining(config)
    signal, xyz = batch
    out = model(signal_vectors=signal, xyz_vectors=xyz)
    out.loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Missing gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"


def test_from_pretrained_round_trip(config, batch):
    """Save and load model; verify weights are restored and loaded model runs."""
    model = BrainLMForPretraining(config)
    signal, xyz = batch
    state_before = {k: v.clone() for k, v in model.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        config_loaded = BrainLMConfig.from_pretrained(tmpdir)
        model_loaded = BrainLMForPretraining.from_pretrained(tmpdir, config=config_loaded)

    for key in state_before:
        assert key in model_loaded.state_dict(), f"Missing key {key} after load"
        assert torch.allclose(state_before[key], model_loaded.state_dict()[key]), (
            f"Weight mismatch for {key}"
        )

    model_loaded.eval()
    with torch.no_grad():
        out = model_loaded(signal_vectors=signal, xyz_vectors=xyz)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.logits[0].shape == (2, 424, 10, 20)


def test_brainlm_model_forward(config, batch):
    backbone = BrainLMModel(config)
    signal, xyz = batch
    hidden_states, mask = backbone(signal_vectors=signal, xyz_vectors=xyz)
    num_tokens = config.num_timepoints_per_voxel // config.timepoint_patching_size
    assert hidden_states.shape == (2, 1 + num_tokens * 424, config.hidden_size)
    assert mask.shape == (2, 424, num_tokens)

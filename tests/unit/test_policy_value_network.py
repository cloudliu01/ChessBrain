from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.chessbrain.domain.models.policy_value_network import (
    ACTION_SPACE_SIZE,
    AlphaZeroResidualNetwork,
    BOARD_CHANNELS,
)


def test_policy_value_network_shapes() -> None:
    model = AlphaZeroResidualNetwork(residual_blocks=2, channels=64)
    inputs = torch.zeros((3, BOARD_CHANNELS, 8, 8))

    output = model(inputs)

    assert output.policy_logits.shape == (3, model.action_space_planes, 8, 8)
    assert output.flatten_policy().shape == (3, ACTION_SPACE_SIZE)
    assert output.value.shape == (3, 1)


def test_inference_temperature_scaling() -> None:
    model = AlphaZeroResidualNetwork(residual_blocks=1, channels=32)
    inputs = torch.randn((1, BOARD_CHANNELS, 8, 8))

    result_default = model.inference(inputs)
    result_scaled = model.inference(inputs, temperature=0.5)

    assert result_default.policy_logits.shape == result_scaled.policy_logits.shape
    assert result_default.flatten_policy().shape == (1, ACTION_SPACE_SIZE)
    assert result_default.value.shape == result_scaled.value.shape
    assert not torch.allclose(
        result_default.flatten_policy(),
        result_scaled.flatten_policy(),
    )

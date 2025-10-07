"""Tests for the VibeCheck class."""

from unittest.mock import Mock, patch

import pytest

from vibechecks.config.config import VibeCheckConfig
from vibechecks.utils.logger import console_logger
from vibechecks.vibecheck import VibeCheck


@pytest.fixture
def mock_vibe_llm_client():
    """Fixture to mock the VibeLlmClient."""
    with patch("vibechecks.vibecheck.VibeLlmClient") as mock_client:
        yield mock_client


def test_vibecheck_init(mock_vibe_llm_client: Mock):
    """Test the __init__ method of VibeCheck."""
    mock_llm_instance = Mock()
    mock_vibe_llm_client.return_value = mock_llm_instance

    client = Mock()
    model = "test-model"
    config = VibeCheckConfig()

    vibe_check = VibeCheck(client, model, config=config)

    mock_vibe_llm_client.assert_called_once_with(client, model, config, console_logger)
    assert vibe_check.llm is mock_llm_instance


def test_vibecheck_call(mock_vibe_llm_client: Mock):
    """Test the __call__ method of VibeCheck."""
    mock_llm_instance = Mock()
    mock_llm_instance.vibe_eval.return_value = True
    mock_vibe_llm_client.return_value = mock_llm_instance

    vibe_check = VibeCheck(Mock(), "test-model")

    statement = "is the sky blue?"
    result = vibe_check(statement)

    mock_llm_instance.vibe_eval.assert_called_once_with(statement, bool)
    assert result is True


def test_vibecheck_call_false(mock_vibe_llm_client: Mock):
    """Test the __call__ method of VibeCheck when vibe_eval returns False."""
    mock_llm_instance = Mock()
    mock_llm_instance.vibe_eval.return_value = False
    mock_vibe_llm_client.return_value = mock_llm_instance

    vibe_check = VibeCheck(Mock(), "test-model")

    statement = "is the sky green?"
    result = vibe_check(statement)

    mock_llm_instance.vibe_eval.assert_called_once_with(statement, bool)
    assert result is False


def test_vibecheck_init_with_dict_config(mock_vibe_llm_client: Mock):
    """Test the __init__ method of VibeCheck with a dictionary config."""
    mock_llm_instance = Mock()
    mock_vibe_llm_client.return_value = mock_llm_instance

    client = Mock()
    model = "test-model"
    config_dict = {"system_instruction": "Custom instruction.", "timeout": 12345, "vibe_mode": "CHILL"}

    vibe_check = VibeCheck(client, model, config=config_dict)

    mock_vibe_llm_client.assert_called_once()
    args, _ = mock_vibe_llm_client.call_args
    passed_config = args[2]
    assert isinstance(passed_config, VibeCheckConfig)
    assert passed_config.system_instruction == "Custom instruction."
    assert passed_config.timeout == 12345
    assert passed_config.vibe_mode.name == "CHILL"
    assert vibe_check.llm is mock_llm_instance

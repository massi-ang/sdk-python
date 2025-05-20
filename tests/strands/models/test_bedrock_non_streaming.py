"""Tests for Bedrock model non-streaming mode."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from strands.models import BedrockModel
from strands.types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException,
)


@pytest.fixture
def mock_boto3_session():
    """Mock boto3 session."""
    with patch("boto3.Session") as mock_session:
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client
        yield mock_session, mock_client


def test_bedrock_non_streaming_config():
    """Test that the streaming flag is set correctly in the config."""
    # Test default value
    model = BedrockModel(model_id="test-model")
    assert model.config.get("streaming") is True

    # Test explicit value
    model = BedrockModel(model_id="test-model", streaming=False)
    assert model.config.get("streaming") is False


def test_bedrock_complete_response(mock_boto3_session):
    """Test the _complete_response method."""
    _, mock_client = mock_boto3_session
    mock_client.converse.return_value = {
        "output": {"message": {"content": "Test response"}},
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }

    # Create model and call _complete_response
    model = BedrockModel(model_id="test-model", streaming=False)
    response = model._complete_response({"modelId": "test-model"})

    # Verify API was called
    mock_client.converse.assert_called_once_with(modelId="test-model")
    assert response["output"]["message"]["content"] == "Test response"


def test_bedrock_complete_response_error_handling(mock_boto3_session):
    """Test error handling in the _complete_response method."""
    _, mock_client = mock_boto3_session

    # Test throttling exception
    mock_client.converse.side_effect = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        "Converse",
    )

    # Create model and call _complete_response
    model = BedrockModel(model_id="test-model", streaming=False)

    # Verify exception is raised
    with pytest.raises(ModelThrottledException):
        model._complete_response({"modelId": "test-model"})

    # Test context window overflow
    mock_client.converse.side_effect = ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "Input is too long for requested model",
            }
        },
        "Converse",
    )

    # Verify exception is raised
    with pytest.raises(ContextWindowOverflowException):
        model._complete_response({"modelId": "test-model"})


def test_bedrock_format_complete_response(mock_boto3_session):
    """Test the _format_complete_response method."""
    # Create model
    model = BedrockModel(model_id="test-model", streaming=False)

    # Test with text response
    response = {
        "output": {"message": {"content": "Test response"}},
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }

    # Get formatted chunks
    chunks = list(model._format_complete_response(response))

    # Verify chunks
    assert len(chunks) == 6  # message_start, content_start, content_delta, content_stop, message_stop, metadata
    assert chunks[0]["chunk_type"] == "message_start"
    assert chunks[1]["chunk_type"] == "content_start"
    assert chunks[2]["chunk_type"] == "content_delta"
    assert chunks[2]["data"] == "Test response"
    assert chunks[3]["chunk_type"] == "content_stop"
    assert chunks[4]["chunk_type"] == "message_stop"
    assert chunks[4]["data"] == "end_turn"
    assert chunks[5]["chunk_type"] == "metadata"
    assert chunks[5]["data"] == response["usage"]


def test_bedrock_format_complete_response_with_tool_calls(mock_boto3_session, monkeypatch):
    """Test the _format_complete_response method with tool calls."""
    # Mock uuid.uuid4 to return a predictable value
    monkeypatch.setattr(uuid, "uuid4", lambda: "mock-uuid")

    # Create model
    model = BedrockModel(model_id="test-model", streaming=False)

    # Test with tool call response
    response = {
        "output": {
            "message": {
                "content": "I'll help you with that.",
                "tool_calls": [
                    {
                        "name": "get_weather",
                        "id": "tool_1",
                        "arguments": {"location": "Seattle"},
                    }
                ],
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }

    # Get formatted chunks
    chunks = list(model._format_complete_response(response))

    # Verify chunks
    assert (
        len(chunks) == 8
    )  # message_start, content_start, content_delta, content_stop, tool_start, tool_stop, message_stop, metadata
    assert chunks[0]["chunk_type"] == "message_start"
    assert chunks[1]["chunk_type"] == "content_start"
    assert chunks[2]["chunk_type"] == "content_delta"
    assert chunks[2]["data"] == "I'll help you with that."
    assert chunks[3]["chunk_type"] == "content_stop"
    assert chunks[4]["chunk_type"] == "content_start"
    assert chunks[4]["data_type"] == "tool"
    assert chunks[4]["data"]["function"]["name"] == "get_weather"
    assert chunks[4]["data"]["function"]["arguments"] == {"location": "Seattle"}
    assert chunks[4]["data"]["id"] == "tool_1"
    assert chunks[5]["chunk_type"] == "content_stop"
    assert chunks[6]["chunk_type"] == "message_stop"
    assert chunks[6]["data"] == "tool_use"
    assert chunks[7]["chunk_type"] == "metadata"


def test_bedrock_stream_non_streaming_mode(mock_boto3_session):
    """Test the stream method in non-streaming mode."""
    _, mock_client = mock_boto3_session
    mock_client.converse.return_value = {
        "output": {"message": {"content": "Test response"}},
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
    }

    # Create model and call stream
    model = BedrockModel(model_id="test-model", streaming=False)
    chunks = list(model.stream({"modelId": "test-model"}))

    # Verify API was called
    mock_client.converse.assert_called_once_with(modelId="test-model")

    # Verify chunks
    assert len(chunks) == 6  # message_start, content_start, content_delta, content_stop, message_stop, metadata
    assert chunks[0]["chunk_type"] == "message_start"
    assert chunks[1]["chunk_type"] == "content_start"
    assert chunks[2]["chunk_type"] == "content_delta"
    assert chunks[2]["data"] == "Test response"
    assert chunks[3]["chunk_type"] == "content_stop"
    assert chunks[4]["chunk_type"] == "message_stop"
    assert chunks[4]["data"] == "end_turn"
    assert chunks[5]["chunk_type"] == "metadata"
    assert chunks[5]["data"] == mock_client.converse.return_value["usage"]


def test_bedrock_stream_non_streaming_mode_error_handling(mock_boto3_session):
    """Test error handling in the stream method in non-streaming mode."""
    _, mock_client = mock_boto3_session

    # Test throttling exception
    mock_client.converse.side_effect = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        "Converse",
    )

    # Create model and call stream
    model = BedrockModel(model_id="test-model", streaming=False)

    # Verify exception is raised
    with pytest.raises(ModelThrottledException):
        list(model.stream({"modelId": "test-model"}))

    # Test context window overflow
    mock_client.converse.side_effect = ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "Input is too long for requested model",
            }
        },
        "Converse",
    )

    # Verify exception is raised
    with pytest.raises(ContextWindowOverflowException):
        list(model.stream({"modelId": "test-model"}))

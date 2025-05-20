"""Integration tests for Bedrock non-streaming mode."""

import os

import pytest

from strands import Agent
from strands.models import BedrockModel


@pytest.mark.skipif(
    not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"),
    reason="AWS credentials not available",
)
def test_bedrock_non_streaming():
    """Test Bedrock model in non-streaming mode."""
    # Create model with streaming disabled
    model = BedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", streaming=False)

    # Create agent
    agent = Agent(model=model)

    # Test simple query
    result = agent("What is the capital of France?")

    # Verify result
    assert result.content is not None
    assert "Paris" in result.content


@pytest.mark.skipif(
    not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"),
    reason="AWS credentials not available",
)
def test_bedrock_non_streaming_with_tools():
    """Test Bedrock model in non-streaming mode with tools."""
    # Create model with streaming disabled
    model = BedrockModel(model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", streaming=False)

    # Create agent with a tool
    agent = Agent(model=model)

    # Define a simple tool
    @agent.tool
    def get_current_date():
        """Get the current date."""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d")

    # Test query that should use the tool
    result = agent("What is today's date?")

    # Verify result
    assert result.content is not None
    assert "date" in result.content.lower()
    assert len(result.tool_uses) > 0
    assert result.tool_uses[0].name == "get_current_date"

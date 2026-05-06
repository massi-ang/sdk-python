"""AgentCore Gateway plugin for Strands Agents.

Connects a Strands agent to an AWS AgentCore gateway over MCP: registers the
tools the gateway advertises and exposes a ``discover_tools`` tool for semantic
search-based tool discovery.

Example:
    ```python
    from strands import Agent
    from strands.vended_plugins.agentcore_gateway import (
        AgentCoreGateway,
        OAuthAuth,
    )

    plugin = AgentCoreGateway(
        gateway_url="https://agentcore.example.com/mcp",
        authentication=OAuthAuth(bearer_token_provider=lambda: "token"),
    )

    agent = Agent(plugins=[plugin])
    ```
"""

from .agentcore_gateway import (
    AgentCoreGateway,
    Authentication,
    IAMAuth,
    OAuthAuth,
)

__all__ = [
    "AgentCoreGateway",
    "Authentication",
    "IAMAuth",
    "OAuthAuth",
]

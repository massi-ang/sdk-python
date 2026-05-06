"""AgentCore Gateway plugin.

This module provides the AgentCoreGateway class that extends the Plugin base class
to connect a Strands agent to an AWS AgentCore gateway over MCP. On initialization
the plugin registers all tools the gateway advertises via ``list_tools`` (masking
the gateway's raw semantic search tool) and exposes a single ``discover_tools``
tool the agent can call to register additional tools returned by semantic search.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Union

from mcp.types import Tool as MCPTool

from ...plugins import Plugin
from ...tools.decorator import tool
from ...tools.mcp import MCPClient
from ...tools.mcp.mcp_agent_tool import MCPAgentTool
from ...types.tools import ToolContext, ToolSpec

if TYPE_CHECKING:
    from ...agent.agent import Agent
    from ...tools.mcp.mcp_types import MCPTransport

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOOLS = 5
_DISCOVERY_TOOL_NAME = "x_amz_bedrock_agentcore_search"
_DEFAULT_REGION = "us-east-1"
_DEFAULT_DISCOVER_TOOLS_DESCRIPTION = (
    "Use this tool to retrieve a list of additional tools provided by this server "
    "that match the provided query"
)


@dataclass
class IAMAuth:
    """IAM/SigV4 authentication for AgentCore gateway.

    Uses AWS credentials from the environment (environment variables, credentials file,
    or IAM role) to sign requests with SigV4. Requires the ``mcp-proxy-for-aws``
    library to be installed.

    Attributes:
        type: Discriminator field, must be ``"IAM"``.
        region: AWS region where the AgentCore gateway is deployed.
            Defaults to ``us-east-1``.
    """

    type: Literal["IAM"] = "IAM"
    region: str = _DEFAULT_REGION


@dataclass
class OAuthAuth:
    """OAuth/Bearer token authentication for AgentCore gateway.

    Uses a bearer token provided by a callable. The callable is invoked each time
    a new transport is created, allowing for token refresh.

    Attributes:
        type: Discriminator field, must be ``"OAUTH"``.
        bearer_token_provider: Callable that returns the current bearer token.
            Called each time a new connection is established to support token refresh.
    """

    bearer_token_provider: Callable[[], str]
    type: Literal["OAUTH"] = "OAUTH"


Authentication = Union[IAMAuth, OAuthAuth]
"""Authentication configuration for AgentCore gateway.

Either ``IAMAuth`` for SigV4/IAM authentication or ``OAuthAuth`` for bearer token.
"""


class AgentCoreGateway(Plugin):
    """Plugin that connects a Strands agent to an AWS AgentCore gateway over MCP.

    On initialization the plugin:

    1. Opens a streamable HTTP MCP session to the gateway
    2. Paginates through ``list_tools`` and registers every advertised tool with
       the agent, except ``x_amz_bedrock_agentcore_search``, which is masked
    3. Exposes a single ``discover_tools`` tool that wraps the gateway's semantic
       search. When the agent calls it, the plugin invokes the gateway search and
       registers any matching tools directly from the search response — this
       covers tools the gateway does not enumerate in ``list_tools``

    Registered tools execute against the same MCP session the plugin opened to the
    gateway, so there is a single authenticated connection per agent.

    Authentication is configured via the ``authentication`` parameter, which accepts
    either ``IAMAuth`` (SigV4) or ``OAuthAuth`` (bearer token).

    Example with OAuth/Bearer Token:
        ```python
        from strands import Agent
        from strands.vended_plugins.agentcore_gateway import (
            AgentCoreGateway,
            OAuthAuth,
        )

        def get_bearer_token() -> str:
            return os.environ["AGENTCORE_TOKEN"]

        plugin = AgentCoreGateway(
            gateway_url="https://agentcore.example.com/mcp",
            authentication=OAuthAuth(bearer_token_provider=get_bearer_token),
        )

        agent = Agent(plugins=[plugin])
        ```

    Example with IAM/SigV4:
        ```python
        from strands import Agent
        from strands.vended_plugins.agentcore_gateway import (
            AgentCoreGateway,
            IAMAuth,
        )

        plugin = AgentCoreGateway(
            gateway_url="https://agentcore.us-east-1.amazonaws.com/mcp",
            authentication=IAMAuth(region="us-east-1"),
        )

        agent = Agent(plugins=[plugin])
        ```
    """

    name = "agentcore_gateway"

    def __init__(
        self,
        gateway_url: str,
        authentication: Authentication,
        max_tools: int = _DEFAULT_MAX_TOOLS,
        startup_timeout: int = 30,
        discover_tools_description: str | None = None,
    ) -> None:
        """Initialize the AgentCoreGateway plugin.

        Args:
            gateway_url: URL of the AgentCore gateway MCP endpoint.
            authentication: Authentication configuration. Either ``IAMAuth`` for
                SigV4/IAM auth or ``OAuthAuth`` for bearer token auth.
            max_tools: Maximum number of tools to register from a single discovery
                call. The gateway returns all matching tools; this cap is applied
                client-side to limit context impact.
            startup_timeout: Timeout in seconds for MCP client initialization.
            discover_tools_description: Optional description for the ``discover_tools``
                tool, shown to the model in the tool spec. Defaults to a generic
                description suitable for most AgentCore gateways.

        Raises:
            TypeError: If ``authentication`` is not an ``IAMAuth`` or ``OAuthAuth`` instance.
        """
        if not isinstance(authentication, (IAMAuth, OAuthAuth)):
            raise TypeError(
                f"authentication must be IAMAuth or OAuthAuth, got {type(authentication).__name__}"
            )

        self._gateway_url = gateway_url
        self._authentication = authentication
        self._max_tools = max_tools
        self._startup_timeout = startup_timeout
        self._discover_tools_description = (
            discover_tools_description or _DEFAULT_DISCOVER_TOOLS_DESCRIPTION
        )

        self._mcp_client: MCPClient | None = None

        super().__init__()

    def _build_transport_factory(self) -> Callable[[], MCPTransport]:
        """Build the MCP transport factory based on authentication config.

        Returns:
            A callable that returns an MCP transport.

        Raises:
            ImportError: If required auth library is not installed.
        """
        if isinstance(self._authentication, IAMAuth):
            return self._build_iam_transport_factory(self._authentication)
        return self._build_oauth_transport_factory(self._authentication)

    def _build_iam_transport_factory(self, auth: IAMAuth) -> Callable[[], MCPTransport]:
        """Build a transport factory using SigV4 via mcp-proxy-for-aws.

        Args:
            auth: IAM authentication configuration.

        Returns:
            Transport factory callable.

        Raises:
            ImportError: If ``mcp-proxy-for-aws`` is not installed.
        """
        try:
            from mcp_proxy_for_aws import create_streamable_http_transport
        except ImportError as e:
            raise ImportError(
                "IAM authentication requires the 'mcp-proxy-for-aws' library. "
                "Install it with: pip install mcp-proxy-for-aws"
            ) from e

        gateway_url = self._gateway_url
        region = auth.region

        def factory() -> MCPTransport:
            logger.debug("region=<%s> | creating SigV4 transport", region)
            return create_streamable_http_transport(url=gateway_url, region=region)

        return factory

    def _build_oauth_transport_factory(
        self, auth: OAuthAuth
    ) -> Callable[[], MCPTransport]:
        """Build a transport factory using streamable HTTP with bearer token.

        Args:
            auth: OAuth authentication configuration.

        Returns:
            Transport factory callable.
        """
        from mcp.client.streamable_http import streamablehttp_client

        gateway_url = self._gateway_url
        token_provider = auth.bearer_token_provider

        def factory() -> MCPTransport:
            token = token_provider()
            headers = {"Authorization": f"Bearer {token}"}
            logger.debug("creating streamable HTTP transport with bearer token")
            return streamablehttp_client(url=gateway_url, headers=headers)

        return factory

    def init_agent(self, agent: Agent) -> None:
        """Initialize the plugin with an agent instance.

        Starts the MCP connection to the gateway, enumerates all tools the gateway
        exposes via ``list_tools``, and registers them with the agent — except the
        raw search tool (``x_amz_bedrock_agentcore_search``), which is masked since
        ``discover_tools`` provides the agent-facing interface for semantic search.

        Args:
            agent: The agent instance to extend with dynamic tool discovery.

        Raises:
            RuntimeError: If the MCP client fails to initialize or the gateway
                tool listing fails.
        """
        logger.debug(
            "gateway_url=<%s>, auth_type=<%s> | initializing dynamic tool discovery plugin",
            self._gateway_url,
            self._authentication.type,
        )

        self._apply_discover_tools_description()

        transport_factory = self._build_transport_factory()
        self._mcp_client = MCPClient(
            transport_factory, startup_timeout=self._startup_timeout
        )

        try:
            self._mcp_client.start()
            self._register_gateway_tools(agent)
            logger.debug("dynamic tool discovery plugin initialized successfully")
        except Exception as e:
            logger.error("error=<%s> | failed to initialize MCP client", e)
            raise RuntimeError(
                f"Failed to initialize dynamic tool discovery: {e}"
            ) from e

    def _register_gateway_tools(self, agent: Agent) -> None:
        """Enumerate gateway tools and register everything except the search tool.

        Paginates through all pages of ``list_tools`` to collect the full set of
        tools the gateway chooses to advertise, then registers each with the agent.
        The raw search tool is masked — the agent only sees ``discover_tools``, which
        wraps it. Logs a warning if the search tool is absent, since discovery will
        not work in that case.

        Args:
            agent: The agent instance to register tools with.

        Raises:
            RuntimeError: If the gateway tool listing fails.
        """
        if self._mcp_client is None:
            return

        try:
            pagination_token: str | None = None
            page_count = 0
            registered_count = 0
            skipped_search_tool = False

            while True:
                page = self._mcp_client.list_tools_sync(
                    pagination_token=pagination_token
                )
                for gateway_tool in page:
                    was_registered, was_search = self._register_single_gateway_tool(
                        agent, gateway_tool
                    )
                    if was_registered:
                        registered_count += 1
                    if was_search:
                        skipped_search_tool = True

                pagination_token = page.pagination_token
                page_count += 1
                if pagination_token is None:
                    break

            if not skipped_search_tool:
                logger.warning(
                    "tool_name=<%s> | search tool not found in gateway list_tools; "
                    "removing discover_tools from the agent (dynamic discovery unavailable)",
                    _DISCOVERY_TOOL_NAME,
                )
                self._remove_discover_tools_from_plugin()

            logger.info(
                "registered_count=<%d>, pages=<%d>, search_tool_masked=<%s> | gateway tools registered",
                registered_count,
                page_count,
                skipped_search_tool,
            )
        except Exception as e:
            logger.error("error=<%s> | failed to register gateway tools", e)
            raise RuntimeError(f"Failed to register gateway tools: {e}") from e

    @staticmethod
    def _register_single_gateway_tool(
        agent: Agent, gateway_tool: Any
    ) -> tuple[bool, bool]:
        """Register one gateway tool, masking the search tool and skipping duplicates.

        Args:
            agent: The agent instance.
            gateway_tool: The tool returned by ``list_tools_sync``.

        Returns:
            Tuple of ``(was_registered, was_search_tool)``. ``was_registered`` is
            True only when the tool was added to the agent's registry;
            ``was_search_tool`` is True when the tool was the masked search tool.
        """
        tool_name = gateway_tool.tool_name

        if tool_name == _DISCOVERY_TOOL_NAME:
            logger.debug(
                "tool_name=<%s> | masking gateway search tool (wrapped by discover_tools)",
                tool_name,
            )
            return False, True

        if tool_name in agent.tool_registry.registry:
            logger.debug("tool_name=<%s> | already registered, skipping", tool_name)
            return False, False

        try:
            agent.tool_registry.register_tool(gateway_tool)
            logger.debug("tool_name=<%s> | registered gateway tool", tool_name)
            return True, False
        except Exception as e:
            logger.warning(
                "tool_name=<%s>, error=<%s> | failed to register gateway tool",
                tool_name,
                e,
            )
            return False, False

    def _remove_discover_tools_from_plugin(self) -> None:
        """Remove the ``discover_tools`` tool from the plugin's auto-registered tools.

        Called when the gateway does not advertise the semantic search tool in
        ``list_tools``. Mutating ``self._tools`` before the plugin registry reads
        it prevents ``discover_tools`` from being wired into the agent's tool
        registry, so the model never sees a tool it cannot use.
        """
        self._tools = [t for t in self._tools if t.tool_name != "discover_tools"]
        logger.debug("discover_tools removed from plugin tools")

    def _apply_discover_tools_description(self) -> None:
        """Override the ``discover_tools`` tool description with the configured value.

        The ``@tool`` decorator derives the description from the method docstring.
        To let callers customize what the model sees, mutate the tool spec on the
        plugin's auto-discovered tool instance before it is registered with the agent.
        """
        for plugin_tool in self.tools:
            if plugin_tool.tool_name == "discover_tools":
                spec: ToolSpec = {
                    **plugin_tool.tool_spec,
                    "description": self._discover_tools_description,
                }
                plugin_tool.tool_spec = spec
                break

    @tool(context=True)
    def discover_tools(
        self, query: str, tool_context: ToolContext
    ) -> str:  # noqa: D417
        """Discover and register tools relevant to a task via semantic search.

        Use this tool to find and register tools that might help with a specific
        task. Searches the AgentCore gateway's tool catalog by semantic similarity.

        Args:
            query: Description of the task or capability needed.
        """
        if not query or not query.strip():
            return "Error: query parameter is required and cannot be empty."

        agent = tool_context.agent
        logger.debug("query=<%s> | discovering tools", query)

        discovered = self._search_tools(query)
        if discovered is None:
            return "Error: tool discovery failed. Check plugin logs for details."

        if not discovered:
            return f"No relevant tools found for query: {query}"

        registered_count, tool_details = self._register_discovered_tools(
            agent, discovered
        )

        if tool_details:
            tool_list = "\n".join(
                f"  - {t['name']}: {t['description']}" for t in tool_details
            )
        else:
            tool_list = "  (no new tools registered)"

        return (
            f"Successfully discovered and registered {registered_count} tool(s):\n\n"
            f"{tool_list}\n\n"
            f"These tools are now available for use."
        )

    def _search_tools(self, query: str) -> list[dict[str, Any]] | None:
        """Call the gateway search tool and parse the response.

        Args:
            query: Search query.

        Returns:
            List of tool data dicts (capped at ``max_tools``), empty list if no
            results, or None if the search call failed.
        """
        if self._mcp_client is None:
            return None

        try:
            result = self._mcp_client.call_tool_sync(
                tool_use_id=f"discovery_{id(query)}",
                name=_DISCOVERY_TOOL_NAME,
                arguments={"query": query},
            )
        except Exception as e:
            logger.warning("error=<%s> | search tool call failed", e)
            return None

        if result["status"] != "success":
            logger.debug("search tool returned non-success status")
            return None

        parsed = self._parse_search_response(result)
        # The gateway returns all matches; cap at max_tools to limit context impact.
        return parsed[: self._max_tools]

    @staticmethod
    def _parse_search_response(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse the search tool response into a list of tool data dicts.

        Handles the AgentCore gateway response format ``{"tools": [...]}``, which
        can arrive either in ``structuredContent`` or as JSON text in ``content``.

        Args:
            result: Tool result dict from MCP call.

        Returns:
            List of tool data dicts, empty if parsing fails or no tools found.
        """
        tools_json: Any = None

        if "structuredContent" in result:
            tools_json = result["structuredContent"]
        elif result.get("content"):
            for block in result["content"]:
                if "text" in block:
                    try:
                        tools_json = json.loads(block["text"])
                        break
                    except json.JSONDecodeError:
                        logger.warning("failed to parse tool search response as JSON")

        if isinstance(tools_json, dict) and "tools" in tools_json:
            return list(tools_json["tools"])
        if isinstance(tools_json, list):
            return tools_json
        return []

    def _register_discovered_tools(
        self, agent: Agent, discovered: list[dict[str, Any]]
    ) -> tuple[int, list[dict[str, str]]]:
        """Register discovered tools in the agent's tool registry.

        Builds an ``MCPAgentTool`` directly from each search result, wrapping the
        gateway MCP client so the tool invokes the gateway when the agent calls it.
        The search response is trusted — no cross-check against ``list_tools`` is
        performed since the gateway may not enumerate the full catalog there.

        Args:
            agent: The agent instance.
            discovered: List of tool data dicts from the search response.

        Returns:
            Tuple of (count_registered, list_of_tool_details).
        """
        if self._mcp_client is None:
            return 0, []

        registered_count = 0
        tool_details: list[dict[str, str]] = []

        for tool_data in discovered:
            tool_name = tool_data.get("name")
            if not tool_name:
                logger.warning("tool_data=<%s> | missing name field", tool_data)
                continue

            if tool_name in agent.tool_registry.registry:
                logger.debug("tool_name=<%s> | already registered, skipping", tool_name)
                continue

            try:
                agent_tool = self._build_mcp_agent_tool(tool_data)
                agent.tool_registry.register_tool(agent_tool)
                registered_count += 1
                tool_details.append(
                    {
                        "name": tool_name,
                        "description": tool_data.get("description", "No description"),
                    }
                )
                logger.debug("tool_name=<%s> | tool registered", tool_name)
            except Exception as e:
                logger.warning(
                    "tool_name=<%s>, error=<%s> | failed to register tool", tool_name, e
                )

        return registered_count, tool_details

    def _build_mcp_agent_tool(self, tool_data: dict[str, Any]) -> MCPAgentTool:
        """Build an MCP-backed AgentTool from a search result entry.

        Constructs an ``mcp.types.Tool`` from the search response data and wraps it
        in an ``MCPAgentTool`` pointed at the gateway MCP client. When the agent
        calls the tool, execution is routed through the same MCP session used by
        the search tool.

        Args:
            tool_data: Dict containing ``name``, ``description``, and ``inputSchema``
                from the search response.

        Returns:
            An ``MCPAgentTool`` ready to register with the agent.

        Raises:
            AssertionError: If ``_mcp_client`` is None (guarded by the caller).
        """
        assert self._mcp_client is not None

        mcp_tool = MCPTool(
            name=tool_data["name"],
            description=tool_data.get("description", ""),
            inputSchema=tool_data.get(
                "inputSchema", {"type": "object", "properties": {}}
            ),
        )
        return MCPAgentTool(mcp_tool=mcp_tool, mcp_client=self._mcp_client)

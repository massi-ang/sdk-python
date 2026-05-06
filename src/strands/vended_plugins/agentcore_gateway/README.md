# AgentCore Gateway Plugin

Connects Strands agents to an AWS AgentCore gateway over the MCP protocol (streamable HTTP). The plugin registers the tools the gateway advertises and exposes a `discover_tools` tool backed by the gateway's semantic search for on-demand tool discovery.

## Features

- Registers all tools the gateway advertises via `list_tools` at initialization, except the raw search tool, which is masked
- Exposes a single `discover_tools` tool the agent can call to register additional tools returned by semantic search
- Tools returned by the gateway's semantic search are trusted and registered directly — no additional `list_tools` round-trip
- Registered tools execute against the same gateway MCP client when invoked
- Structured authentication: `IAMAuth` (SigV4) or `OAuthAuth` (bearer token)

## Authentication

The plugin accepts a structured `authentication` parameter. Two types are supported:

### OAuth (Bearer Token)

```python
from strands import Agent
from strands.vended_plugins.agentcore_gateway import (
    AgentCoreGateway,
    OAuthAuth,
)

def get_token() -> str:
    return os.environ["AGENTCORE_TOKEN"]

plugin = AgentCoreGateway(
    gateway_url="https://agentcore.example.com/mcp",
    authentication=OAuthAuth(bearer_token_provider=get_token),
)

agent = Agent(plugins=[plugin])
```

The `bearer_token_provider` callable is invoked each time a new transport is created, so you can implement token refresh logic inside it.

### IAM (SigV4)

Requires the `mcp-proxy-for-aws` library:

```bash
pip install mcp-proxy-for-aws
```

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

AWS credentials are resolved from the standard chain: environment variables, credentials file, or IAM role.

## Parameters

| Parameter                    | Type                   | Default         | Description                                                    |
| ---------------------------- | ---------------------- | --------------- | -------------------------------------------------------------- |
| `gateway_url`                | `str`                  | required        | AgentCore gateway MCP endpoint URL                             |
| `authentication`             | `IAMAuth \| OAuthAuth` | required        | Authentication configuration                                   |
| `max_tools`                  | `int`                  | `5`             | Max tools to register per discovery call (applied client-side) |
| `startup_timeout`            | `int`                  | `30`            | MCP client startup timeout (seconds)                           |
| `discover_tools_description` | `str \| None`          | generic default | Description shown to the model for the `discover_tools` tool   |

### Customizing the discover_tools description

By default, the `discover_tools` tool is described as:

> Use this tool to retrieve a list of additional tools provided by this server that match the provided query

Override this when you want to steer the model more specifically about when to discover tools:

```python
plugin = AgentCoreGateway(
    gateway_url="https://agentcore.example.com/mcp",
    authentication=OAuthAuth(bearer_token_provider=get_token),
    discover_tools_description=(
        "Search for finance, trading, or market data tools when the user asks about "
        "pricing, orders, or portfolio analysis. Call this before attempting an answer."
    ),
)
```

## How It Works

1. On initialization, the plugin opens an MCP session to the gateway and paginates through `list_tools` to enumerate every tool the gateway advertises. Each of those is registered with the agent, except `x_amz_bedrock_agentcore_search`, which is masked because `discover_tools` wraps it.
2. If `list_tools` does not include `x_amz_bedrock_agentcore_search`, the plugin logs a warning and removes `discover_tools` from the agent — the model will not see a tool it cannot use. Any other advertised tools are still registered normally.
3. When the agent calls `discover_tools(query)`, the plugin invokes the gateway's semantic search tool and receives a list of matching tool specs (name, description, input schema).
4. For each match, the plugin constructs an MCP-backed `AgentTool` pointed at the shared gateway client and registers it with the agent's tool registry. Tools already in the registry are skipped. This covers the case where the gateway's `list_tools` does not enumerate the full catalog — search can still surface tools the agent can use.
5. The agent can then call those tools directly; invocation is routed through the same MCP session that served the search.

## Gateway Requirements

The AgentCore gateway must:

1. Expose an MCP server over streamable HTTP
2. Include the `x_amz_bedrock_agentcore_search` tool for semantic search
3. Accept `tools/call` requests for any tool name returned by the search — even if that tool is not enumerated by `list_tools`

### Expected Search Tool Response

The `x_amz_bedrock_agentcore_search` tool should return a JSON payload in this format:

```json
{
  "tools": [
    {
      "name": "LambdaUsingSDK___farming_weather",
      "description": "Get agricultural weather data including soil conditions and frost risk",
      "inputSchema": {
        "type": "object",
        "properties": {
          "place": { "type": "string" }
        },
        "required": ["place"]
      }
    }
  ]
}
```

The response can be delivered either in `structuredContent` or as JSON text in `content`. Both are parsed transparently.

## Usage

```python
plugin = AgentCoreGateway(
    gateway_url="https://agentcore.example.com/mcp",
    authentication=OAuthAuth(bearer_token_provider=get_token),
)

agent = Agent(plugins=[plugin])
agent("Use discover_tools to find weather utilities, then check the forecast for Seattle")
```

## Known Warnings

### "Tool `<name>` not listed by server, cannot validate any structured content"

You may see this warning from the underlying MCP client when the agent invokes a tool that was discovered via `discover_tools` rather than through `list_tools`:

```
Tool LambdaUsingSDK___current_weather not listed by server, cannot validate any structured content
```

This is expected and safe to ignore.

**Why it happens:** The MCP client caches the tool catalog returned by `list_tools` and uses it to validate structured content returned by tool calls. AgentCore gateways typically do not enumerate their full tool catalog in `list_tools` — the complete set is only accessible through semantic search. When the agent calls a tool that was registered on the fly from a search response, the client notices the name isn't in its cache and logs this warning before falling back to unvalidated output.

**Impact:** None on the plugin's behavior. The tool call executes normally and the result is returned to the agent. Only the client-side structured-content validation step is skipped.

**If you want to silence it,** filter the MCP client logger in your application:

```python
import logging
logging.getLogger("mcp.client").setLevel(logging.ERROR)
```

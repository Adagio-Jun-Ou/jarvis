---
name: mcp-client
description: Model Context Protocol client for connecting to MCP servers and using their tools seamlessly
author: jarvis
version: 1.0.0
tags: [mcp, protocol, tools, integration]
metadata:
  jarvis:
    requires:
      bins: [npx]
      env: []
      config: []
    always: false
    user-invocable: true
    emoji: "🔌"
---

# MCP Client Skill

Connect to MCP (Model Context Protocol) servers and seamlessly use their tools, resources, and prompts.

## What is MCP?

MCP (Model Context Protocol) is an open protocol that enables AI assistants to connect to external tools and data sources through a standardized interface. MCP servers can provide:

- **Tools**: Functions the AI can call
- **Resources**: Data the AI can read
- **Prompts**: Templates for common tasks

## Connecting to MCP Servers

### Stdio Transport
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
    }
  }
}
```

### HTTP Transport
```json
{
  "mcpServers": {
    "my-server": {
      "url": "http://localhost:3000/mcp"
    }
  }
}
```

## Popular MCP Servers

### Filesystem
```bash
npx -y @modelcontextprotocol/server-filesystem /home/user/projects
```
Tools: `read_file`, `write_file`, `list_directory`, `search_files`, `get_file_info`

### PostgreSQL
```bash
npx -y @modelcontextprotocol/server-postgres postgresql://user:pass@localhost/db
```
Tools: `query`, `list_tables`, `describe_table`, `execute`

### GitHub
```bash
npx -y @modelcontextprotocol/server-github
# Requires GITHUB_TOKEN env var
```
Tools: `create_issue`, `create_pr`, `search_repos`, `get_file_contents`, `push_files`

### Puppeteer
```bash
npx -y @modelcontextprotocol/server-puppeteer
```
Tools: `navigate`, `screenshot`, `click`, `type`, `evaluate`

### Memory
```bash
npx -y @modelcontextprotocol/server-memory
```
Tools: `create_entities`, `create_relations`, `add_observations`, `search_nodes`

### Brave Search
```bash
npx -y @modelcontextprotocol/server-brave-search
# Requires BRAVE_API_KEY env var
```
Tools: `brave_search`, `brave_local_search`

### Slack
```bash
npx -y @modelcontextprotocol/server-slack
# Requires SLACK_BOT_TOKEN env var
```
Tools: `send_message`, `list_channels`, `get_channel_history`

## Using MCP Tools

When an MCP server is connected:

1. Tools are automatically discovered
2. Call tools using standard function calling:
```
Use the filesystem tool to read /projects/myapp/src/main.py
```

3. Resources are accessible via URI:
```
Read the resource at file:///projects/myapp/src/main.py
```

## Discovery Commands

List available MCP servers:
```
What MCP servers are available?
```

List tools from a server:
```
What tools does the github MCP server provide?
```

## Configuration Location

MCP servers are typically configured in:
- `~/.jarvis/mcp_servers.json`
- Or workspace-specific `.jarvis/mcp_servers.json`

## Example Session

```
User: Connect to the filesystem server for /home/user/projects

Assistant: Connecting to filesystem MCP server at /home/user/projects...
[Connects via npx @modelcontextprotocol/server-filesystem]

Available tools:
- read_file: Read file contents
- write_file: Write to file
- list_directory: List directory contents
- search_files: Search for files
- get_file_info: Get file metadata

User: List the directory contents

Assistant: [Uses list_directory tool]
Found 5 items:
- src/ (directory)
- tests/ (directory)
- pyproject.toml
- README.md
- .gitignore

User: Read the README and summarize it

Assistant: [Uses read_file tool]
The README shows this is a Python project called...
```

## Best Practices

1. **Security**: Only connect to trusted MCP servers
2. **Sandboxing**: Use filesystem servers with restricted paths
3. **Authentication**: Store API keys securely via environment variables
4. **Error Handling**: MCP servers may fail - handle gracefully
5. **Resource Limits**: Some servers may have rate limits
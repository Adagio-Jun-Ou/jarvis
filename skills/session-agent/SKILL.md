---
name: session-agent
description: Multi-agent coordination for parallel task execution. Spawn sub-agents, coordinate workflows, and aggregate results.
author: jarvis
version: 1.0.0
tags: [agent, multi-agent, coordination, parallel, workflow]
metadata:
  jarvis:
    requires:
      bins: []
      env: []
      config: []
    always: false
    user-invocable: true
    emoji: "🤖"
---

# Session Agent Skill

Enable multi-agent coordination by spawning sub-agents for parallel task execution, delegating work, and aggregating results.

## Capabilities

### 1. Spawn Sub-Agents

Create specialized sub-agents for different tasks:

```
Spawn a research agent to investigate X while I work on Y
```

```
Create 3 parallel agents:
- One to analyze the codebase
- One to search for documentation
- One to check for similar issues
```

### 2. Agent Communication

Agents can communicate through:

- **Shared Context**: Common state accessible to all agents
- **Message Passing**: Direct agent-to-agent messages
- **Result Channels**: Dedicate channels for results

### 3. Coordination Patterns

#### Map-Reduce Pattern
```
Map work across N agents → Reduce results into final output
```

#### Pipeline Pattern
```
Agent A → Agent B → Agent C → Final Result
```

#### Fan-out/Fan-in Pattern
```
        ┌→ Agent A ─┐
Main ───┼→ Agent B ─┼→ Aggregator → Result
        └→ Agent C ─┘
```

## Usage Examples

### Parallel Research
```
"I need to understand the authentication system. Spawn agents to:
1. Analyze the auth middleware code
2. Search for auth-related documentation
3. Check test coverage for auth
Then aggregate findings into a summary."
```

### Code Review Pipeline
```
"Review this PR using a pipeline:
1. SecurityAgent: Check for vulnerabilities
2. StyleAgent: Enforce code style
3. TestAgent: Verify test coverage
Generate a combined review report."
```

### Distributed Task Execution
```
"Deploy this change across all 5 environments in parallel,
then verify each deployment and report results."
```

## Implementation Notes

When using session-agent:

1. **Define Clear Boundaries**: Each agent should have a well-defined scope
2. **Avoid Conflicts**: Use namespacing for shared resources
3. **Set Timeouts**: Sub-agents should have reasonable timeouts
4. **Aggregate Wisely**: Combine results intelligently, not just concatenate

## Sub-agent Types

| Type | Purpose |
|------|---------|
| `researcher` | Search, fetch, and analyze information |
| `coder` | Write, edit, and refactor code |
| `reviewer` | Analyze quality, security, performance |
| `tester` | Generate and run tests |
| `deployer` | Handle deployment and infrastructure |
| `coordinator` | Manage other agents |

## Best Practices

- Start with 2-3 agents for simple parallel work
- Use pipelines for dependent tasks
- Always define success criteria before spawning
- Aggregate in the main session, not in sub-agents
- Clean up agents after task completion
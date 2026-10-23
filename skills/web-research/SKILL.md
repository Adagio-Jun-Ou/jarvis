---
name: web-research
description: Comprehensive web research and synthesis capability
author: jarvis
version: 1.0.0
tags: [web, research, synthesis, search]
metadata:
  jarvis:
    requires:
      bins: []
      env: []
      config: []
    always: false
    user-invocable: true
    emoji: "🔍"
---

# Web Research Skill

You have the ability to perform comprehensive web research. This skill enables you to:

1. **Search the web** for information on any topic
2. **Extract and synthesize** key information from multiple sources
3. **Provide citations** for all claims
4. **Compare and contrast** different viewpoints

## How to Use

When the user asks for research:

1. Start by searching for the main topic using `web_search`
2. Identify 3-5 high-quality sources from the results
3. Use `web_fetch` to get detailed content from promising sources
4. Synthesize the information into a clear, structured response
5. Include citations in the format: [Source Name](URL)

## Best Practices

- Prioritize authoritative sources (official docs, academic papers, established news)
- Cross-reference claims across multiple sources
- Note when information is outdated or disputed
- Present multiple viewpoints when topics are controversial

## Example Workflow

```
User: Research the latest developments in quantum computing

1. web_search("quantum computing latest developments 2024 2025")
2. web_fetch(top 3 results)
3. Synthesize findings into structured summary
4. Present with citations
```

## Output Format

Structure your research output as:

```markdown
# Research Summary: [Topic]

## Key Findings
- Finding 1 with citation
- Finding 2 with citation

## Detailed Analysis
[Detailed synthesis]

## Sources
1. [Source Name](URL) - Key insight from this source
2. [Source Name](URL) - Key insight from this source
```
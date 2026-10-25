---
name: data-pipeline
description: Build and execute composable data processing pipelines
author: jarvis
version: 1.0.0
tags: [data, etl, pipeline, transform]
metadata:
  jarvis:
    requires:
      bins: [python]
      env: []
      config: []
    always: false
    user-invocable: true
    emoji: "📊"
---

# Data Pipeline Skill

You can build and execute composable data processing pipelines using the Lobster-style syntax.

## Pipeline Composition

Use the `|` operator to chain stages:

```python
from core.workflow_shell import Pipeline, stage, parallel, retry, timeout

# Sequential pipeline
pipeline = (
    Pipeline("process_data")
    | stage(extract_data)
    | stage(transform_data)
    | stage(load_data)
)

# With parallel execution
pipeline = (
    Pipeline("parallel_process")
    | stage(extract)
    || stage(process_a)  # Run in parallel
    || stage(process_b)
    | stage(combine)
)

# With error handling
pipeline = (
    Pipeline("robust_pipeline")
    | retry(stage(fetch_data), max_retries=3)
    | timeout(stage(process), seconds=30)
    | fallback(stage(primary_api), stage(backup_api))
)
```

## Stage Functions

A stage is any async function that takes input data and a context:

```python
@stage
async def extract_data(input_data, context):
    # Your extraction logic
    return extracted_data
```

## Built-in Stages

| Stage | Purpose |
|-------|---------|
| `retry(stage, max_retries)` | Retry on failure |
| `timeout(stage, seconds)` | Timeout after specified seconds |
| `fallback(primary, backup)` | Try backup if primary fails |
| `when(condition, if_stage, else_stage)` | Conditional execution |
| `parallel(*stages)` | Run stages in parallel |

## Context Object

The `PipelineContext` provides state sharing:

```python
context.set("key", value)  # Store value
value = context.get("key")  # Retrieve value
result = context.get_stage_result("stage_name")  # Get previous stage result
```

## Typical Pipeline Patterns

### ETL Pipeline
```
Extract → Transform → Validate → Load
```

### Map-Reduce
```
Extract → [Process A | Process B | Process C] (parallel) → Reduce
```

### Fan-out/Fan-in
```
Fetch → [Transform A | Transform B] → Merge → Store
```

## Usage

When the user wants to process data, use this pattern:

1. Understand the data transformation requirements
2. Design a pipeline with appropriate stages
3. Use parallel execution for independent operations
4. Add error handling with retry/fallback
5. Execute and return results